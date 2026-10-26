# Adapted from Isaac Lab's stock quadcopter_env.py (direct RL workflow).
# Gate-racing extension -- built against the REAL file structure Thea pasted,
# not a guess. Key things carried over exactly as-is from the original:
#   - permanent_wrench_composer.set_forces_and_torques for _apply_action
#   - projected_gravity_b in the observation (not raw quaternion)
#   - tanh-mapped distance reward, step_dt scaling, per-term episode logging
#   - full _reset_idx pattern (env_ids None-check, _ALL_INDICES, logging block)
#
# Gate-racing additions:
#   - Randomized serpentine gate SEQUENCE per episode (not one static goal)
#   - Strict-order pass detection (position + heading, gate plane crossing)
#   - Two-tier elevation (ground-level vs "elevated"/container-height gates)
#   - Reward: same lin_vel/ang_vel smoothness terms as stock + distance-to-
#     CURRENT-gate (replacing distance-to-fixed-goal) + gate-pass bonus
#
# Hardware context to keep consistent: CUAV X25 EVO (FC) + Intel RealSense
# (camera) + Jetson Orin Nano (companion) + 7-inch frame.

from __future__ import annotations

import gymnasium as gym
import os
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, subtract_frame_transforms

from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


class GateCourseRandomizer:
    def __init__(
        self,
        num_envs: int,
        device: str,
        num_gates_range=(4, 6),
        gate_size_range=(0.8, 1.5),
        segment_length_range=(3.0, 6.0),
        lateral_swing_range=(0.5, 1.5),
        ground_height_range=(1.5, 2.0),
        elevated_height_range=(2.4, 3.0),
        elevated_gate_fraction=0.25,
    ):
        self.num_envs = num_envs
        self.device = device
        self.num_gates_range = num_gates_range
        self.gate_size_range = gate_size_range
        self.segment_length_range = segment_length_range
        self.lateral_swing_range = lateral_swing_range
        self.ground_height_range = ground_height_range
        self.elevated_height_range = elevated_height_range
        self.elevated_gate_fraction = elevated_gate_fraction
        self.max_gates = num_gates_range[1]

    def sample_course(self, env_ids: torch.Tensor):
        n = len(env_ids)
        device = self.device
        max_gates = self.max_gates

        num_gates = torch.randint(
            self.num_gates_range[0], self.num_gates_range[1] + 1, (n,), device=device
        )

        gate_positions = torch.zeros(n, max_gates, 3, device=device)
        gate_headings = torch.zeros(n, max_gates, device=device)
        is_elevated = torch.zeros(n, max_gates, dtype=torch.bool, device=device)

        forward_pos = torch.zeros(n, device=device)
        lateral_sign = torch.ones(n, device=device)

        for i in range(max_gates):
            seg_len = torch.empty(n, device=device).uniform_(*self.segment_length_range)
            swing = torch.empty(n, device=device).uniform_(*self.lateral_swing_range)

            forward_pos = forward_pos + seg_len
            lateral_pos = lateral_sign * swing
            lateral_sign = -lateral_sign

            gate_positions[:, i, 0] = forward_pos
            gate_positions[:, i, 1] = lateral_pos

            elevated_roll = torch.rand(n, device=device) < self.elevated_gate_fraction
            is_elevated[:, i] = elevated_roll
            ground_h = torch.empty(n, device=device).uniform_(*self.ground_height_range)
            elevated_h = torch.empty(n, device=device).uniform_(*self.elevated_height_range)
            gate_positions[:, i, 2] = torch.where(elevated_roll, elevated_h, ground_h)

        # FIX (Ben): gate faces the BISECTOR of incoming + outgoing direction.
        # Before, it faced only the NEXT gate, so on a serpentine the drone often
        # had to cross the gate plane at a steep angle (the "hairpin" problem).
        def _unit(v):
            return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        start_xy = torch.zeros(n, 2, device=device)
        for i in range(max_gates):
            prev_xy = start_xy if i == 0 else gate_positions[:, i - 1, :2]
            here_xy = gate_positions[:, i, :2]
            incoming = _unit(here_xy - prev_xy)
            if i < max_gates - 1:
                outgoing = _unit(gate_positions[:, i + 1, :2] - here_xy)
                is_last = (i == num_gates - 1).unsqueeze(-1)
                outgoing = torch.where(is_last, incoming, outgoing)  # last real gate: no next gate
            else:
                outgoing = incoming
            direction = _unit(incoming + outgoing)
            gate_headings[:, i] = torch.atan2(direction[:, 1], direction[:, 0])

        gate_size = torch.empty(n, device=device).uniform_(*self.gate_size_range)

        return {
            "num_gates": num_gates,
            "gate_positions": gate_positions,
            "gate_headings": gate_headings,
            "gate_size": gate_size,
            "is_elevated": is_elevated,
        }


def check_gate_passed(drone_pos, drone_prev_pos, gate_pos, gate_heading, gate_size):
    """Strict-order pass check: crossed the current gate's plane, within bounds."""
    to_drone = drone_pos[:, :2] - gate_pos[:, :2]
    to_drone_prev = drone_prev_pos[:, :2] - gate_pos[:, :2]

    normal = torch.stack([torch.cos(gate_heading), torch.sin(gate_heading)], dim=-1)

    dist_now = (to_drone * normal).sum(-1)
    dist_prev = (to_drone_prev * normal).sum(-1)
    crossed_plane = (dist_prev < 0) & (dist_now >= 0)

    lateral = to_drone - dist_now.unsqueeze(-1) * normal
    within_bounds = lateral.norm(dim=-1) < (gate_size / 2)

    z_diff = (drone_pos[:, 2] - gate_pos[:, 2]).abs()
    within_height = z_diff < (gate_size / 2)

    offset_norm = torch.maximum(
        lateral.norm(dim=-1) / (gate_size / 2),
        z_diff / (gate_size / 2),
    )
    return (crossed_plane & within_bounds & within_height), offset_norm


class QuadcopterGateRaceEnvWindow(BaseEnvWindow):
    """Window manager for the gate-race environment."""

    def __init__(self, env: QuadcopterGateRaceEnv, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterGateRaceEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0  # longer than stock (10s) -- multi-gate course needs more time
    decimation = 2
    action_space = 4
    observation_space = 19  # NEW (Ben v4): +own height (1) +next gate position (3); was  # stock 12, with desired_pos_b swapped for gate_pos_b (3) + gate_heading sin/cos (2) + gate_idx_norm (1)
    state_space = 0
    debug_vis = True

    # NEW (Ben): camera follows drone of env 0 -> used by the GUI AND recorded videos
    viewer: ViewerCfg = ViewerCfg(
        eye=(-4.0, -2.0, 2.5),
        lookat=(2.0, 0.0, 0.0),
        origin_type="asset_root",
        asset_name="robot",
        env_index=0,
    )

    ui_window_class_type = QuadcopterGateRaceEnvWindow

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=40.0,  # was 2.5: courses are up to ~36 m long, so envs overlapped visually
        replicate_physics=True, clone_in_fabric=True
    )

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(
    prim_path="/World/envs/env_.*/Robot",
    spawn=CRAZYFLIE_CFG.spawn.replace(
        mass_props=sim_utils.MassPropertiesCfg(
            mass=1.3,  # ESTIMATE - see hardware doc, pending mentor confirmation
        ),
    ),
)
    thrust_to_weight = 3.2
    moment_scale = 0.02

    # reward scales -- lin_vel/ang_vel kept identical to stock (smoothness);
    # distance_to_goal replaced conceptually by distance_to_gate
    lin_vel_reward_scale = -0.01  # Ben v6: was -0.05
    ang_vel_reward_scale = -0.05  # Ben v7: was -0.01, smoother turns (less overshoot/loops)
    spawn_height = 1.2  # Ben v7: start higher (was 0.5 m, too close to the 0.1 m floor limit)
    distance_to_gate_reward_scale = 15.0
    action_penalty_scale = 0.0
    centering_weight = 0.6  # Ben v6: centre pass = full bonus
    gate_pass_bonus = 60.0
    crash_penalty = -150.0  # NEW (Ben v2): crashing used to cost nothing
    alive_reward_scale = -1.0  # Ben v6: now a clock, -1 per second (was +2)

    # gate course randomization params
    num_gates_range = (4, 6)
    gate_size_range = (0.8, 1.5)
    segment_length_range = (3.0, 6.0)
    lateral_swing_range = (0.3, 1.0)  # NEW (Ben v4): gentler zigzag to start (was 0.5-1.5)
    ground_height_range = (1.5, 2.0)
    elevated_height_range = (2.4, 3.0)


class QuadcopterGateRaceEnv(DirectRLEnv):
    cfg: QuadcopterGateRaceEnvCfg

    def __init__(self, cfg: QuadcopterGateRaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self._randomizer = GateCourseRandomizer(
            num_envs=self.num_envs,
            device=self.device,
            num_gates_range=self.cfg.num_gates_range,
            gate_size_range=self.cfg.gate_size_range,
            segment_length_range=self.cfg.segment_length_range,
            lateral_swing_range=self.cfg.lateral_swing_range,
            ground_height_range=self.cfg.ground_height_range,
            elevated_height_range=self.cfg.elevated_height_range,
        )

        max_gates = self.cfg.num_gates_range[1]
        self._num_gates = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._gate_positions_w = torch.zeros(self.num_envs, max_gates, 3, device=self.device)
        self._gate_headings = torch.zeros(self.num_envs, max_gates, device=self.device)
        self._gate_size = torch.zeros(self.num_envs, device=self.device)
        self._is_elevated = torch.zeros(self.num_envs, max_gates, dtype=torch.bool, device=self.device)
        self._current_gate_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._prev_root_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_gate",
                "gate_pass_bonus",
                "action_penalty",
                "crash",
                "alive",
            ]
        }

        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        self.set_debug_vis(self.cfg.debug_vis)

        # NEW (Ben v3): flight recorder for env 0, only on when DRONE_LOG_DIR is set
        self._log_dir = os.environ.get("DRONE_LOG_DIR")
        self._log_traj = []
        self._log_saved = 0
        if self._log_dir:
            os.makedirs(self._log_dir, exist_ok=True)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # NOTE: gate meshes intentionally not spawned yet -- pure state-based
        # training doesn't need them visible. Add cuboid-frame gate meshes
        # here once Phase 2 (vision) needs the RealSense feed to see them.

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

    def _current_gate_pos_heading(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        idx = self._current_gate_idx.clamp(max=self.cfg.num_gates_range[1] - 1)
        pos = self._gate_positions_w[env_ids, idx]
        heading = self._gate_headings[env_ids, idx]
        return pos, heading

    def _get_observations(self) -> dict:
        gate_pos_w, gate_heading = self._current_gate_pos_heading()
        gate_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, gate_pos_w
        )
        gate_idx_norm = (
            self._current_gate_idx.float() / max(self.cfg.num_gates_range[1] - 1, 1)
        ).unsqueeze(-1)

        # FIX (Ben): gate direction was given in WORLD frame, but the policy never
        # sees its own yaw (projected_gravity has no yaw info), so world cos/sin
        # was meaningless to it. Rotate the gate normal into the BODY frame.
        gate_normal_w = torch.stack(
            [torch.cos(gate_heading), torch.sin(gate_heading), torch.zeros_like(gate_heading)], dim=-1
        )
        gate_normal_b = quat_apply_inverse(self._robot.data.root_quat_w, gate_normal_w)

        # NEW (Ben v4): show the drone the NEXT gate (to plan turns) and its own height (to see the floor)
        all_ids = torch.arange(self.num_envs, device=self.device)
        next_idx = (self._current_gate_idx + 1).clamp(max=self.cfg.num_gates_range[1] - 1)
        next_gate_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._gate_positions_w[all_ids, next_idx]
        )
        height = self._robot.data.root_pos_w[:, 2:3]

        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                gate_pos_b,
                gate_normal_b[:, :2],  # same 2 slots as before -> observation_space stays 15
                gate_idx_norm,
                next_gate_pos_b,  # NEW (Ben v4)
                height,  # NEW (Ben v4)
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        gate_pos_w, gate_heading = self._current_gate_pos_heading()
        root_pos_w = self._robot.data.root_pos_w

        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)

        distance_to_gate = torch.linalg.norm(gate_pos_w - root_pos_w, dim=1)
        prev_distance_to_gate = torch.linalg.norm(gate_pos_w - self._prev_root_pos_w, dim=1)
        progress_to_gate = prev_distance_to_gate - distance_to_gate

        just_passed, offset_norm = check_gate_passed(
            root_pos_w, self._prev_root_pos_w, gate_pos_w, gate_heading, self._gate_size
        )
        centering_quality = (1.0 - self.cfg.centering_weight * offset_norm).clamp(min=0.0)
        gate_pass_bonus = just_passed.float() * self.cfg.gate_pass_bonus * centering_quality

        self._current_gate_idx = torch.where(
            just_passed, self._current_gate_idx + 1, self._current_gate_idx
        )

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_gate": progress_to_gate * self.cfg.distance_to_gate_reward_scale,
            "action_penalty": torch.sum(self._actions[:, 1:] ** 2, dim=1) * self.cfg.action_penalty_scale,
            "gate_pass_bonus": gate_pass_bonus,
            # NEW (Ben v2): one-time penalty on the step the drone hits floor/ceiling
            "alive": torch.full((self.num_envs,), self.cfg.alive_reward_scale * self.step_dt, device=self.device),  # NEW (Ben v4)
            "crash": (self._term_floor | self._term_ceiling).float() * self.cfg.crash_penalty
            if hasattr(self, "_term_floor")
            else torch.zeros(self.num_envs, device=self.device),
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for key, value in rewards.items():
            self._episode_sums[key] += value

        if self._log_dir and self._log_saved < 20:  # NEW (Ben v3): record env 0 each step
            p = (root_pos_w[0] - self._terrain.env_origins[0]).detach().cpu().numpy()
            self._log_traj.append([p[0], p[1], p[2], float(self._current_gate_idx[0])])

        self._prev_root_pos_w = root_pos_w.clone()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self._term_floor = self._robot.data.root_pos_w[:, 2] < 0.1
        self._term_ceiling = self._robot.data.root_pos_w[:, 2] > 5.0
        crashed = torch.logical_or(self._term_floor, self._term_ceiling)
        self._term_finished = self._current_gate_idx >= self._num_gates
        died = torch.logical_or(crashed, self._term_finished)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        final_distance_to_gate = torch.linalg.norm(
            self._gate_positions_w[env_ids, self._current_gate_idx[env_ids].clamp(max=self.cfg.num_gates_range[1] - 1)] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_gate"] = final_distance_to_gate.item()
        extras["Metrics/gates_passed"] = torch.mean(self._current_gate_idx[env_ids].float()).item()
        if hasattr(self, "_term_floor"):
            extras["Metrics/term_floor"] = torch.count_nonzero(self._term_floor[env_ids]).item()
            extras["Metrics/term_ceiling"] = torch.count_nonzero(self._term_ceiling[env_ids]).item()
            extras["Metrics/term_finished"] = torch.count_nonzero(self._term_finished[env_ids]).item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0

        # NEW (Ben v3): save env 0's finished episode before a new course is drawn
        if self._log_dir and len(self._log_traj) > 1 and bool((env_ids == 0).any()):
            n = int(self._num_gates[0])
            origin = self._terrain.env_origins[0]
            crashed = bool(self._term_floor[0] | self._term_ceiling[0]) if hasattr(self, "_term_floor") else False
            np.savez(
                os.path.join(self._log_dir, f"episode_{self._log_saved:03d}.npz"),
                traj=np.array(self._log_traj, dtype=np.float32),
                gates=(self._gate_positions_w[0, :n] - origin).cpu().numpy(),
                headings=self._gate_headings[0, :n].cpu().numpy(),
                gate_size=float(self._gate_size[0]),
                gates_passed=int(self._current_gate_idx[0]),
                crashed=crashed,
                dt=float(self.step_dt),
            )
            print(f"[flight recorder] saved episode {self._log_saved}: {int(self._current_gate_idx[0])}/{n} gates, crashed={crashed}")
            self._log_saved += 1
            self._log_traj = []

        course = self._randomizer.sample_course(env_ids)
        self._num_gates[env_ids] = course["num_gates"]
        self._gate_positions_w[env_ids] = course["gate_positions"] + self._terrain.env_origins[env_ids].unsqueeze(1)
        self._gate_headings[env_ids] = course["gate_headings"]
        self._gate_size[env_ids] = course["gate_size"]
        self._is_elevated[env_ids] = course["is_elevated"]
        self._current_gate_idx[env_ids] = 0

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 2] = self._terrain.env_origins[env_ids, 2] + self.cfg.spawn_height  # Ben v7
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._prev_root_pos_w[env_ids] = default_root_state[:, :3]

    def _set_debug_vis_impl(self, debug_vis: bool):
        # NEW (Ben): draw every gate as a see-through panel.
        # green = gate it's flying to now, orange = upcoming, grey = already passed
        if debug_vis:
            if not hasattr(self, "gate_visualizer"):
                def _panel(color, opacity):
                    return sim_utils.CuboidCfg(
                        size=(0.05, 1.0, 1.0),  # thin along gate normal; scaled by gate_size at draw time
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, opacity=opacity),
                    )

                gate_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Gates",
                    markers={
                        "current": _panel((0.1, 1.0, 0.2), 0.6),
                        "upcoming": _panel((1.0, 0.5, 0.1), 0.35),
                        "passed": _panel((0.5, 0.5, 0.5), 0.15),
                    },
                )
                self.gate_visualizer = VisualizationMarkers(gate_cfg)
            self.gate_visualizer.set_visibility(True)
        else:
            if hasattr(self, "gate_visualizer"):
                self.gate_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        n, g = self._gate_positions_w.shape[:2]
        gate_ids = torch.arange(g, device=self.device).unsqueeze(0).expand(n, g)

        pos = self._gate_positions_w.clone()
        exists = gate_ids < self._num_gates.unsqueeze(-1)
        pos[~exists] = torch.tensor([0.0, 0.0, -100.0], device=self.device)  # hide unused gate slots underground

        yaw = self._gate_headings.reshape(-1)
        zeros = torch.zeros_like(yaw)
        quat = quat_from_euler_xyz(zeros, zeros, yaw)

        size = self._gate_size.unsqueeze(-1).expand(n, g).reshape(-1)
        scales = torch.stack([torch.ones_like(size), size, size], dim=-1)

        cur = self._current_gate_idx.unsqueeze(-1)
        idx = torch.full((n, g), 1, dtype=torch.long, device=self.device)  # upcoming
        idx[gate_ids == cur] = 0  # current
        idx[gate_ids < cur] = 2  # passed

        self.gate_visualizer.visualize(
            translations=pos.reshape(-1, 3),
            orientations=quat,
            scales=scales,
            marker_indices=idx.reshape(-1),
        )
