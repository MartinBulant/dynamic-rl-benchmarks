import numpy as np

from gymnasium import spaces
from PIL import Image, ImageDraw
from dyn_rl_benchmarks.envs.goal_env import GoalEnv


class DrawbridgeEnv(GoalEnv):
    metadata = {"render_modes": ["rgb_array", "human"]}

    _max_vel = 0.1
    max_episode_length = 1000
    _goal_radius = 1.0
    _river_length = 10.0
    _goal = _river_length - _goal_radius
    _drawbridge_start = 500.0
    _unfurl_speed = 0.03
    _sail_drag = 0.0001
    _max_vel = 0.03

    def __init__(self, subgoal_radius=0.05):
        super().__init__()

        desired_goal_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        achieved_goal_space = desired_goal_space

        box_space_1d = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        obs_space = spaces.Dict(
            {
                "ship_pos": box_space_1d,
                "ship_vel": box_space_1d,
                "sails_unfurled": box_space_1d,
                "bridge_phase": box_space_1d,
            }
        )

        self.observation_space = spaces.Dict(
            {
                "observation": obs_space,
                "desired_goal": desired_goal_space,
                "achieved_goal": achieved_goal_space,
            }
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.window = None

        self.window_width = 1920
        self.window_height = 1080
        self.background_color = (1.0, 1.0, 1.0, 1.0)
        self._position = (5.0, 3.0, -13.0)
        self._lookat = (0.0, 0.0, -7.3, 0.0)

        self.current_step = 0

        self._subgoals = []
        self._timed_subgoals = []
        self._tolerances = []
        self.subgoal_radius = float(subgoal_radius)

        self._env_geoms = ["riverbank", "water", "bridge_base", "underground"]
        self._n_sails = 8

    def compute_reward(self, achieved_goal, desired_goal, info):
        if (
            abs(achieved_goal[0] - desired_goal[0])
            <= self._goal_radius / self._river_length
        ):
            return 0.0
        else:
            return -1.0

    def _get_obs(self):
        partial_obs = {
            "ship_pos": [2.0 * self.ship_pos / self._river_length - 1.0],
            "ship_vel": [self.ship_vel / self._max_vel],
            "sails_unfurled": [self.sails_unfurled * 2.0 - 1.0],
            "bridge_phase": [2.0 * self.current_step / self.max_episode_length - 1.0],
        }
        obs = {
            "observation": partial_obs,
            "desired_goal": [2.0 * self._goal / self._river_length - 1.0],
            "achieved_goal": partial_obs["ship_pos"],
        }
        return obs

    def step(self, action):
        self.sails_unfurled += action[0] * self._unfurl_speed
        self.sails_unfurled = np.clip(self.sails_unfurled, 0.0, 1.0)
        self.ship_pos += self.ship_vel
        self.ship_pos = np.clip(self.ship_pos, 0.0, self._river_length)
        self.ship_vel += self._sail_drag * self.sails_unfurled
        self.ship_vel = np.clip(self.ship_vel, -self._max_vel, self._max_vel)
        if self._get_drawbridge_angle() > -25.0 and self.ship_pos > 4.3:
            if self.ship_vel > 0.0:
                self.ship_vel = -0.1 * self.ship_vel
        obs = self._get_obs()
        info = {}
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        self.current_step += 1

        terminated = reward == 0.0
        truncated = self.current_step >= self.max_episode_length
        return obs, reward, terminated, truncated, info

    def _get_drawbridge_angle(self):
        return -min(max(self.current_step - self._drawbridge_start, 0.0) * 0.4, 90.0)

    def reset(self, seed=None, options=None):
        self.ship_pos = 0.0
        self.ship_vel = 0.0
        self.sails_unfurled = 0.0
        self.current_step = 0
        return self._get_obs(), {}

    def update_subgoals(self, subgoals):
        self._subgoals = subgoals

    def update_timed_subgoals(self, timed_subgoals, tolerances):
        self._timed_subgoals = timed_subgoals
        self._tolerances = tolerances

    def render(self):

        W, H = 1200, 400
        img = Image.new("RGB", (W, H), color=(168, 216, 234))
        draw = ImageDraw.Draw(img)

        def to_px(x, y):
            """Převod souřadnic prostředí na pixely."""
            px = int((x + 0.5) / (self._river_length + 1.0) * W)
            py = int((1.0 - (y + 1.5) / 4.5) * H)
            return px, py

        x0, y0 = to_px(-0.5, -0.5)
        x1, y1 = to_px(self._river_length + 0.5, -1.5)
        draw.rectangle([x0, y0, x1, y1], fill=(139, 105, 20))

        angle = self._get_drawbridge_angle()
        bridge_x = self._river_length / 2.0
        if angle > -90.0:
            draw.line(
                [to_px(bridge_x - 0.6, 0.0), to_px(bridge_x, 0.0)],
                fill=(92, 64, 51),
                width=6,
            )
            draw.line(
                [to_px(bridge_x, 0.0), to_px(bridge_x + 0.6, 0.0)],
                fill=(92, 64, 51),
                width=6,
            )
        else:
            draw.line(
                [to_px(bridge_x - 0.6, 0.0), to_px(bridge_x, 0.6)],
                fill=(92, 64, 51),
                width=6,
            )
            draw.line(
                [to_px(bridge_x, 0.6), to_px(bridge_x + 0.6, 0.0)],
                fill=(92, 64, 51),
                width=6,
            )

        sx0, sy0 = to_px(self.ship_pos - 0.3, 0.0)
        sx1, sy1 = to_px(self.ship_pos + 0.3, -0.5)
        draw.rectangle([sx0, sy0, sx1, sy1], fill=(128, 128, 128))

        sail_height = self.sails_unfurled * 1.5
        if sail_height > 0.01:
            px0, py0 = to_px(self.ship_pos - 0.05, sail_height)
            px1, py1 = to_px(self.ship_pos + 0.05, 0.0)
            draw.rectangle([px0, py0, px1, py1], fill=(221, 221, 221))

        gx0, gy0 = to_px(self._goal - self._goal_radius, 0.0)
        gx1, gy1 = to_px(self._goal + self._goal_radius, -0.5)
        draw.rectangle([gx0, gy0, gx1, gy1], fill=(0, 200, 0, 80))
        gx = to_px(self._goal, 0.0)[0]
        draw.line([(gx, 0), (gx, H)], fill=(0, 180, 0), width=3)

        colors_rgb = [(255, 100, 0), (0, 100, 255), (200, 0, 200)]
        for i, sg in enumerate(self._subgoals):
            if sg is not None:
                pos = (sg["ship_pos"] + 1.0) * 0.5 * self._river_length
                px = to_px(pos, 0.0)[0]
                color = colors_rgb[i % len(colors_rgb)]
                draw.line([(px, 0), (px, H)], fill=color, width=2)

        draw.text(
            (10, 10),
            f"step={self.current_step} | ship_pos={self.ship_pos:.2f} | bridge={angle:.1f}°",
            fill=(0, 0, 0),
        )

        return np.array(img)
