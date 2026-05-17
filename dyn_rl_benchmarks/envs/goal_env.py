import gymnasium


class GoalEnv(gymnasium.Env):
    def compute_reward(self, achieved_goal, desired_goal, info):
        raise NotImplementedError

    def compute_terminated(self, achieved_goal, desired_goal, info):
        raise NotImplementedError

    def compute_truncated(self, achieved_goal, desired_goal, info):
        raise NotImplementedError
