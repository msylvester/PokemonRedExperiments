class RewardManager:
    def __init__(self, env):
        self.env = env
        self.total_reward = 0

    def compute_reward(self):
        battle_reward = 100 if self.env.emulator.read_memory(0xD057) else 0
        dir_reward = self.env.dir_reward()
        step_reward = (dir_reward + battle_reward) - self.total_reward
        self.total_reward += step_reward
        return step_reward

    def dir_reward(self):
        target_position = np.array([338, 94])
        current_position = np.array(self.env.get_global_coords())
        distance = np.linalg.norm(current_position - target_position)
        return max(10 - distance * 0.1, 0)
