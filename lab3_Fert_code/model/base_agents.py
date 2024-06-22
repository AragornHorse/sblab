import numpy as np

from env.sb3_wrapper import Formator


class Null_Agent:
    def __init__(self, env) -> None:
        self.env = env
        self.action_formator = Formator(env.env.env)
    
    def predict(self, obs, state=None, episode_start=None, deterministic=None):
        normalized_action = self.action_formator.normalize_actions([0])
        return np.array(normalized_action, dtype=np.float32), None


class Expert_Agent:
    def __init__(self, env):
        self.env = env
        self.action_formator = Formator(env.env.env)
        assert 'dap' in env.observation_variables
        all_policy_dic = {
            'fertilization': {
                40: 27,
                45: 35,
                80: 54,
            },
        }
        self.policy_dic = all_policy_dic[self.env.mode]
        self.zero_day_flag = True

    def _policy(self, info):
        if info == None:
            return [0]
        dap = int(info['dap'])
        if 0 in self.policy_dic.keys():
            if dap == 0 and self.zero_day_flag:
                action = [self.policy_dic[dap]]
                self.zero_day_flag = False
            elif dap == 0 and not self.zero_day_flag:
                action = [0]
            else:
                action = [self.policy_dic[dap] if dap in self.policy_dic.keys() else 0]
        else:
            action = [self.policy_dic[dap] if dap in self.policy_dic.keys() else 0]
        return action

    def reset(self):
        self.zero_day_flag = True
    
    def predict(self, info, state=None, episode_start=None, deterministic=None):
        action = self._policy(info)
        normalized_action = self.action_formator.normalize_actions(action)
        return np.array(normalized_action, dtype=np.float32), None