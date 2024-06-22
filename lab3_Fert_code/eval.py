import os
import json
import warnings
import numpy as np
import gym

from tqdm import tqdm

from env.sb3_wrapper import GymDssatWrapper
from gym_dssat_pdi.envs.utils import utils as dssat_utils
from model.PPO import PPO_agent
from model.base_agents import Expert_Agent, Null_Agent


warnings.simplefilter('ignore', UserWarning)

def dssat_evaluate(agent_name,
                   agent,
                   eval_episodes: int = 100) -> dict:
    if agent_name != 'Null':
        null_yield = np.load('env/null_data/null_yield_100.npy')
        null_N_uptake = np.load('env/null_data/null_N_uptake_100.npy')
    '''
    指标说明：
    score: gym_dssat累积rew
    total_fert: 各epi总施肥量, 在后续指标计算中记为F, kg/ha
    yield: 各epi产量, 在后续指标计算中记为Y, Y_0表示未施肥条件下的产量, kg/ha
    application_times: 各epi施肥次数
    grain_N_weight: 各epi收获部分N含量, 后续指标计算中记为U_H, kg/ha
    N_uptake: 各epi N吸收量, kg/ha, 后续指标计算中记为U, 未施肥条件下记为U_0
    PEP: 偏生产力, PEP = Y / F
    AE: 农学效率, AE = (Y-Y_0) / F
    PNB: 偏因子养分平衡, PNB = U_H / F
    RE: 表观回收率, (U - U_0) / F
    IE: 内在利用率, Y / U
    PE: 生理效率, (Y - Y_0) / (U - U_0)
    '''
    score_seen = []
    total_fert_seen = []
    application_times_seen = []
    yield_seen = []
    grain_N_weight_seen = []
    N_uptake_seen = []
    PEP_seen = []
    AE_seen = []
    PNB_seen = []
    RE_seen = []
    IE_seen = []
    PE_seen = []
    epi_fert_seen = []
    days_seen = []
    all_rain_seen = []
    all_ep_seen = []
    all_nstres_seen = []
    
    env_args = {
            'log_saving_path': f'./log/pdi_logs/eval_{agent_name}_dssat_pdi.log',
            'mode': 'fertilization',
            # 'mode': 'irrigation',
            'seed': 345,
            'random_weather': True,
            'cultivar': 'maize'
        }
    env = GymDssatWrapper(gym.make('gym_dssat_pdi:GymDssatPdi-v0', **env_args))
    
    if agent_name == 'Null':
        agent = Null_Agent(env)
    elif agent_name == 'Expert':
        agent = Expert_Agent(env)

    for i in tqdm(range(eval_episodes)):
        state = env.reset()
        if agent_name == 'Expert' or agent_name == 'HRL':
            agent.reset()
        last_dap = 0
        action_times = 0
        score = 0.
        reward = 0.
        grain_weight = 0.
        total_fert = 0.
        fert = 0.
        massic = 0.
        N_uptake = 0.
        pep = 0.
        ae = 0.
        pnb = 0.
        re = 0.
        ie = 0.
        pe = 0.
        done = False
        action_seen = []
        rain_seen = []
        nstres_seen = []
        ep_seen = []
        info = None
        while not done:
            if agent_name == 'self_PPO':
                action = agent.select_action(state)
            elif agent_name == "Expert" or agent_name == 'HRL':
                action, _ = agent.predict(info,state=state,deterministic=True)
            else:
                action, _ = agent.predict(state, deterministic=True)
            # print(action, type(action))
            state, reward, done, info = env.step(action)
            fert = info['cumsumfert'] - total_fert
            if fert > 1e-5:
                action_times += 1
            N_uptake += info['trnu']
            grain_weight = info['grnwt']
            total_fert = info['cumsumfert']
            massic = info['pcngrn']
            score += reward
            last_dap = info['dap']
            action_seen.append((info['dap'], fert))
            rain_seen.append((info['dap'], info['rain']))
            ep_seen.append((info['dap'], info['ep']))
            nstres_seen.append((info['dap'], info['nstres']))
        if agent_name != 'Null' and total_fert != 0:
                pep = grain_weight / total_fert
                ae = (grain_weight - null_yield[i]) / total_fert
                pnb = grain_weight * massic / total_fert
                re = (N_uptake - null_N_uptake[i]) / total_fert
                ie = grain_weight / N_uptake
                pe = (grain_weight - null_yield[i]) / (N_uptake - null_N_uptake[i]) if (N_uptake - null_N_uptake[i]) > 1e-5 else 0.

        PEP_seen.append(pep)
        AE_seen.append(ae)
        PNB_seen.append(pnb)
        RE_seen.append(re)
        IE_seen.append(ie)
        PE_seen.append(pe)
        score_seen.append(score)
        yield_seen.append(grain_weight)
        epi_fert_seen.append(action_seen)
        total_fert_seen.append(total_fert)
        grain_N_weight_seen.append(grain_weight * massic)
        application_times_seen.append(action_times)       
        N_uptake_seen.append(N_uptake)
        days_seen.append(last_dap)
        all_rain_seen.append(rain_seen)
        all_ep_seen.append(ep_seen)
        all_nstres_seen.append(nstres_seen)
        

    if agent_name == 'Null':
        null_yield = np.array(yield_seen, dtype=np.float32)
        null_N_uptake = np.array(N_uptake_seen, dtype=np.float32)
        np.save('env/null_data/null_yield_100.npy', null_yield)
        np.save('env/null_data/null_N_uptake_100.npy', null_N_uptake)
    
    result = {
        'agent_name': agent_name,
        'score': (np.mean(score_seen),np.std(score_seen)),
        'total_fert': (np.mean(total_fert_seen), np.std(total_fert_seen)),
        'application_times': (np.mean(application_times_seen), np.std(application_times_seen)),
        'yield': (np.mean(yield_seen), np.std(yield_seen)),
        'grain_N_weight': (np.mean(grain_N_weight_seen), np.std(grain_N_weight_seen)),
        'N_uptake': (np.mean(N_uptake_seen), np.std(N_uptake_seen)),
        'PEP': (np.mean(PEP_seen), np.std(PEP_seen)),
        'AE': (np.mean(AE_seen), np.std(AE_seen)),
        'PNB': (np.mean(PNB_seen), np.std(PNB_seen)),
        'RE': (np.mean(RE_seen), np.std(RE_seen)),
        'IE': (np.mean(IE_seen), np.std(IE_seen)),
        'PE': (np.mean(PE_seen), np.std(PE_seen)),
        'all_actions': epi_fert_seen,
        'all_days': days_seen,
        'all_nstres': all_nstres_seen,
        'all_rain': all_rain_seen,
        'all_ep': all_ep_seen,
        }
    env.close()
    return result

def dssat_result(result:dict,
                 agent_name:str,
                 result_csv_path:str = './log/result_csv/test_result.csv',
                 result_json_dir:str = './log/result_jsons/'):
    csv_exist_flag = os.path.exists(result_csv_path)
    with open(result_csv_path, 'a') as csv_file:
        if not csv_exist_flag:
            for key in result.keys():
                if key[:4] == 'all_':
                    continue
                print(key, end=',', file=csv_file)
            print(file=csv_file)
        for key in result.keys():
            if key[:4] == 'all_':
                continue
            if key == 'agent_name':
                print(result[key], end=',', file=csv_file)
                print(f"{key} = {result[key]}")
            else:
                print(f'{result[key][0]:.2f} +/- {result[key][1]:.2f}', end=',', file=csv_file)
                print(f'{key} = {result[key][0]:.2f} +/- {result[key][1]:.2f}')
        print(file=csv_file)
    json_path = os.path.join(result_json_dir, f"{agent_name}_test_result.json")
    with open(json_path, mode='w') as json_file:
        json.dump(result, json_file, indent=4)

    

if __name__ == '__main__':
    agents = {
        'Null':'Null',
        'Expert':'Expert',
        'PPO': PPO_agent(state_dim=11, action_dim=1, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, K_epochs=10, eps_clip=0.2, has_continuous_action_space=True, action_std_init=0.2),
    }
    if 'PPO' in agents.keys():
        agents['PPO'].load('output/PPO/PPO_123_0.pth')
    for agent_name in agents.keys():
        result = dssat_evaluate(agent_name, agents[agent_name], 100)
        dssat_result(result, agent_name)
 