from env import HYPER_PARAMS, network_config, CustomEnv, View
from dqn import CustomEnvWrapper, make_env, Networks

import os
import argparse
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt

from torch import device, cuda

CONFIG = "platoon_simplified"

class Observe(View):
    def __init__(self, args):
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

        super(Observe, self).__init__(type(self).__name__.upper(),
                                      make_env(
                                          env=CustomEnvWrapper(CustomEnv(type(self).__name__.lower())),
                                          max_episode_steps=args.max_s)
                                      )

        model_pack = args.d.split('/')[-1].split('_model.pack')[0]

        self.network = getattr(Networks, {
            "DQNAgent": "DeepQNetwork",
            "DoubleDQNAgent": "DeepQNetwork",
            "DuelingDoubleDQNAgent": "DuelingDeepQNetwork",
            "PerDuelingDoubleDQNAgent": "DuelingDeepQNetwork"
        }[model_pack.split('_lr')[0]])(
            device(("cuda:" + args.gpu) if cuda.is_available() else "cpu"),
            float(model_pack.split('_lr')[1].split('_')[0]),
            network_config,
            self.env.observation_space,
            self.env.action_space.n
        )

        self.network.load(args.d)

        self.obs = np.zeros(self.env.observation_space.shape, dtype=np.float32)

        self.repeat = 0
        self.action = 0
        self.ep = 0

        print()
        print("OBSERVE")
        print()
        [print(arg, "=", getattr(args, arg)) for arg in vars(args)]

        self.max_episodes = args.max_e
        self.log = (args.log, args.log_s, args.log_dir + model_pack)

        self.evaluation = {}

    def setup(self):
        self.obs = self.env.reset()
        print(self.obs)

    def loop(self):
        self.action = []
        if self.repeat % (HYPER_PARAMS['repeat'] or 1) == 0:
            for o in self.obs:
                a = self.network.actions([o.tolist()])[0]
                self.action.append(a)

        self.repeat += 1

        self.obs, _, done, info = self.env.step(self.action)
        # print(self.obs, self.action, done)
        self.env.log_info_writer(info, done, *self.log)
        if self.env.custom_env.sumo_env.s_step >= 17000:
            veh, cav, platoon, processus, danger_try, faillure, collison, processus_lance, platoon_edges, n_wrong = self.env.custom_env.sumo_env.n_veh_simulation, \
                                           self.env.custom_env.sumo_env.n_cav_simulation, \
                                           self.env.custom_env.sumo_env.n_platoon_simulation, \
                                           self.env.custom_env.sumo_env.n_processus_simulation,\
                                           self.env.custom_env.sumo_env.danger_try_simulation,\
                                           self.env.custom_env.sumo_env.n_faillure_simulation,\
                                           self.env.custom_env.sumo_env.n_collison_simulation, \
                                           self.env.custom_env.sumo_env.n_processus_summary, \
                                           self.env.custom_env.sumo_env.platoons_edge, \
                                           self.env.custom_env.sumo_env.n_wrong_request
            key = str(self.env.custom_env.sumo_env.m_flow[0]) + '_' + str(self.env.custom_env.sumo_env.p_flow) + '_' + str(self.env.custom_env.sumo_env.cav_p_rate)
            if key in self.evaluation.keys():
                self.evaluation[key].append([veh, cav, platoon, processus, danger_try, faillure, collison, processus_lance,
                                        platoon_edges, n_wrong])
            else:
                self.evaluation[key] = [veh, cav, platoon, processus, danger_try, faillure, collison, processus_lance, platoon_edges, n_wrong]
            if len(self.evaluation) < 96:
                self.setup()
            with open('evaluation.pkl', 'wb') as f:
                pickle.dump(self.evaluation, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OBSERVE")
    str2bool = (lambda v: v.lower() in ("yes", "y", "true", "t", "1"))
    parser.add_argument('-d', type=str, default='./save/' + CONFIG + "/PerDuelingDoubleDQNAgent_lr0.0002_model.pack", help='Directory')
    parser.add_argument('-gpu', type=str, default='0', help='GPU #')
    parser.add_argument('-max_s', type=int, default=0, help='Max steps per episode if > 0, else inf')
    parser.add_argument('-max_e', type=int, default=0, help='Max episodes if > 0, else inf')
    parser.add_argument('-log', type=str2bool, default=False, help='Log csv to ./logs/test/')
    parser.add_argument('-log_s', type=int, default=0, help='Log step if > 0, else episode')
    parser.add_argument('-log_dir', type=str, default="./logs/test/", help='Log directory')
    random.seed(42)

    Observe(parser.parse_args()).run()
