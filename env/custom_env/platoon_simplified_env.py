# https://sumo.dlr.de/pydoc/

from __future__ import absolute_import, print_function

from .utils import SUMO_PARAMS

import os
import sys
import json
import random
import math
import copy
import pickle
import numpy as np
import xml.etree.ElementTree as ET
# https://github.com/michele-segata/plexe-pyapi
from plexe import Plexe, ACC, CACC, DRIVER, FAKED_CACC, SPEED, POS_X, POS_Y

SUMO_HOME = "/usr/share/sumo/"

sys.path.append(SUMO_HOME + 'tools')

from sumolib import net  # noqa
import traci  # noqa

VAR_SPEED = traci.constants.VAR_SPEED
VAR_POSITION = traci.constants.VAR_POSITION
MIM_GAP = 10
MIM_SPEED = 15
MAX_SPEED = 33
class PlatoonEnvSimp:
    SUMO_ENV = "./env/custom_env/"

    @staticmethod
    def pretty_print(d):
        print(json.dumps(d, sort_keys=True, indent=4))

    @staticmethod
    def arg_max(_list):
        return max(range(len(_list)), key=lambda i: _list[i])

    @staticmethod
    def arg_min(_list):
        return min(range(len(_list)), key=lambda i: _list[i])

    @staticmethod
    def clip(min_clip, max_clip, x):
        return max(min_clip, min([max_clip, x])) if min_clip < max_clip else x

    def __init__(self, gui=False, log=True, rnd=(False, False)):
        self.args = SUMO_PARAMS

        self.config = self.args["config"]
        self.multi_joiners = self.args["multi_joiners"]
        self.data_dir = self.SUMO_ENV + "data/" + self.config + "/"

        # Here is our network
        self.net = net.readNet(self.data_dir + self.config + ".net.xml")

        self.action_space_n = 6
        self.observation_space_n = 14


        self.gui = False
        self.log = log
        self.rnd = rnd

        traci.start(self.set_params())

        self.path = './env/custom_env/data/platoon_simplified/'
        self.routes = self.get_all_routes()
        self.p_lane = 1

        """=====================================Customer set========================================="""
        self.LANE_PLATOON_INDEX = 1
        # vehicle length
        self.LENGTH = 4
        # inter-vehicle distance
        self.DISTANCE = 5
        # cruising speed
        self.SPEED = 100 / 3.6
        self.SPEED_CAV_LANES = 120 / 3.6
        # states for joining
        self.REQUEST = 0
        self.CATCHING_PLATOON = 1
        self.OPENING_GAP = 2
        self.SPLITTING_PLATOON = 3
        self.LEAVING_PLATOON = 4
        self.CHANGING_LANE = 5
        self.COMPLETED = 0
        self.LANE_CHANGE_DURATION = 0
        self.choose_position_method = "last"
        self.choose_veh_method = "random"

        self.simulation_duration = 1800
        self.warmup_duration = 120
        self.step_length = 0.1
        self.join_distance = 10
        # self.leave_distance = 10
        self.join_speed = self.SPEED
        self.detector_1 = "D1"
        self.detector_2 = "D2"
        self.plexe = Plexe()
        self.s_step = 0
        # topology: a dictionary pointing each vehicle id to its front
        # vehicle and platoon leader. each entry of the dictionary is a dictionary
        # which includes the keys "leader" and "front"
        self.topology = {}
        self.platoons = {}
        self.leave_list = []
        self.join_list = []
        self.merge_list = []
        self.buffer_list = []
        self.neighbors = []
        self.tlp_list = []
        self.obs_done = {}
        self.edges = ["gneE1", "gneE2", "gneE3", "gneE4", "gneE6", "gneE7", "gneE8", "gneE9", "gneE10",
                      "gneE11", "gneE12", "gneE14", "gneE15", "gneE16", "gneE17", "gneE19", "gneE20",
                      "gneE21", "gneE22", "gneE23", "gneE24", "gneE12", "gneE26", "gneE27", "gneE28", "gneE29", "gneE30",
                      "gneE31", "gneE32", "gneE34", "gneE35", "gneE36", "gneE37", "gneE38", "gneE39"]
        # self.route_edge = {"off0": "gneE3", "off1": "gneE11", "off2": "gneE16"}
        # self.pass_list = {"off0": [], "off1": [], "off2": []}
        '''self.edge_m = ["gneE1", "gneE2", "gneE6", "gneE7", "gneE8", "gneE9", "gneE10",
                       "gneE14", "gneE15", "gneE19", "gneE20", "gneE21", "gneE22", "gneE23"]'''
        self.edge_m = ["gneE1", "gneE2", "gneE3", "gneE4", "gneE6", "gneE7", "gneE8", "gneE9", "gneE10",
                      "gneE11", "gneE12", "gneE14", "gneE15", "gneE16", "gneE17", "gneE19", "gneE20",
                      "gneE21", "gneE22", "gneE23", "gneE24", "gneE12", "gneE26", "gneE27", "gneE28", "gneE29", "gneE30",
                      "gneE31", "gneE32", "gneE34", "gneE35", "gneE36", "gneE37", "gneE38", "gneE39"]
        self.max_try = 5
        self.n_failure = 0
        self.n_success = 0
        self.t_reward = 0
        self.n_danger = 0
        self.n_collison = 0
        self.reward_info = 0
        self.n_step = 0
        self.sub = {}
        self.cav_p_rate = 0.5


        self.params = self.set_params()

        self.ep_count = 0

    def set_params(self):
        params = [
            SUMO_HOME + "bin/sumo" + ("-gui" if self.gui else ""), "-c",
            self.data_dir + self.config + ".sumocfg",
            "--tripinfo-output", self.data_dir + "tripinfo.xml",
            "--waiting-time-memory", "1800",
            "--no-step-log", "true",
            "--lanechange.duration", "0",
            "--collision.action", "remove",
            "--collision.stoptime", "0.1"
        ]

        if self.gui:
            params += [
                # "--delay", str(self.args["delay"]),
                "--start", "true", "--quit-on-end", "false",
                "--gui-settings-file", self.SUMO_ENV + "data/" + self.config + "/gui-settings.cfg"
            ]

        return params

    ####################################################################################################################
    ####################################################################################################################

    def start(self):
        # Start the simulation
        traci.start(self.set_params())
        traci.addStepListener(self.plexe)
        self.warm_up()

    def warm_up(self):
        while self.s_step < int(self.warmup_duration / self.step_length):
            self.simulation_step()
        while not self.join_list:
            self.join_list = self.get_join_list()
            self.simulation_step()
        print("Warm up ends")


    def stop(self):
        traci.close()
        sys.stdout.flush()


    def simulation_reset(self):
        self.stop()
        # Get a new & random penetration rate
        # self.cav_penetration_rate()
        # Write to the cfg file
        ########################################### To lock if multiprocess ############################################
        # self.set_penetration_rate()
        # Write a new random flow
        self.cav_penetration_rate()
        self.set_penetration_rate()
        self.write_flow(self.generate_random_flow())
        self.s_step = 0
        self.topology = {}
        self.platoons = {}
        self.leave_list = []
        self.join_list = []
        self.merge_list = []
        self.buffer_list = []
        self.tlp_list = []
        self.n_failure = 0
        self.n_collison = 0
        self.n_success = 0
        self.n_danger = 0
        self.n_step = 0
        self.t_reward = 0
        self.n_veh_simulation = [[], []]
        self.n_cav_simulation = [[], []]
        self.n_platoon_simulation = []
        self.n_processus_simulation = []
        self.danger_try_simulation = []
        self.n_faillure_simulation = []
        self.n_collison_simulation = []
        self.n_processus_summary = []
        self.n_veh_teleported = []
        self.n_cav_teleported = []
        self.n_veh_arrived = []
        self.n_cav_arrived = []
        self.platoons_edge = {}
        self.n_processus = 0
        self.n_wrong_request = 0
        # record of the origin observation for an episode
        self.o0 = []
        # record of current observation for an episode
        self.o = []
        self.collision = []
        self.danger_try = 0
        self.start()


    def simulation_step(self):

        '''if self.leave_list:
            # print("Input:", leave_list)
            self.do_leaves()
            # print("Output:", leave_list)
        if self.step == 0:
            traci.gui.setOffset("View #0", -400, 60)
            traci.gui.setZoom("View #0", 1500)
        # if self.s_step % 10 == 1:
            # self.check_all_leave_requests()
            # self.check_new_cavs_from_ramps()
        if self.join_list:
            joiner_id = self.join_list[1]
            traci.vehicle.subscribeContext(joiner_id, traci.constants.CMD_GET_VEHICLE_VARIABLE, 1000,
                                           [traci.constants.POSITION_2D, traci.constants.VAR_SPEED], 0, 1800)'''
        self.communicate()
        if self.s_step == 0:
            for e in traci.edge.getIDList():
                self.platoons_edge[e] = []
        traci.simulationStep()
        if self.multi_joiners:
            if self.join_list:
                for l in self.join_list:
                    joiner = l[1]
                    if joiner not in self.sub.keys():
                        traci.vehicle.subscribeContext(joiner, traci.constants.CMD_GET_VEHICLE_VARIABLE, 100,
                                                       [VAR_SPEED, VAR_POSITION])
                        traci.vehicle.addSubscriptionFilterLeadFollow(lanes=[0, 1])
                    if joiner in traci.vehicle.getIDList():
                        self.sub[joiner] = traci.vehicle.getContextSubscriptionResults(joiner)
                    # self.sub2 = traci.vehicle.getContextSubscriptionResults(self.join_list[2])
                    else:
                        self.sub[joiner] = -1
                        # self.sub2 = -1
        else:
            if self.join_list:
                joiner = self.join_list[1]
                if joiner not in self.sub.keys():
                    traci.vehicle.subscribeContext(joiner, traci.constants.CMD_GET_VEHICLE_VARIABLE, 100,
                                                    [VAR_SPEED, VAR_POSITION])
                    traci.vehicle.addSubscriptionFilterLeadFollow(lanes=[0, 1])
                if joiner in traci.vehicle.getIDList():
                    self.sub[joiner] = traci.vehicle.getContextSubscriptionResults(joiner)
            # self.sub2 = traci.vehicle.getContextSubscriptionResults(self.join_list[2])
                else:
                    self.sub[joiner] = -1
                    # self.sub2 = -1

        self.initialize_traffic_control()
        self.update_teleport_vehicles()
        self.update_left_vehicles()
        self.s_step += 1
        if self.s_step % 5 == 0:
            self.info_simulation()


    def reset(self):
        self.join_list = []
        if self.s_step == 0:
            self.simulation_reset()
        elif self.s_step * self.step_length >= self.simulation_duration - 100:
            with open("record_simulation.txt", "a") as f:
                # Writing data to a file
                f.write('Failure : ' + str(self.n_failure) + ', Danger : ' + str(self.n_danger) + ', Collison : ' + str(
                    self.n_collison) + ', Success : ' + str(self.n_success) + ', Average Reward : ' +
                        str(self.t_reward / (
                                    self.n_failure + self.n_success + self.n_collison)) + ', Number of steps : ' + str(
                    self.n_step) + '\n')
            e = {str(self.m_flow[0]) + '_' + str(self.p_flow): [self.n_veh_simulation,
                                                                self.n_cav_simulation,
                                                                self.n_platoon_simulation,
                                                                self.n_processus_simulation,
                                                                self.danger_try_simulation,
                                                                self.n_faillure_simulation,
                                                                self.n_collison_simulation,
                                                                self.n_processus_summary,
                                                                self.n_veh_teleported,
                                                                self.n_veh_arrived,
                                                                self.n_processus]}
            with open('evaluation_train.pkl', 'wb') as f:
                pickle.dump(e, f)
            self.simulation_reset()
        else:
            while not self.join_list:
                self.join_list = self.get_join_list()
                self.simulation_step()

        '''joiner_id = self.join_list[1]
        traci.vehicle.subscribeContext(joiner_id, traci.constants.CMD_GET_VEHICLE_VARIABLE, 100, [VAR_SPEED, VAR_POSITION])
        traci.vehicle.addSubscriptionFilterLeadFollow(lanes=[0, 1])
        # traci.vehicle.subscribeContext(self.join_list[2], traci.constants.CMD_GET_VEHICLE_VARIABLE, 100, [VAR_SPEED, VAR_POSITION])
        # traci.vehicle.addSubscriptionFilterLeadFollow(lanes=[0])
        self.simulation_step() # activate subscription to get observations
        self.o0 = self.obs()
        print(joiner_id)'''

    def step(self, action):
        o = self.obs()
        '''Safety check, distance to decide, some ugly hard coding here'''
        for i in range(len(o)):
            if min(o[i][-4], o[i][-5]) < 0 and action[i] == 5:
                # action[i] = 2
                self.danger_try += 1
        self.do_joins(action)
        self.action = action
        j_list = copy.deepcopy(self.join_list)
        if self.multi_joiners and self.s_step * self.step_length < self.simulation_duration - 100:
            self.join_list = self.get_join_list()
            while not self.join_list:
                self.join_list = self.get_join_list()
                self.simulation_step()  # activate subscription to get observations
        for l in self.join_list:
            if not l:
                self.join_list.remove(l)
        for i in range(5):
            d = self.done()
            for j in range(len(j_list)):
                if action[j] == 5:
                    if d[j] == 0:
                        traci.vehicle.setSpeed(self.join_list[j][1], self.o0[j][0])
                    elif self.join_list[j] and self.join_list[j][1] in traci.vehicle.getIDList():
                        traci.vehicle.setSpeed(self.join_list[j][1], -1)
            self.simulation_step()
            collision_step = traci.simulation.getCollidingVehiclesIDList()
            if collision_step:
                self.collision = collision_step

            '''for k in range(len(self.join_list)):
                if d[k]:
                    self.obs_done[self.join_list[k][1]] = o[k]'''

        r = self.rew()
        self.t_reward += sum(r)
        self.reward_info = r
        self.n_step += 1
        for i in range(len(self.join_list)):
            if d[i] and self.join_list[i] and self.join_list[i][1] in traci.vehicle.getIDList() and r[i] > -50:
                traci.vehicle.setSpeed(self.join_list[i][1], -1)
            if r[i] == -100000:
                self.n_collison += 1
            if d[i] == 1:
                # print(r[i])
                if r[i] == -50:
                    self.clean_fail_joiner(self.join_list[i])
                    self.n_failure += 1
                elif r[i] == -30:
                    self.n_danger += 1
                elif r[i] == -100000:
                    pass
                    '''self.n_collison += 1
                    with open("record_collision.txt", "a") as f:
                        list_print = self.o0 + self.o + list(self.collision) + [self.join_list[0], self.join_list[1],
                                                                                self.join_list[2], self.lane_front_id,
                                                                                self.lane_follow_id,
                                                                                self.plane_front_id, self.plane_follow_id]
                        f.write(','.join(str(e) for e in list_print))
                        f.write('\n')'''
                else:
                    self.n_success += 1
                    with open("record_s.txt", "a") as f:
                        list_print = self.o0 + self.o
                        f.write(','.join(str(e) for e in list_print))
                        f.write('\n')
        j_l = copy.deepcopy(self.join_list)
        for i in range(len(self.join_list)):
            if d[i] == 1:
                self.join_list.remove(j_l[i])
                self.sub.pop(j_l[i][1])

    def obs(self):
        observation = []
        for i in range(len(self.join_list)):
            l = self.join_list[i]
            joiner_id = l[1]
            if joiner_id in self.obs_done.keys():
                observation.append(self.obs_done[joiner_id])
            else:
                platoon_front_id = l[2]
                veh_list = traci.vehicle.getIDList()
                position_geo = {'joiner_id': (-1, -1), 'lane_front_id': (-1, -1), 'lane_follow_id': (-1, -1),
                                'plane_front_id': (-1, -1), 'plane_follow_id': (-1, -1), 'platoon_front_id': (-1, -1),
                                'platoon_follow_id': (-1, -1)}
                v = {'joiner_id': 27, 'lane_front_id': 27, 'lane_follow_id': 27, 'plane_front_id': 33,
                     'plane_follow_id': 33,
                     'platoon_front_id': 33, 'platoon_follow_id': 33}
                # a = {'joiner_id': 0, 'lane_front_id': 0, 'lane_follow_id': 0, 'plane_front_id': 0, 'plane_follow_id': 0}
                t1 = -1
                t2 = -1
                lane_front_id = 0
                lane_follow_id = 0
                plane_front_id = 0
                plane_follow_id = 0
                if self.sub[joiner_id] and self.sub[joiner_id] != -1 and joiner_id in veh_list:
                    l_joiner = self.get_veh_relative_lane(joiner_id, veh_list)
                    l_platoon = self.get_veh_relative_lane(platoon_front_id, veh_list)
                    neighbors = list(self.sub[joiner_id].keys())
                    self.neighbors = neighbors
                    p_joiner = self.get_veh_pos(joiner_id, veh_list)
                    position_geo['joiner_id'] = p_joiner
                    v['joiner_id'] = self.get_veh_speed(joiner_id, veh_list)
                    position_geo['lane_front_id'] = (p_joiner[0] + 100, p_joiner[1])
                    position_geo['lane_follow_id'] = (p_joiner[0] - 100, p_joiner[1])
                    position_geo['plane_front_id'] = (p_joiner[0] + 100, p_joiner[1])
                    position_geo['plane_follow_id'] = (p_joiner[0] - 100, p_joiner[1])
                    for veh in neighbors:
                        p_v = self.sub[joiner_id][veh][VAR_POSITION]
                        l_v = self.get_veh_relative_lane(veh, veh_list)
                        if l_joiner == l_v:
                            if p_joiner[0] < p_v[0] and p_v[0] <= position_geo['lane_front_id'][0]:
                                lane_front_id = veh
                                position_geo['lane_front_id'] = p_v
                                v['lane_front_id'] = self.sub[joiner_id][veh][VAR_SPEED]
                            elif p_joiner[0] >= p_v[0] and p_v[0] >= position_geo['lane_follow_id'][0]:
                                lane_follow_id = veh
                                position_geo['lane_follow_id'] = p_v
                                v['lane_follow_id'] = self.sub[joiner_id][veh][VAR_SPEED]
                        if l_platoon == l_v:
                            if p_joiner[0] < p_v[0] and p_v[0] <= position_geo['plane_front_id'][0]:
                                plane_front_id = veh
                                position_geo['plane_front_id'] = p_v
                                v['plane_front_id'] = self.sub[joiner_id][veh][VAR_SPEED]
                            elif p_joiner[0] >= p_v[0] and p_v[0] >= position_geo['plane_follow_id'][0]:
                                plane_follow_id = veh
                                position_geo['plane_follow_id'] = p_v
                                v['plane_follow_id'] = self.sub[joiner_id][veh][VAR_SPEED]

                position_geo['platoon_front_id'] = self.get_veh_pos(platoon_front_id, veh_list)
                v['platoon_front_id'] = self.get_veh_speed(platoon_front_id, veh_list)
                # a['platoon_front_id'] = self.get_veh_accel(platoon_front_id, veh_list)

                '''if platoon_front_id in veh_list and self.sub2 and self.sub2 != -1:
                    neighbors_platoon = list(self.sub2.keys())
                    for veh in neighbors_platoon:
                        p_v = self.sub2[veh][VAR_POSITION]
                        if position_geo['platoon_front_id'][0] > p_v[0]:
                            platoon_follow_id = veh
                            position_geo['platoon_follow_id'] = p_v
                            v['platoon_follow_id'] = self.sub2[veh][VAR_SPEED]
                            break'''
                leader = l[0]
                leaders = list(self.platoons.keys())
                if platoon_front_id in veh_list and leader in veh_list and leader in leaders:
                    idx = leaders.index(leader)
                    if len(leaders) > idx + 1:
                        platoon_follow_id = leaders[idx + 1]
                        position_geo['platoon_follow_id'] = self.get_veh_pos(platoon_follow_id, veh_list)
                        v['platoon_follow_id'] = self.get_veh_speed(platoon_follow_id, veh_list)

                d1 = self.get_veh_dist('lane_front_id', 'joiner_id',
                                       position_geo)  # distance between joiner and its front
                d2 = self.get_veh_dist('joiner_id', 'lane_follow_id',
                                       position_geo)  # distance between joiner and its follower
                d3 = self.get_veh_dist('plane_front_id', 'joiner_id', position_geo)
                d4 = self.get_veh_dist('joiner_id', 'plane_follow_id', position_geo)
                d5 = self.get_veh_dist('platoon_front_id', 'joiner_id', position_geo)
                d6 = self.get_veh_dist('platoon_front_id', 'platoon_follow_id',
                                       position_geo)  # distance between last member of platoon and its follower

                # observation = list(np.array(list(v.values())) / 30) + list(np.array([d1, d2, d3, d4, d5, d6]) / 200)
                o = list(v.values()) + [d1, d2, d3, d4, d5, d6]
                # obs = v + a + [d1, d2, d3, ds1, ds2]
                if d5 < 1:
                    o.append(1)
                else:
                    o.append(0)
                observation.append(o)
        self.o0 = observation
        return observation

    def rew(self):
        rew = []
        # collision = 0
        observ = self.obs()
        do = self.done()
        for i in range(len(self.join_list)):
            l = self.join_list[i]
            joiner_id = l[1]
            platoon_front_id = l[2]
            d = do[i]
            if joiner_id in self.tlp_list or joiner_id not in traci.vehicle.getIDList():
                r = -100000
                print(r)
                # collision = 1
            else:
                if d == 1:
                    if traci.vehicle.getLeader(joiner_id) and traci.vehicle.getLeader(joiner_id)[0] == platoon_front_id:
                        r = 10
                        print(r)
                        # for evaluation
                        '''dis = self.get_distance(joiner_id, platoon_front_id)
                        if dis < 5:
                            r = -30
                        if dis > 20:
                            r = r * 20 / d'''
                        with open("d.txt", "a") as f:
                            # Writing data to a file
                            f.write(str(d) + '\n')
                    else:
                        r = -50
                        print(r)
                else:
                    r = -1
                    # need to change to relative for final net
                    if self.get_veh_laneIdx(joiner_id) != self.get_veh_laneIdx(joiner_id) and self.action[i]==5:
                        r -= 10
                    # For simlify evaluation
                    '''v = traci.vehicle.getSpeed(joiner_id)
                    print("traci.vehicle.getSpeed", joiner_id)
                    o = observ[i]
                    if o[-1] < MIM_GAP:  # Too small space with preceding/following veh :
                        r -= 10
                    if v > 35:
                        r = r - 10
                    if v < 15:
                        r = r - 5'''
            rew.append(r)
            ''' o = self.obs()
               if o[7] < MIM_GAP or o[8] < MIM_GAP: # Too small space with preceding/following veh :
                   r -= 100
               if abs(o[11]) < 1.5 * MIM_GAP:
                   r -= 200
               if o[0] < MIM_SPEED: # Too small speed
                   r += 10 * (o[0] - MIM_SPEED)
               elif o[0] > MAX_SPEED:# Too big speed
                   r += 10 * (MAX_SPEED - o[0])'''
            # print("Reward : ", r)
        return rew

    def done(self):
        done = []
        for l in self.join_list:
            do = 0
            leader_id = l[0]
            joiner_id = l[1]
            veh_list = traci.vehicle.getIDList()
            if joiner_id in self.tlp_list or joiner_id not in veh_list or leader_id not in veh_list:
                do = 1
            else:
                l1 = self.get_veh_laneIdx(joiner_id)
                l2 = self.get_veh_laneIdx(leader_id)
                if self.get_veh_laneIdx(joiner_id) - traci.edge.getLaneNumber(
                        traci.vehicle.getRoadID(joiner_id)) == self.get_veh_laneIdx(
                        leader_id) - traci.edge.getLaneNumber(traci.vehicle.getRoadID(leader_id)):
                    if leader_id in self.platoons.keys():
                        self.platoons[leader_id]["state"] = 0
                    do = 1
            done.append(do)
        return done

    def done_one_joiner(self, join_list):
        do = 0
        leader_id = join_list[0]
        joiner_id = join_list[1]
        veh_list = traci.vehicle.getIDList()
        if joiner_id in self.tlp_list or joiner_id not in veh_list or leader_id not in veh_list:
            do = 1
        else:
            l1 = self.get_veh_laneIdx(joiner_id)
            l2 = self.get_veh_laneIdx(leader_id)
            if self.get_veh_laneIdx(joiner_id) - traci.edge.getLaneNumber(
                    traci.vehicle.getRoadID(joiner_id)) == self.get_veh_laneIdx(leader_id) - traci.edge.getLaneNumber(
                    traci.vehicle.getRoadID(leader_id)):
                if leader_id in self.platoons.keys():
                    self.platoons[leader_id]["state"] = 0
                do = 1
        return do

    def info(self):
        return {'r': self.reward_info} if not self.log else self.log_info()

    def is_simulation_end(self):
        return traci.simulation.getMinExpectedNumber() == 0

    def get_current_time(self):
        return traci.simulation.getCurrentTime() // 1000

    ####################################################################################################################
    ####################################################################################################################
    # platoon
    # https://github.com/michele-segata/plexe-pyapi
    def communicate(self):
        """
        Performs data transfer between vehicles, i.e., fetching data from
        leading and front vehicles to feed the CACC algorithm
        """
        for vid, l in self.topology.items():
            if "leader" in l.keys():
                # get data about platoon leader
                ld = self.plexe.get_vehicle_data(l["leader"])
                # pass leader vehicle data to CACC
                self.plexe.set_leader_vehicle_data(vid, ld)
                # pass data to the fake CACC as well, in case it's needed
                # self.plexe.set_leader_vehicle_fake_data(vid, ld)
            if "front" in l.keys():
                # get data about platoon leader
                fd = self.plexe.get_vehicle_data(l["front"])
                # pass front vehicle data to CACC
                self.plexe.set_front_vehicle_data(vid, fd)
                # compute GPS distance and pass it to the fake CACC
                # distance = self.get_distance(vid, l["front"])
                # self.plexe.set_front_vehicle_fake_data(vid, fd, distance)

    # https://github.com/michele-segata/plexe-pyapi
    def get_distance(self, v1, v2):
        """
        Returns the distance between two vehicles, removing the length
        :param v1: id of first vehicle
        :param v2: id of the second vehicle
        :return: distance between v1 and v2
        """
        v1_data = self.plexe.get_vehicle_data(v1)
        v2_data = self.plexe.get_vehicle_data(v2)
        return math.sqrt((v1_data[POS_X] - v2_data[POS_X]) ** 2 +
                         (v1_data[POS_Y] - v2_data[POS_Y]) ** 2) - 4

    # initialize controllers of different types of vehicles, and update topology if necessary
    def initialize_traffic_control(self, min_platoon_size=2, max_platoon_size=2):
        # get list of input vehicles in the last step
        if len(traci.simulation.getDepartedIDList()) != 0:
            for vid in traci.simulation.getDepartedIDList():
                # individual cavs, in normal lanes when input
                if traci.vehicle.getTypeID(vid) == "cav":
                    # if "on" in vid:
                    # plexe.set_active_controller(vid, DRIVER)
                    # else:
                    # plexe.set_active_controller(vid, ACC)
                    self.plexe.set_active_controller(vid, DRIVER)
                    # no lane change unless demander afterwards
                    # traci.vehicle.setSpeedMode(vid, 0)
                    # self.plexe.set_cc_desired_speed(vid, SPEED)
                # cavs in the lane destined
                if traci.vehicle.getTypeID(vid) == "cav2":
                    # Begin with ACC
                    self.plexe.set_active_controller(vid, ACC)
                    self.plexe.set_fixed_lane(vid, self.p_lane, False)
                    # most checks off (legacy) -> [0 0 0 0 0 0] -> Speed Mode = 0
                    traci.vehicle.setSpeedMode(vid, 0)
                    # Faster
                    self.plexe.set_cc_desired_speed(vid, self.SPEED_CAV_LANES)
                    # No car in front
                    if traci.vehicle.getLeader(vid) is None:
                        self.platoons[vid] = {"ini_size": random.randint(min_platoon_size, max_platoon_size),
                                              "members": [vid], "state": 0}
                    # At least one car in front, vid_front
                    else:
                        vid_front = traci.vehicle.getLeader(vid)[0]
                        leader_id = 0
                        # The car in front is not yet in any platoon, that is to say, a single leader
                        if vid_front in self.platoons.keys():
                            leader_id = vid_front
                        # The car in front is already in a platoon
                        # elif vid_front in topology.keys():
                        else:
                            leader_id = self.topology[vid_front]["leader"]
                        # The platoon reached the defined length, begin new platoon
                        if leader_id != 0 and self.platoons[leader_id]["ini_size"] <= len(
                                self.platoons[leader_id]["members"]):
                            self.platoons[vid] = {"ini_size": random.randint(min_platoon_size, max_platoon_size),
                                                  "members": [vid], "state": 0}
                        # else, add current vehicle as a member
                        else:
                            self.plexe.set_active_controller(vid, CACC)
                            self.platoons[leader_id]["members"].append(vid)
                            self.topology[vid] = {"front": vid_front, "leader": leader_id}

    def update_left_vehicles(self):
        did_1 = self.detector_1
        did_2 = self.detector_2
        veh_left = traci.inductionloop.getLastStepVehicleIDs(did_2) + traci.inductionloop.getLastStepVehicleIDs(did_1)
        if len(veh_left) != 0:
            # print("Vehicle leaved:", veh_left)
            for leader in veh_left:
                if leader in self.platoons.keys():
                    platoon = self.platoons[leader]["members"][1:]
                    for veh in platoon:
                        self.topology.pop(veh, None)
                    self.platoons.pop(leader, None)
                elif leader in self.topology.keys():
                    self.topology.pop(leader, None)

    def update_teleport_vehicles(self):
        veh_tlp = traci.simulation.getCollidingVehiclesIDList()
        if len(veh_tlp) != 0:
            # print("Vehicle moved:", veh_tlp)
            for veh in veh_tlp:
                if veh in self.platoons.keys():
                    leader_id = veh
                    p = self.platoons[leader_id]["members"]
                    for v in p:
                        if v in self.topology.keys():
                            self.topology.pop(v)
                        traci.vehicle.remove(v)
                    self.platoons.pop(leader_id)
                elif veh in self.topology.keys():
                    leader_id = self.topology[veh]["leader"]
                    p = self.platoons[leader_id]["members"]
                    for v in p:
                        if v in self.topology.keys():
                            self.topology.pop(v)
                        traci.vehicle.remove(v)
                    self.platoons.pop(leader_id)
                elif veh in traci.vehicle.getIDList():
                    traci.vehicle.remove(veh)
            self.tlp_list += veh_tlp

    def clean_fail_joiner(self, join_list):
        leader_id = join_list[0]
        joiner_id = join_list[1]
        if leader_id in self.platoons.keys():
            self.platoons[leader_id]["state"] = 0
            self.platoons[leader_id]["members"].remove(joiner_id)
        if joiner_id in self.topology.keys():
            self.topology.pop(joiner_id)
        try:
            traci.vehicle.remove(joiner_id)
        except:
            pass

    def catch_platoon(self, joiner_id, platoon_leader_id, platoon_front_id):
        # update topology
        self.topology[joiner_id] = {"leader": platoon_leader_id, "front": platoon_front_id}
        # update platoons
        index = self.platoons[platoon_leader_id]["members"].index(platoon_front_id) + 1
        self.platoons[platoon_leader_id]["members"].insert(index, joiner_id)
        # maybe different cases exist
        # self.plexe.set_cc_desired_speed(joiner_id, self.join_speed)

    def finish_join(self, joiner_id, leader_id, flist):
        edge = traci.vehicle.getRoadID(joiner_id)
        lane_num = traci.edge.getLaneNumber(edge)
        traci.vehicle.setSpeedMode(joiner_id, 0b001000000000)
        traci.vehicle.setVehicleClass(joiner_id, 'hov')
        traci.vehicle.changeLaneRelative(joiner_id, 1, 10)
        self.plexe.set_fixed_lane(joiner_id, lane_num - 1, safe=False)
        self.plexe.set_active_controller(joiner_id, CACC)
        self.plexe.set_cc_desired_speed(joiner_id, self.SPEED_CAV_LANES)
        self.plexe.set_path_cacc_parameters(joiner_id, distance=self.DISTANCE)
        traci.vehicle.updateBestLanes(joiner_id)
        '''for v in flist:
            self.plexe.set_active_controller(v, CACC)
        # self.plexe.enable_auto_feed(joiner_id, True, self.topology[joiner_id]["leader"], self.topology[joiner_id]["front"])
        for vid in flist:
            self.plexe.set_path_cacc_parameters(vid, distance=self.DISTANCE)
        self.reset_leader(joiner_id, flist, leader_id)
        return self.topology'''

    def do_joins(self, action):
        for i in range(len(self.join_list)):
            l = self.join_list[i]
            a = action[i]
            leader_id = l[0]
            joiner_id = l[1]
            edge = traci.vehicle.getRoadID(joiner_id)
            platoon_front_id = l[2]
            platoon_behind_id = l[3]
            flist = l[4]
            state = l[5]
            a1 = (a == 5)
            a2 = a - 2
            # a1 = (action == 5)
            # a2 = (action - 2)
            if a1:
                a2 = 0
            # Wait too long for the last operation to finish
            if state == self.REQUEST and leader_id in self.platoons.keys() and self.platoons[leader_id]["state"] != 0:
                self.join_list[i] = []
                self.n_wrong_request += 1
                # print("Join request denied: Platoon occupied", joiner_id, leader_id)
                return -1
            elif leader_id in self.platoons.keys():
                traci.vehicle.setSpeedMode(joiner_id, 32)
                traci.vehicle.slowDown(joiner_id, max(0, traci.vehicle.getSpeed(joiner_id) + a2/2), 0.5)
                self.platoons[leader_id]["state"] = 1
                if state == self.REQUEST:
                    # print("Join request approved:", joiner_id, leader_id)
                    # traci.vehicle.setColor(leader_id, (50, 100, 0))
                    # traci.vehicle.setColor(platoon_front_id, (50, 100, 0))
                    traci.vehicle.setLaneChangeMode(joiner_id, 0b000000000000)
                    self.join_list[i][5] = self.CATCHING_PLATOON
                    '''lane_num = traci.edge.getLaneNumber(edge)
                    lane_idx = traci.vehicle.getLaneIndex(joiner_id)
                    # first change to the lane near the lane of platoon
                    if lane_num - 1 - lane_idx > 1:
                        self.plexe.set_fixed_lane(joiner_id, lane_num - 2, safe=True)
                    self.join_list[5] = self.CATCHING_PLATOONnoon_front_id) < self.join_distance + 10:
                    self.open_gap(flist)
                    self.join_list[5] = self.OPENING_GAP'''

                if a1:
                    # when the gap is large enough, complete the maneuver
                    '''dj = traci.vehicle.getLanePosition(joiner_id)
                    dp = traci.vehicle.getLanePosition(platoon_front_id)
    
                    lane_num = traci.edge.getLaneNumber(edge)
                    lane_idx = traci.vehicle.getLaneIndex(joiner_id)
    
                    if lane_num - 1 - lane_idx == 1 and dj < dp - self.DISTANCE and traci.vehicle.getRoadID(
                            platoon_front_id) == edge and \
                            (platoon_behind_id == -1 or self.get_distance(platoon_front_id, platoon_behind_id) > \
                            2 * self.join_distance + 0):'''
                    lane_num = traci.edge.getLaneNumber(edge)
                    if traci.vehicle.getVehicleClass(joiner_id) != 'hov':
                        try:
                            self.finish_join(joiner_id, leader_id, flist)
                            self.catch_platoon(joiner_id, leader_id, platoon_front_id)
                        except KeyError:
                            pass
                            # print(joiner_id)
                    elif traci.vehicle.getLaneIndex(joiner_id) != lane_num - 1:
                        traci.vehicle.changeLaneRelative(joiner_id, 1, 10)
                        self.plexe.set_fixed_lane(joiner_id, lane_num - 1, safe=False)
                        self.plexe.set_cc_desired_speed(joiner_id, self.SPEED_CAV_LANES)
                        traci.vehicle.updateBestLanes(joiner_id)

    ####################################################################################################################
    ####################################################################################################################
    # control

    # get the list of ids of leaders of platoons, which are behinds the cav and in the same edge
    def get_platoons_available(self, edge, veh):
        platoons_available = []
        # getLanePosition return : the distance from the front bumper to the start of the lane in [m]
        p_v = traci.vehicle.getLanePosition(veh)
        # ids of leaders of platoons in the same edge
        veh_edge_list = set(traci.edge.getLastStepVehicleIDs(edge))
        for leader in set(self.platoons.keys()).intersection(veh_edge_list):

            if self.platoons[leader]["state"] == 0 and leader not in [l[0] for l in self.join_list if len(l) is not 0]:
                fronter = self.platoons[leader]["members"][-1]
                if fronter in veh_edge_list:
                    p_p = traci.vehicle.getLanePosition(fronter)
                    # print(p_p)
                    # only record those behinds the cav, name and position
                    if p_p <= p_v + 5 and p_p >= p_v - 150:
                        '''==========Add more informations if needed=========='''
                        platoons_available.append(leader)
        # last edge
        '''for leader in set(self.platoons.keys()).intersection(set(traci.edge.getLastStepVehicleIDs(edge[:4] + str(int(edge[4:]) - 1)))):
            platoons_available.append(leader)'''
        return platoons_available

    def choose_position(self, veh, platoon_leader):
        '''method: last, random, best'''
        # get all the members of platoons
        p = self.platoons[platoon_leader]["members"]
        # always following the last
        method = self.choose_position_method
        if method == "last":
            return p[-1]
        # randomly
        if method == "random":
            i = random.randrange(0, len(p))
            return p[i]
        # choose based on routes
        if method == "best":
            r = traci.vehicle.getRouteID(veh)
            # follow the last veh in the same route, otherwise follow the leader, because it is the longest route
            if r == "m":
                r_platoons = map(lambda v: traci.vehicle.getRouteID(v), p)
                for i in range(len(r_platoons) - 1, 0):
                    if r_platoons[i] == 'm':
                        return p[i]
                return platoon_leader
            # follow the last veh leave in the same ramp, or the next ramp, otherwise follow the leader
            if 'off' in r:
                r_index = int(r[-1])
                r_platoons = map(lambda v: traci.vehicle.getRouteID(v)[-1], p)
                for i in range(len(r_platoons) - 1, 0):
                    if r_platoons[i] != 'm' or int(r_platoons[i]) >= r_index:
                        return p[i]

    def get_route_after_edge(self, edge, veh):
        r = traci.vehicle.getRoute(veh)
        idx = r.index(edge)
        # route after the edge
        r = r[idx:]
        return r

    def get_score_4_1platoon(self, edge, veh, platoon_leader):
        # the distance already passed of the edge
        common_length = - traci.vehicle.getLanePosition(veh)
        # get the length of common route
        r = self.get_route_after_edge(edge, veh)
        # do not forget the leaving on advance for off rampe
        if "off" in r[-1]:
            common_length = common_length - 450
        p = self.platoons[platoon_leader]["members"]
        r_platoons = []
        for v in p:
            r_platoons.append(self.get_route_after_edge(edge=edge, veh=v))
        r_max = set(max(r_platoons, key=len))
        # common route
        for e in set(r) & r_max:
            if "off" not in e:
                common_length += traci.lane.getLength(edge + '_0')

        # while joining
        # Assuming no speed ajusted x/v1 = (x+d)/v2
        # speed of joiner
        v_v = traci.vehicle.getSpeed(veh)
        # position of joiner
        p_v = traci.vehicle.getLanePosition(veh)
        # speed of platoon leader
        v_p = traci.vehicle.getSpeed(platoon_leader)
        # position of platoon leader
        p_p = traci.vehicle.getLanePosition(platoon_leader)

        # get vehicle's lane index
        # calculate total lane change duration for join
        t_lc = abs(traci.vehicle.getLaneIndex(veh) - traci.vehicle.getLaneIndex(platoon_leader)) * self.LANE_CHANGE_DURATION

        # distance to run for vehicle before meet
        d = v_v * (p_v - p_p) / (v_v - v_p)
        common_length = common_length - d - t_lc * v_p
        # origin ttc in common route for vehicle - new ttc
        score = common_length / v_v - common_length / v_p

        # calculate extra time cost for followers --- no influence ??? last is best ????
        # calculate leaving time cost
        score = score - t_lc
        # print(platoon_leader, score)
        return score

    def get_score_4_veh(self, edge, veh, p_list):
        score = []
        for p in p_list:
            score.append(self.get_score_4_1platoon(edge, veh, p))
        return p_list[np.argmax(score)], np.max(score)

    def get_joiner(self, cavs, i=0):
        '''method: first, random, best'''
        method = self.choose_veh_method
        if method == "first":
            i = i
        elif method == "random":
            i = random.randrange(0, len(cavs))
        elif method == "best":
            i = np.argmax(map(self.get_score_4_veh, cavs))
        else:
            print("Method name wrong")
        joiner = cavs[i]

        return joiner

    '''join_list: leader, joiner, front, behind, followers'''

    def get_join_list(self):
        veh_list = traci.vehicle.getIDList()
        t = 0
        j_list = copy.deepcopy(self.join_list)
        if not self.multi_joiners:
            while len(self.join_list) == 0 and t < self.max_try:
                e = random.choice(self.edge_m)
                l = traci.edge.getLaneNumber(e)
                # print("Try ", e)
                # just the next lane
                vehs = traci.lane.getLastStepVehicleIDs(e + '_' + str(l - 2))
                # right order ?
                cavs = [v for v in vehs if traci.vehicle.getTypeID(v) == 'cav']
                if cavs:
                    joiner = self.get_joiner(cavs)
                    p_list = self.get_platoons_available(e, joiner)
                    if len(p_list) > 0:
                        '''p, s = self.get_score_4_veh(e, joiner, p_list)
                        position = self.choose_position(joiner, p)
                        if s > 0:
                            self.join_list = [p, joiner, position, self.get_following_vehicle(position),
                                              self.get_all_following_vehicles(position), 0]'''
                        p = random.choice(p_list)
                        position = self.choose_position(joiner, p)
                        j_list = [p, joiner, position, -1,
                                          -1, 0]
                t += 1
            self.edge_m = copy.deepcopy(self.edges)
            idx = self.edges.index(e)
            for i in range(idx, min(idx+2, len(self.edges))):
                e = self.edges[i]
                self.edge_m.remove(e)
        else:
            j_list = copy.deepcopy(self.join_list)
            joiners = [j_l[1] for j_l in j_list if j_l]
            joiners_active = [j for j in joiners if j in veh_list]
            e_occupied = [traci.vehicle.getRoadID(j) for j in joiners_active]
            for e in self.edge_m:
                if e not in e_occupied:
                    l = traci.edge.getLaneNumber(e)
                    vehs = traci.lane.getLastStepVehicleIDs(e + '_' + str(l - 2))
                    cavs = [v for v in vehs if traci.vehicle.getTypeID(v) == 'cav']
                    if cavs:
                        for joiner in cavs:
                            if not j_list or (joiner not in [list[1] for list in j_list if list]):
                                p_list = self.get_platoons_available(e, joiner)
                                if len(p_list) > 0:
                                    '''p, s = self.get_score_4_veh(e, joiner, p_list)
                                    position = self.choose_position(joiner, p)
                                    if s > 0:
                                        self.join_list = [p, joiner, position, self.get_following_vehicle(position),
                                                          self.get_all_following_vehicles(position), 0]'''
                                    p = random.choice(p_list)
                                    position = self.choose_position(joiner, p)
                                    j_list.append([p, joiner, position, -1, -1, 0])
                                    self.n_processus += 1
                                    break
        return j_list


    ####################################################################################################################
    ####################################################################################################################
    # lane

    def get_lane_veh_ids(self, lane_id):
        return traci.lane.getLastStepVehicleIDs(lane_id)

    def get_lane_veh_n(self, lane_id):
        return traci.lane.getLastStepVehicleNumber(lane_id)

    def get_lane_length(self, lane_id):
        return traci.lane.getLength(lane_id)

    def get_lane_edge_id(self, lane_id):
        return traci.lane.getEdgeID(lane_id)

    # edge

    def get_edge_veh_ids(self, edge_id):
        return traci.edge.getLastStepVehicleIDs(edge_id)

    def get_edge_lane_n(self, edge_id):
        return traci.edge.getLaneNumber(edge_id)

    # car
    def set_veh_speed(self, veh_id, change_of_speed):
        speed = traci.vehicle.getSpeed(veh_id)
        traci.vehicle.setSpeedMode(veh_id, 0)
        traci.vehicle.setSpeed(veh_id, speed + change_of_speed)

    def get_veh_type(self, veh_id):
        return traci.vehicle.getTypeID(veh_id)

    def get_veh_speed(self, veh_id, veh_list):
        if veh_id == -1 or veh_id not in veh_list:
            return 27
        else:
            return traci.vehicle.getSpeed(veh_id)

    def get_veh_accel(self, veh_id, veh_list):
        if veh_id == -1 or veh_id not in veh_list:
            return 0
        else:
            return traci.vehicle.getAcceleration(veh_id)

    def get_veh_lane(self, veh_id):
        return traci.vehicle.getLaneID(veh_id)

    def get_veh_edge(self, veh_id):
        return traci.vehicle.getRoadID(veh_id)

    def get_veh_laneIdx(self, veh_id):
        return traci.vehicle.getLaneIndex(veh_id)

    def get_veh_relative_lane(self, veh_id, veh_list):
        if veh_id == -1 or veh_id not in veh_list:
            return 0
        else:
            l_num = self.get_edge_lane_n(self.get_veh_edge(veh_id))
            l = self.get_veh_laneIdx(veh_id)
            return l_num - l

    def get_veh_pos_on_lane(self, veh_id):
        return traci.vehicle.getLanePosition(veh_id)

    def get_veh_follower(self, veh_id):
        lane_id = self.get_veh_lane(veh_id)
        vehs = self.get_lane_veh_ids(lane_id)
        if veh_id in vehs:
            idx = vehs.index(veh_id)
            if idx != 0:
                return vehs[idx - 1]
            else:
                return -1
        else:
            return -1

    def get_veh_pos(self, veh_id, veh_list):
        if veh_id == -1 or veh_id not in veh_list:
            return (-1, -1)
        else:
            return traci.vehicle.getPosition(veh_id)

    def get_veh_dist(self, front, follow, position):
        if position[front][0] == -1 or position[follow][0] == -1:
            return 100
        else:
            return (position[front][0] - position[follow][0])

    ####################################################################################################################
    ####################################################################################################################
    def info_simulation(self):
        n_veh_step_lane0 = 0
        n_veh_step_lane1 = 0
        n_cav_step_lane0 = 0
        n_cav_step_lane1 = 0

        veh_teleport = traci.simulation.getEndingTeleportIDList()
        veh_arrive = traci.simulation.getArrivedIDList()
        n_veh_teleport = len(veh_teleport)
        n_veh_arrive = len(veh_arrive)
        # n_cav_teleport = len([v for v in veh_teleport if traci.vehicle.getTypeID(v) == ('cav' or 'cav2')])
        # n_cav_arrive = len([v for v in veh_arrive if traci.vehicle.getTypeID(v) == ('cav' or 'cav2')])

        step_n_platoons_edge = {}
        step_l_platoons_edge = {}
        for p in self.platoons.keys():
            edge_p = traci.vehicle.getRoadID(p)
            if edge_p in step_l_platoons_edge.keys():
                step_l_platoons_edge[edge_p] += len(self.platoons[p]['members'])
                step_n_platoons_edge[edge_p] += 1
            else:
                step_l_platoons_edge[edge_p] = len(self.platoons[p]['members'])
                step_n_platoons_edge[edge_p] = 1
                v0 = traci.edge.getLastStepVehicleIDs(edge_p)
                test = [traci.vehicle.getTypeID(v) for v in v0]
        edges_with_platoon = step_n_platoons_edge.keys()

        for e in traci.edge.getIDList():
            v0 = traci.lane.getLastStepVehicleIDs(e + '_0')
            cav0 = [v for v in v0 if traci.vehicle.getTypeID(v) == ('cav' or 'cav2')]
            n_e = len(cav0)
            n = traci.edge.getLaneNumber(e)
            if n >= 2:
                v1 = traci.lane.getLastStepVehicleIDs(e + '_1')
                cav1 = v1
                n_veh_step_lane1 += len(v1)
                n_cav_step_lane1 += len(cav1)
                n_e += len(cav1)
            if n == 3:
                v1 = traci.lane.getLastStepVehicleIDs(e + '_2')
                cav1 = v1
                n_veh_step_lane1 += len(v1)
                n_cav_step_lane1 += len(cav1)
                n_e += len(cav1)
            n_veh_step_lane0 += len(v0)
            n_cav_step_lane0 += len(cav0)
            if e in edges_with_platoon:
                self.platoons_edge[e].append([step_n_platoons_edge[e], step_l_platoons_edge[e], n_e])
            else:
                self.platoons_edge[e].append([0, 0, n_e])

        self.n_veh_simulation[0].append(n_veh_step_lane0)
        self.n_veh_simulation[1].append(n_veh_step_lane1)
        self.n_cav_simulation[0].append(n_cav_step_lane0)
        self.n_cav_simulation[1].append(n_cav_step_lane1)
        self.n_platoon_simulation.append(len(self.platoons))
        self.n_processus_simulation.append(len(self.join_list))
        self.danger_try_simulation.append(self.danger_try)
        self.n_faillure_simulation.append(self.n_failure)
        self.n_collison_simulation.append(self.n_collison)
        self.n_processus_summary.append(self.n_processus)
        self.n_veh_teleported.append(n_veh_teleport)
        self.n_veh_arrived.append(n_veh_arrive)


    # rou & flow logic

    def get_all_routes(self):
        # read the route file
        filename = self.path + "freeway.rou.xml"
        tree = ET.parse(filename)
        root = tree.getroot()
        # record all routes
        routes = []
        for r in root.iter('route'):
            routes.append(r.attrib['id'])
        return routes

    def generate_random_flow(self, simDuration=1800, period=600):
        '''limit_of_flow : [lower bound, upper bound]'''
        routes_list = self.routes
        n_period = int(simDuration / period)
        flow = {}
        for route in routes_list:
            # The origin or the destination is ramp, the flow is more limited
            self.m_flow = random.choices(self.args["m_flow"])
            flow[route] = self.m_flow * np.ones(shape=(n_period,))
        return flow

    def write_flow(self, flow, vtype='vmix', departLane='random', simDuration=1800, period=600):
        '''departLane : string'''
        filename = self.path + "freeway.flow.xml"
        n_period = int(simDuration / period)
        root = ET.Element('additional')
        root.text = "\n\t"
        tree = ET.ElementTree(root)

        for i in range(n_period):
            for r in flow.keys():
                e = ET.SubElement(root, 'flow', attrib={"id": r + str(i),
                                                        "begin": str(i * period),
                                                        "end": str((i + 1) * period),
                                                        "departPos": "last",
                                                        "departSpeed": "max",
                                                        "departLane": departLane,
                                                        "vehsPerHour": str(int(flow[r][i])),
                                                        "type": vtype,
                                                        "route": r})
                e.tail = "\n\t"
                '''if i == n_period - 1 and r == list(flow.keys())[-1]:
                    e.tail = "\n"
                else:
                    e.tail = "\n\t"'''
        '''p_flow = {'m': random.randint(500, 1200), 'm_off1': random.randint(10, 200),
                    'm_off2': random.randint(100, 600)}'''
        self.p_flow = random.choices(self.args["p_flow"])[0]
        p_flow = {'m': self.p_flow}
        for r in p_flow.keys():
            e = ET.SubElement(root, 'flow', attrib={"id": 'p_' + r,
                                                    "begin": str(0),
                                                    "end": str(simDuration),
                                                    "departPos": "last",
                                                    "departSpeed": "max",
                                                    "departLane": str(self.p_lane),
                                                    "vehsPerHour": str(int(p_flow[r])),
                                                    "type": 'cav2',
                                                    "route": r})
            if r == list(p_flow.keys())[-1]:
                e.tail = "\n"
            else:
                e.tail = "\n\t"
        with open(filename, 'wb') as f:
            tree.write(f)

    def set_seed(self):
        if self.args["seed"]:
            random.seed(self.ep_count)
            np.random.seed(self.ep_count)

    def cav_penetration_rate(self):
        self.cav_p_rate = random.randint(0, 7) / 10 + 0.3

    def set_penetration_rate(self):
        filename = self.path + "freeway.rou.xml"
        tree = ET.parse(filename)
        root = tree.getroot()
        for vtype in root.iter('vType'):
            if vtype.attrib['id'] == 'cav':
                vtype.set('probability', str(self.cav_p_rate))
            elif vtype.attrib['id'] == 'passenger':
                vtype.set('probability', str(1 - self.cav_p_rate))
        with open(filename, 'wb') as f:
            tree.write(f)

    ####################################################################################################################
    ####################################################################################################################


    # Log info

    def log_info(self):
        pass
