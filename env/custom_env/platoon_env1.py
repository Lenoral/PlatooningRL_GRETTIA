# https://sumo.dlr.de/pydoc/

from __future__ import absolute_import, print_function

from .utils import SUMO_PARAMS

import os
import sys
import json
import random
import math
import copy
import numpy as np
import xml.etree.ElementTree as ET
# https://github.com/michele-segata/plexe-pyapi
from plexe import Plexe, ACC, CACC, DRIVER, FAKED_CACC, RPM, GEAR, ACCELERATION, SPEED, RADAR_DISTANCE, POS_X, POS_Y
from itertools import permutations

SUMO_HOME = "/usr/share/sumo/"

sys.path.append(SUMO_HOME + 'tools')

from sumolib import net  # noqa
import traci  # noqa


class PlatoonEnv:
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

    def __init__(self, gui=False, log=False, rnd=(False, False)):
        self.args = SUMO_PARAMS

        self.gui = False
        self.config = self.args["config"]
        self.data_dir = self.SUMO_ENV + "data/" + self.config + "/"

        # Here is our network
        self.net = net.readNet(self.data_dir + self.config + ".net.xml")

        self.action_space_n = 20
        self.observation_space_n = 17
        """
        SumoEnv.pretty_print(self.tl_net)
        exit()
        """

        """
        SumoEnv.pretty_print(self.route_net)
        exit()
        """

        traci.start(self.set_params())

        self.gui = gui
        self.log = log
        self.rnd = rnd

        self.veh_n = 0
        self.flow = []
        self.con_p_rate = 1.
        self.ctrl_con_p_rate = 1.
        self.veh_n_p_hour = []
        self.path = './env/custom_env/data/platoon/'
        self.routes = self.get_all_routes()
        self.p_lane = 2

        self.LANE_PLATOON_INDEX = 2
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
        self.LANE_CHANGE_DURATION = 1
        self.choose_position_method = "last"
        self.choose_veh_method = "first"

        self.simulation_duration = 1800
        self.warmup_duration = 80
        self.step_length = 0.1
        self.min_dist = 1e6
        self.join_distance = 10
        self.leave_distance = 10
        self.join_speed = self.SPEED
        self.detector_1 = "d_end_1"
        self.detector_2 = "d_end_2"
        self.plexe = Plexe()
        self.s_step = 0
        self.topology = {}
        self.platoons = {}
        self.leave_list = []
        self.join_list = []
        self.merge_list = []
        self.buffer_list = []
        self.tlp_list = []
        self.edges = ["gneE9", "gneE16", "gneE21"]
        self.route_edge = {"off0": "gneE3", "off1": "gneE11", "off2": "gneE16"}
        self.pass_list = {"off0": [], "off1": [], "off2": []}
        '''self.edge_m = ["gneE1", "gneE2", "gneE6", "gneE7", "gneE8", "gneE9", "gneE10",
                       "gneE14", "gneE15", "gneE19", "gneE20", "gneE21", "gneE22", "gneE23"]'''
        self.edge_m = ["gneE1", "gneE2", "gneE3", "gneE4", "gneE5", "gneE6", "gneE7", "gneE8"]
        self.flag = 0
        self.max_try = 5
        self.n_failure = 0
        self.n_success = 0
        self.t_reward = 0
        self.n_collison = 0
        self.reward_info = 0
        self.reset_flag = 1


        """
        self.update_flow_logic()
        SumoEnv.pretty_print(self.flow_logic)
        # [print(f) for f in self.flow]
        exit()
        """

        self.params = self.set_params()

        self.ep_count = 0

    def set_params(self):
        params = [
            SUMO_HOME + "bin/sumo" + ("-gui" if self.gui else ""), "-c",
            self.data_dir + self.config + ".sumocfg",
            "--tripinfo-output", self.data_dir + "tripinfo.xml",
            "--waiting-time-memory", "1800",
            "--no-step-log", "true",
            "--lanechange.duration", "1",
            "--collision.action", "remove",
            "--collision.stoptime", "0.1",
            "--log",  self.data_dir + "log.xml"
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
        traci.start(self.set_params(), traceFile=self.data_dir + "log.xml")
        traci.addStepListener(self.plexe)
        self.warm_up()

    def warm_up(self):
        while self.s_step < int(self.warmup_duration / self.step_length) or len(self.join_list) == 0:
            self.simulation_step()
        print("Warm up ends")


    def stop(self):
        traci.close()
        sys.stdout.flush()


    def simulation_reset(self):
        self.stop()
        # Get a new & random penetration rate
        self.cav_penetration_rate()
        # Write to the cfg file
        ########################################### To lock if multiprocess ############################################
        self.set_penetration_rate()
        # Write a new random flow
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
        self.t_reward = 0
        self.start()


    def simulation_step(self):
        self.initialize_traffic_control()
        self.update_teleport_vehicles()
        self.update_left_vehicles()
        '''if self.leave_list:
            # print("Input:", leave_list)
            self.do_leaves()
            # print("Output:", leave_list)
        if self.step == 0:
            traci.gui.setOffset("View #0", -400, 60)
            traci.gui.setZoom("View #0", 1500)'''
        # if self.s_step % 10 == 1:
            # self.check_all_leave_requests()
            # self.check_new_cavs_from_ramps()
        self.communicate()
        if self.s_step > self.warmup_duration / self.step_length and not self.join_list:
            self.get_join_list()

        traci.simulationStep()
        self.s_step += 1

    def reset(self):
        self.join_list = []
        if self.s_step == 0:
            self.simulation_reset()
        elif self.s_step * self.step_length >= self.simulation_duration - 10:
            with open("record_simulation.txt", "a") as f:
                # Writing data to a file
                f.write('Failure : ' + str(self.n_failure) + ', Collison : ' + str(self.n_collison) + ', Success : ' + str(self.n_success) + ', Average Reward : ' +
                  str(self.t_reward / (self.n_failure + self.n_success)) + '\n')
            self.simulation_reset()
        else:
            while not self.join_list:
                self.simulation_step()

    def step(self, action):
        self.do_joins(action)
        for i in range(20):
            self.simulation_step()

        d = self.done()
        r = self.rew()
        self.t_reward += r
        self.reward_info = r
        if d == 1:
            if r == -100:
                self.clean_fail_joiner()
                self.n_failure += 1
            elif r == -10000:
                self.n_collison += 1
            else:
                self.n_success += 1
            # o = self.reset()
            '''else:
            o = self.obs()'''
        if d == 1:
            print(d, r, 'action: ', action)
            # print('new joiner : ' + self.join_list[1])
        else:
            print(self.join_list[1], d, r, 'action: ', action)
        # return o, r, d
    def obs(self):
        joiner_id = self.join_list[1]
        platoon_front_id = self.join_list[2]

        veh_list = traci.vehicle.getIDList()
        if joiner_id in veh_list:
            if traci.vehicle.getLeader(joiner_id):
                lane_front_id = traci.vehicle.getLeader(joiner_id)[0]
            else:
                lane_front_id = -1
            lane_follow_id = self.get_veh_follower(joiner_id)
        else:
            lane_front_id = -1
            lane_follow_id = -1
        if platoon_front_id in veh_list:
            platoon_follow_id = self.get_veh_follower(platoon_front_id)
        else:
            platoon_follow_id = -1
        cars = [lane_front_id, joiner_id, lane_follow_id, platoon_front_id, platoon_follow_id]
        p0 = [self.get_veh_pos(v, veh_list)[0] for v in cars]
        p1 = [self.get_veh_pos(v, veh_list)[1] for v in cars]
        v = [self.get_veh_speed(v, veh_list) for v in cars]
        a = [self.get_veh_accel(v, veh_list) for v in cars]

        d1 = math.sqrt((p0[0] - p0[1])**2 + (p1[0] - p1[1])**2) * np.sign((p0[0] - p0[1])) # distance between joiner and its front
        d2 = math.sqrt((p0[2] - p0[1]) ** 2 + (p1[2] - p1[1]) ** 2) * np.sign((p0[2] - p0[1])) # distance between joiner and its follower
        d3 = math.sqrt((p0[3] - p0[4]) ** 2 + (p1[3] - p1[4]) ** 2) * np.sign((p0[3] - p0[4])) # distance between last member of platoon and its follower
        d4 = math.sqrt((p0[1] - p0[3]) ** 2 + (p1[1] - p1[3]) ** 2) * np.sign((p0[1] - p0[3]))

        ds1 = v[0] - v[1] # difference of speed between joiner and its front
        ds2 = v[2] - v[1] # difference of speed between joiner and its follower

        obs = v + a + [d1, d2, d3, d4, ds1, ds2]
        # obs = v + a + [d1, d2, d3, ds1, ds2]
        if joiner_id in veh_list:
            obs.append(traci.vehicle.couldChangeLane(joiner_id, 1))
        else:
            obs.append(-1)
        '''obs.append(self.get_veh_speed(joiner_id))
        obs.append(self.get_veh_accel(joiner_id))
        obs = obs + list(traci.vehicle.getPosition(joiner_id))
        obs.append(traci.vehicle.couldChangeLane(joiner_id, 1))
        if lane_front_id == -1:
            obs.append(-1)
            obs.append(-1)
            obs = obs + [-1, -1]
        else:
            obs.append(self.get_veh_speed(lane_front_id))
            obs.append(self.get_veh_accel(lane_front_id))
            obs = obs + list(traci.vehicle.getPosition(lane_front_id))
        obs.append(self.get_veh_speed(platoon_front_id))
        obs = obs + list(traci.vehicle.getPosition(platoon_front_id))'''
        print("Observe : ", obs)
        return obs

    def rew(self):
        # collision = 0
        joiner_id = self.join_list[1]
        platoon_front_id = self.join_list[2]
        do = self.done()
        if joiner_id in traci.simulation.getCollidingVehiclesIDList() or joiner_id not in traci.vehicle.getIDList():
            r = -10000
            # collision = 1
        else:
            r = -1
            if do == 1:
                if traci.vehicle.getLeader(joiner_id) and traci.vehicle.getLeader(joiner_id)[0] == platoon_front_id:
                    r = 100
                    d = self.get_distance(joiner_id, platoon_front_id)
                    if d < 10:
                        r -= 2 * abs(d - 10)
                    if d > 20:
                        r -= 0.5 * abs(d - 20)
                else:
                    r = -100

        # print("Reward : ", r)
        return r


    def done(self):
        do = 0
        leader_id = self.join_list[0]
        joiner_id = self.join_list[1]
        veh_list = traci.vehicle.getIDList()
        if joiner_id in self.tlp_list or joiner_id not in veh_list or leader_id not in veh_list:
            do = 1
        elif self.get_veh_laneIdx(joiner_id) == self.get_veh_laneIdx(leader_id):
            self.platoons[leader_id]["state"] = 0
            do = 1
        # print("Done : ", do)
        return do

    def info(self):
        return {'l': self.n_success / (self.n_failure + 1), 'r': self.reward_info} if not self.log else self.log_info()

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
        :param plexe: API instance
        :param topology: a dictionary pointing each vehicle id to its front
        vehicle and platoon leader. each entry of the dictionary is a dictionary
        which includes the keys "leader" and "front"
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
        :param plexe: API instance
        :param v1: id of first vehicle
        :param v2: id of the second vehicle
        :return: distance between v1 and v2
        """
        v1_data = self.plexe.get_vehicle_data(v1)
        v2_data = self.plexe.get_vehicle_data(v2)
        return math.sqrt((v1_data[POS_X] - v2_data[POS_X]) ** 2 +
                         (v1_data[POS_Y] - v2_data[POS_Y]) ** 2) - 4

    # initialize controllers of different types of vehicles, and update topology if necessary
    def initialize_traffic_control(self, min_platoon_size=2, max_platoon_size=4):
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
                        if leader_id != 0 and self.platoons[leader_id]["ini_size"] <= len(self.platoons[leader_id]["members"]):
                            self.platoons[vid] = {"ini_size": random.randint(min_platoon_size, max_platoon_size),
                                             "members": [vid], "state": 0}
                        # else, add current vehicle as a member
                        else:
                            self.plexe.set_active_controller(vid, CACC)
                            self.platoons[leader_id]["members"].append(vid)
                            self.topology[vid] = {"front": vid_front, "leader": leader_id}
                            # self.plexe.enable_auto_feed(vid, True, self.topology[vid]["leader"], self.topology[vid]["front"])
        '''return self.platoons, self.topology'''

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
                        # self.plexe.enable_auto_feed(veh, False, self.topology[veh]["leader"], self.topology[veh]["front"])
                        self.topology.pop(veh, None)
                    self.platoons.pop(leader, None)
                elif leader in self.topology.keys():
                    self.topology.pop(leader, None)
        '''return self.platoons, self.topology'''

    def update_quit_vehicle(self, veh):
        # if the car is a leader of platoon
        if veh in self.platoons.keys():
            if len(self.platoons[veh]["members"]) > 1:
                new_leader = self.platoons[veh]["members"][1]
                self.plexe.set_active_controller(new_leader, ACC)
                self.platoons[new_leader] = self.platoons[veh]
                self.platoons[new_leader]["members"].pop(0)
                self.topology.pop(new_leader, None)
                for follower in self.platoons[new_leader]["members"][1:]:
                    self.topology[follower]["leader"] = new_leader
                if self.join_list and veh == self.join_list[0]:
                    self.join_list[0] = new_leader
            self.platoons.pop(veh, None)
        # if the car is a member of platoon
        elif veh in self.topology.keys():
            leader = self.topology[veh]["leader"]
            follower = self.get_following_vehicle(veh)
            self.platoons[leader]["members"].remove(veh)
            # the car is not the last car in the platoon
            if follower != -1:
                self.topology[follower]["front"] = self.topology[veh]["front"]
            # self.plexe.enable_auto_feed(veh, False, leader, self.topology[veh]["front"])
            self.topology.pop(veh, None)
        '''if veh in self.plexe.plexe.lane_changes.keys():
            self.plexe.plexe.lane_changes.pop(veh)
        return self.platoons, self.topology'''

    def change_leader(self, veh):
        new_leader = self.platoons[veh]["members"][1]
        self.platoons[new_leader] = self.platoons[veh]
        self.platoons[new_leader]["members"].pop(0)
        self.platoons.pop(veh, None)
        '''return self.platoons, new_leader'''
        return new_leader

    def split_platoons(self, veh, leader):
        i = self.platoons[leader]["members"].index(veh)
        self.platoons[leader]["members"] = self.platoons[leader]["members"][:i]
        '''return self.platoons, leader'''
        return leader

    def update_quit_vehicle_platoons(self, veh, leader):
        # if the car is a leader of platoon
        if veh in self.platoons.keys():
            if len(self.platoons[veh]["members"]) > 1:
                leader = self.change_leader(veh)
        # if the car is a member of platoon
        else:
            leader = self.split_platoons(veh, leader)
        '''return self.platoons, leader'''
        return leader

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
        '''return self.platoons, self.topology'''

    '''def update_teleport_vehicles(self):
        veh_tlp = traci.simulation.getCollidingVehiclesIDList()
        if len(veh_tlp) != 0:
            print("Vehicle moved:", veh_tlp)
            for veh in veh_tlp:
                self.update_quit_vehicle(veh)
                try:
                    traci.vehicle.remove(veh)
                except:
                    print("Already moved")
            self.tlp_list += veh_tlp
        return self.platoons, self.topology'''

    # get the following vehicle's id according to topology and platoons
    def get_following_vehicle(self, vid):
        if vid in self.topology.keys():
            leader_id = self.topology[vid]["leader"]
        else:
            leader_id = vid
        p = self.platoons[leader_id]["members"]
        i = p.index(vid)
        if i == len(p) - 1:
            return -1
        else:
            return p[i + 1]

    # get all following vehicles' ids according to topology
    def get_all_following_vehicles(self, vid):
        if vid in self.platoons.keys():
            leader_id = vid
        else:
            leader_id = self.topology[vid]["leader"]
        p = self.platoons[leader_id]["members"]
        i = p.index(vid)
        if i == len(p) - 1:
            return []
        else:
            return p[i + 1:]

    def clean_fail_joiner(self):
        leader_id = self.join_list[0]
        joiner_id = self.join_list[1]
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
        # join_speed : in which speed the joiner catches up the target platoon, to calculate or to optimize
        # maybe different cases exist
        self.plexe.set_cc_desired_speed(joiner_id, self.join_speed)
        # plexe.set_active_controller(joiner_id, FAKED_CACC)
        '''return self.topology, self.platoons'''

    def open_gap(self, flist):
        if len(flist) != 0:
            leader = self.topology[flist[0]]["leader"]
            self.plexe.set_active_controller(flist[0], ACC)
            # self.plexe.enable_auto_feed(flist[0], False, leader, self.topology[flist[0]]["front"])
            if len(flist) > 1:
                for vid in flist[1:]:
                    self.topology[vid]["leader"] = flist[0]
                    self.plexe.set_active_controller(vid, FAKED_CACC)
                    self.plexe.set_path_cacc_parameters(vid, distance=self.join_distance)
                    # self.plexe.enable_auto_feed(vid, False, leader, self.topology[vid]["front"])
                    # self.plexe.enable_auto_feed(vid, True, flist[0], self.topology[vid]["front"])
        '''return self.topology'''

    def open_gap_for_leave(self, veh, leader, flist):
        # If it is not the last member, we create a new platoon
        if len(flist) != 0:
            self.plexe.set_active_controller(flist[0], ACC)
            if len(flist) > 1:
                for vid in flist[1:]:
                    self.topology[vid]["leader"] = flist[0]
                    self.plexe.set_active_controller(vid, CACC)
                    self.plexe.set_path_cacc_parameters(vid, distance=self.leave_distance)
        # if the car is a leader of platoon
        if veh in self.platoons.keys():
            if len(self.platoons[veh]["members"]) > 1:
                new_leader = self.platoons[veh]["members"][1]
                self.platoons[new_leader] = self.platoons[veh]
                # self.plexe.enable_auto_feed(new_leader, False, veh, veh)
                # for m in self.platoons[new_leader]["members"][2:]:
                    # self.plexe.enable_auto_feed(m, False, veh, self.topology[m]["front"])
                    # self.plexe.enable_auto_feed(m, True, self.topology[m]["leader"], self.topology[m]["front"])
                self.topology.pop(new_leader, None)
        # if the car is a member of platoon
        else:
            i = self.platoons[leader]["members"].index(veh)
            if i != len(self.platoons[leader]["members"]) - 1:
                follower = self.platoons[leader]["members"][i + 1]
                self.platoons[follower] = {"ini_size": 0, "members": self.platoons[leader]["members"][i + 1:], "state": 0}
                # self.plexe.enable_auto_feed(follower, False, leader, veh)
                for m in self.platoons[follower]["members"]:
                    # self.plexe.enable_auto_feed(m, False, leader, self.topology[m]["front"])
                    # self.plexe.enable_auto_feed(m, True, follower, self.topology[m]["front"])
                    self.topology.pop(follower, None)
            # self.plexe.enable_auto_feed(veh, False, leader, self.topology[veh]["front"])
            self.topology.pop(veh, None)
        '''return self.topology, self.platoons'''

    def reset_leader(self, joiner_id, flist, leader):
        if flist:
            self.topology[flist[0]]["front"] = joiner_id
        for vid in flist:
            # self.plexe.enable_auto_feed(vid, False, flist[0], self.topology[vid]["front"])
            self.topology[vid]["leader"] = leader
            # self.plexe.enable_auto_feed(vid, True, leader, self.topology[vid]["front"])
        '''return self.topology'''

    def finish_join(self, joiner_id, leader_id, flist):
        edge = traci.vehicle.getRoadID(joiner_id)
        lane_num = traci.edge.getLaneNumber(edge)
        traci.vehicle.setSpeedMode(joiner_id, 0)
        traci.vehicle.setVehicleClass(joiner_id, 'hov')
        traci.vehicle.changeLane(joiner_id, lane_num - 1, 1800)
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
        leader_id = self.join_list[0]
        joiner_id = self.join_list[1]
        platoon_front_id = self.join_list[2]
        platoon_behind_id = self.join_list[3]
        flist = self.join_list[4]
        state = self.join_list[5]
        edge = traci.vehicle.getRoadID(joiner_id)
        a1 = action // 10
        a2 = action % 10 - 5
        # Wait too long for the last operation to finish

        if state == self.REQUEST and self.platoons[leader_id]["state"] != 0:
            self.join_list = []
            # print("Join request denied: Platoon occupied", joiner_id, leader_id)
            return -1
        else:
            traci.vehicle.setSpeedMode(joiner_id, 0)
            traci.vehicle.slowDown(joiner_id, max(0, traci.vehicle.getSpeed(joiner_id) + a2), 2)
            self.platoons[leader_id]["state"] = 1
            if state == self.REQUEST:
                # print("Join request approved:", joiner_id, leader_id)
                # traci.vehicle.setColor(leader_id, (50, 100, 0))
                # traci.vehicle.setColor(platoon_front_id, (50, 100, 0))
                self.catch_platoon(joiner_id, leader_id, platoon_front_id)
                self.join_list[5] = self.CATCHING_PLATOON
                '''lane_num = traci.edge.getLaneNumber(edge)
                lane_idx = traci.vehicle.getLaneIndex(joiner_id)
                # first change to the lane near the lane of platoon
                if lane_num - 1 - lane_idx > 1:
                    self.plexe.set_fixed_lane(joiner_id, lane_num - 2, safe=True)
                self.join_list[5] = self.CATCHING_PLATOON
            # if state == CATCHING_PLATOON and get_distance(plexe, joiner_id, platoon_front_id) < join_distance + 10:
            if state == self.CATCHING_PLATOON and self.get_distance(joiner_id, platoon_front_id) < self.join_distance + 10:
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
                    except KeyError:
                        pass
                        # print(joiner_id)
                elif traci.vehicle.getLaneIndex(joiner_id) != lane_num:
                    traci.vehicle.changeLane(joiner_id, lane_num - 1, 1800)
                    self.plexe.set_fixed_lane(joiner_id, lane_num - 1, safe=False)
                    self.plexe.set_cc_desired_speed(joiner_id, self.SPEED_CAV_LANES)
                    traci.vehicle.updateBestLanes(joiner_id)


    '''def do_joins(self, action):
        leader_id = self.join_list[0]
        joiner_id = self.join_list[1]
        platoon_front_id = self.join_list[2]
        platoon_behind_id = self.join_list[3]
        flist = self.join_list[4]
        state = self.join_list[5]
        edge = traci.vehicle.getRoadID(joiner_id)
        # Wait too long for the last operation to finish
        if state == self.REQUEST and self.platoons[leader_id]["state"] != 0:
            self.join_list = []
            print("Join request denied: Platoon occupied", joiner_id, leader_id)
        else:
            self.platoons[leader_id]["state"] = 1
            if state == self.REQUEST:
                print("Join request approved:", joiner_id, leader_id)
                traci.vehicle.setColor(leader_id, (50, 100, 0))
                traci.vehicle.setColor(platoon_front_id, (50, 100, 0))
                self.catch_platoon(joiner_id, leader_id, platoon_front_id)
                lane_num = traci.edge.getLaneNumber(edge)
                lane_idx = traci.vehicle.getLaneIndex(joiner_id)
                # first change to the lane near the lane of platoon
                if lane_num - 1 - lane_idx > 1:
                    self.plexe.set_fixed_lane(joiner_id, lane_num - 2, safe=True)
                self.join_list[5] = self.CATCHING_PLATOON
            # if state == CATCHING_PLATOON and get_distance(plexe, joiner_id, platoon_front_id) < join_distance + 10:
            if state == self.CATCHING_PLATOON and self.get_distance(joiner_id, platoon_front_id) < self.join_distance + 10:
                self.open_gap(flist)
                self.join_list[5] = self.OPENING_GAP

            if state == self.OPENING_GAP:
                # when the gap is large enough, complete the maneuver
                dj = traci.vehicle.getLanePosition(joiner_id)
                dp = traci.vehicle.getLanePosition(platoon_front_id)

                lane_num = traci.edge.getLaneNumber(edge)
                lane_idx = traci.vehicle.getLaneIndex(joiner_id)

                if lane_num - 1 - lane_idx == 1 and dj < dp - self.DISTANCE and traci.vehicle.getRoadID(
                        platoon_front_id) == edge and \
                        (platoon_behind_id == -1 or self.get_distance(platoon_front_id, platoon_behind_id) > \
                         2 * self.join_distance + 0):
                    self.finish_join(joiner_id, leader_id)
                    self.join_list = []
                    self.platoons[leader_id]["state"] = 0
                '''

    def check_new_cavs_from_ramps(self):
        '''multiprocess maybe'''
        if self.s_step % 30 == 0:
            last_buffer = copy.deepcopy(self.buffer_list)
            self.buffer_list = []
            for v in last_buffer:
                self.plexe.set_active_controller(v, ACC)
                '''plexe.set_active_controller(v, ACC)
                traci.vehicle.setLaneChangeMode(v, lcm=218)'''
                self.plexe.set_cc_desired_speed(v, SPEED)
        for e in self.edges:
            if traci.edge.getLastStepVehicleIDs(e):
                vehs = [v for v in traci.edge.getLastStepVehicleIDs(e)
                        if v not in self.merge_list and "on" in v and traci.vehicle.getLaneIndex(
                        v) != 0 and traci.vehicle.getTypeID(v) == "cav2"]
                self.merge_list += vehs
                self.buffer_list += vehs
        '''return buffer_list, merge_list'''

    def check_leave_request_for_veh(self, route, veh_id):
        if route in traci.vehicle.getRouteID(veh_id):
            return True
        else:
            return False

    def check_leave_request_for_edge(self, route, edge, pass_list):
        lane_idx = traci.edge.getLaneNumber(edge) - 1
        if traci.edge.getLastStepVehicleIDs(edge):
            for v in traci.edge.getLastStepVehicleIDs(edge):
                if v not in pass_list and self.check_leave_request_for_veh(route, v):
                    if traci.vehicle.getLaneIndex(v) == lane_idx:
                        if v in self.topology.keys():
                            self.leave_list.append([self.topology[v]["leader"], v])
                        else:
                            self.leave_list.append([v, v])
                    # In normal lanes
                    else:
                        self.plexe.set_active_controller(v, DRIVER)
                    pass_list.append(v)
        return pass_list

    def check_all_leave_requests(self):
        for r in self.route_edge.keys():
            self.pass_list[r] = self.check_leave_request_for_edge(r, self.route_edge[r], self.pass_list[r])


    def do_leaves(self):
        new_leave_list = copy.deepcopy(self.leave_list)
        # Get all leave requests
        for i in range(len(self.leave_list)):
            leader_id = self.leave_list[i][0]
            leave_id = self.leave_list[i][1]
            state = self.platoons[leader_id]["state"]
            # The car is not the very first car
            '''if traci.vehicle.getLeader(leave_id) is not None:
                front_id = traci.vehicle.getLeader(leave_id)[0]
            else:
                front_id = -1'''
            follower = -1
            platoon = self.platoons[leader_id]["members"]
            j = platoon.index(leave_id)
            if j != len(platoon) - 1:
                follower = platoon[j + 1]
            # The platoon is available
            if state == 0:
                # traci.vehicle.setColor(leave_id, (50, 100, 0))
                is_leader = leave_id in self.platoons.keys()
                flist = self.get_all_following_vehicles(leave_id)
                self.open_gap_for_leave(leave_id, leader_id, flist)
                if is_leader == 0:
                    self.plexe.set_active_controller(leave_id, ACC)
                state = self.OPENING_GAP
            if state == self.OPENING_GAP:
                if follower == -1:
                    d = 1000
                else:
                    d = self.get_distance(leave_id, follower)
                if d > self.leave_distance:
                    self.plexe.set_fixed_lane(leave_id, 1, safe=True)
                    traci.vehicle.setMaxSpeed(leave_id, self.SPEED)
                    self.plexe.set_active_controller(leave_id, DRIVER)
                    state = self.CHANGING_LANE
            if state == self.CHANGING_LANE and traci.vehicle.getLaneIndex(leave_id) != str(int(traci.edge.getLaneNumber(traci.vehicle.getRoadID(leave_id))) - 1):
                leader_id = self.update_quit_vehicle_platoons(leave_id, leader_id)
                state = self.COMPLETED
                self.plexe.disable_fixed_lane(leave_id)
                traci.vehicle.setVehicleClass(leave_id, 'passenger')
                # print("vehicle leaved from platoons:", self.leave_list[i])
                new_leave_list.remove(self.leave_list[i])
            self.platoons[leader_id]["state"] = state
        self.leave_list = new_leave_list
        '''return new_leave_list, topology, platoons'''

    ####################################################################################################################
    ####################################################################################################################
    # control

    # get the list of ids of leaders of platoons, which are behinds the cav and in the same edge
    def get_platoons_available(self, edge, veh):
        platoons_available = []
        # getLanePosition return : the distance from the front bumper to the start of the lane in [m]
        p_v = traci.vehicle.getLanePosition(veh)
        # ids of leaders of platoons in the same edge
        for leader in set(self.platoons.keys()).intersection(set(traci.edge.getLastStepVehicleIDs(edge))):

            if self.platoons[leader]["state"] == 0:
                fronter = self.platoons[leader]["members"][-1]
                p_p = traci.vehicle.getLanePosition(fronter)
                # print(p_p)
                # only record those behinds the cav, name and position
                if p_p <= p_v + 15:
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
        t = 0
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
                    self.join_list = [p, joiner, position, -1,
                                      -1, 0]
            t += 1


    ####################################################################################################################
    ####################################################################################################################
    # lane

    def get_lane_veh_ids(self, lane_id):
        return traci.lane.getLastStepVehicleIDs(lane_id)

    def get_lane_veh_n(self, lane_id):
        return traci.lane.getLastStepVehicleNumber(lane_id)

    def get_lane_length(self, lane_id):
        return traci.lane.getLength(lane_id)

    def get_lane_veh_n_in_dist(self, lane_id, dist):
        return sum([1 for veh_id in self.get_lane_veh_ids(lane_id)
                    if (self.get_lane_length(lane_id) - self.get_veh_pos_on_lane(veh_id)) <= dist])

    def get_lane_veh_ids_in_dist(self, lane_id, dist):
        return [veh_id for veh_id in self.get_lane_veh_ids(lane_id)
                if (self.get_lane_length(lane_id) - self.get_veh_pos_on_lane(veh_id)) <= dist]

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
            return -1
        else:
            return traci.vehicle.getSpeed(veh_id)

    def get_veh_accel(self, veh_id, veh_list):
        if veh_id == -1 or veh_id not in veh_list:
            return -1
        else:
            return traci.vehicle.getAcceleration(veh_id)

    def get_veh_lane(self, veh_id):
        return traci.vehicle.getLaneID(veh_id)

    def get_veh_edge(self, veh_id):
        return traci.vehicle.getRoadID(veh_id)

    def get_veh_laneIdx(self, veh_id):
        return traci.vehicle.getLaneIndex(veh_id)

    def get_veh_pos_on_lane(self, veh_id):
        return traci.vehicle.getLanePosition(veh_id)

    def get_veh_dist_from_junction(self, veh_id):
        return self.get_lane_length(self.get_veh_lane(veh_id)) - self.get_veh_pos_on_lane(veh_id)

    def get_veh_waiting_time(self, veh_id):
        return traci.vehicle.getWaitingTime(veh_id)

    def get_veh_accumulated_waiting_time(self, veh_id):
        return traci.vehicle.getAccumulatedWaitingTime(veh_id)

    def get_veh_delay(self, veh_id):
        return 1 - (self.get_veh_speed(veh_id) / self.args["v_max_speed"])

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



    ####################################################################################################################
    ####################################################################################################################

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

    def generate_random_flow(self, simDuration=1800, period=600, limit_of_flow_m=(2000, 4000),
                             limit_of_flow_r=(200, 400)):
        '''limit_of_flow : [lower bound, upper bound]'''
        routes_list = self.routes
        n_period = int(simDuration / period)
        flow = {}
        for route in routes_list:
            # The origin or the destination is ramp, the flow is more limited
            if 'o' in route.lower():
                flow[route] = np.random.randint(low=limit_of_flow_r[0], high=limit_of_flow_r[1], size=(n_period,))
            # Suppose the main flow is always in freeway
            else:
                flow[route] = np.random.randint(low=limit_of_flow_m[0], high=limit_of_flow_m[1], size=(n_period,))
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
                                                        "vehsPerHour": str(flow[r][i]),
                                                        "type": vtype,
                                                        "route": r})
                e.tail = "\n\t"
                '''if i == n_period - 1 and r == list(flow.keys())[-1]:
                    e.tail = "\n"
                else:
                    e.tail = "\n\t"'''
        '''p_flow = {'m': random.randint(500, 1200), 'm_off1': random.randint(10, 200),
                    'm_off2': random.randint(100, 600)}'''
        p_flow = {'m': random.randint(1500, 2400)}
        for r in p_flow.keys():
            e = ET.SubElement(root, 'flow', attrib={"id": 'p_' + r,
                                                    "begin": str(0),
                                                    "end": str(simDuration),
                                                    "departPos": "last",
                                                    "departSpeed": "max",
                                                    "departLane": str(self.p_lane),
                                                    "vehsPerHour": str(p_flow[r]),
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
        self.cav_p_rate = random.randint(0, 1000) / 1000
        return self.cav_p_rate

    def set_penetration_rate(self):
        filename = self.path + "freeway.rou.xml"
        tree = ET.parse(filename)
        root = tree.getroot()
        for vtype in root.iter('vtype'):
            if vtype.attrib['id'] == 'cav':
                vtype.set('probability', str(self.cav_p_rate / (1 - self.cav_p_rate)))
        with open(filename, 'wb') as f:
            tree.write(f)



    """
    def lambda_veh_p_second(self, veh_p_s):
        return 1 / veh_p_s
    """

    def lambda_veh_p_hour(self, veh_p_h):
        return 3600 / veh_p_h

    def insert_lambdas(self):
        lambdas = [self.lambda_veh_p_hour(random.randint(1, 10) * 100) for _ in self.flow_logic]
        return lambdas if self.rnd[1] else [self.lambda_veh_p_hour(f) for f in self.args["veh_p_hour"]]


    ####################################################################################################################
    ####################################################################################################################

    # Connected vehicles

    def is_veh_con(self, veh_id):
        return self.get_veh_type(veh_id) == self.args["v_type_con"]

    ####################################################################################################################
    ####################################################################################################################

    # Log info

    def log_info(self):
        veh_n = 0
        sum_delay, sum_waiting_time, sum_queue_length, sum_acc_waiting_time = 0, 0, 0, 0

        for tl_id in self.tl_ids:
            for veh_id in self.yield_tl_vehs(tl_id):
                veh_n += 1
                sum_delay += self.get_veh_delay(veh_id)
                wt = self.get_veh_waiting_time(veh_id)
                sum_waiting_time += wt
                sum_acc_waiting_time += self.get_veh_accumulated_waiting_time(veh_id)
                if wt > 0:
                    sum_queue_length += 1

        avg_delay = 0 if veh_n == 0 else sum_delay / veh_n
        avg_waiting_time = 0 if veh_n == 0 else sum_waiting_time / veh_n
        avg_acc_waiting_time = 0 if veh_n == 0 else sum_acc_waiting_time / veh_n


        return {
            "id": type(self).__name__.lower(),
            "ep": self.ep_count,
            "cav_p_rate": self.con_p_rate,
            "veh_n_p_hour": json.dumps(self.veh_n_p_hour),
            "veh_n": veh_n,
            "sum_delay": sum_delay,
            "sum_waiting_time": sum_waiting_time,
            "sum_acc_waiting_time": sum_acc_waiting_time,
            "sum_queue_length": sum_queue_length,
            "avg_delay": avg_delay,
            "avg_waiting_time": avg_waiting_time,
            "avg_acc_waiting_time": avg_acc_waiting_time,
        }
