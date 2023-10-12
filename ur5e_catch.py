import os
import math
import numpy as np
import time
import pybullet
import random
from datetime import datetime
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict
from random import *
import gym
from gym import spaces
from time import sleep
from stable_baselines3.common.env_checker import check_env
import maze

'''
ur5e 抓取物体的简单尝试，基于joint生成和解除
'''
class UR5eCatch(gym.Env):
    def __init__(self, method=pybullet.GUI, mode="train"):
        """parameters config"""
        self.method = method
        self.mode = mode
        '''pybullet parameters'''
        pybullet.connect(self.method)
        pybullet.setRealTimeSimulation(True)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.loadURDF("plane.urdf", [0, 0, -0.6300])
        pybullet.setGravity(0, 0, -9.8)
        pybullet.resetDebugVisualizerCamera(
            cameraDistance=1.2,
            cameraYaw=0,
            cameraPitch=-30,
            cameraTargetPosition=[0.45, 0, 0.2]
        )
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

        '''ur5 parameters'''
        self.ur5 = self.load_robot()
        self.ball = self.load_ball()
        self.end_effector_index = 7
        self.num_joints = pybullet.getNumJoints(self.ur5)
        self.control_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                               "wrist_2_joint", "wrist_3_joint"]
        self.joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joint_info = namedtuple("jointInfo",
                                     ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity",
                                      "controllable"])
        self.joints = AttrDict()
        for i in range(self.num_joints):
            info = pybullet.getJointInfo(self.ur5, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if jointName in self.control_joints else False
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce,
                                   jointMaxVelocity, controllable)
            if info.type == "REVOLUTE":
                pybullet.setJointMotorControl2(self.ur5, info.id, pybullet.POSITION_CONTROL, targetVelocity=0,
                                               force=10_000)
            self.joints[info.name] = info
            # print(info)

        '''camera parameters'''
        self.camera_pos = [0.51, 0., 0.6]
        self.target_pos = [0.5, 0., 0.3]
        self.fig_width = 84
        self.fig_height = 84

        '''gym.Env parameters'''

        # 记录时间步数
        self.step_count = 0

        # 动作为x,y轴方向的移动
        self.end_act_high = 0.02
        act_high = np.array([self.end_act_high] * 2, dtype=np.float32)
        self.action_space = spaces.Box(-act_high, act_high)

        # 观测来源于camera
        # 2.rgb/rgba
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.fig_width, self.fig_height, 3),
                                            dtype=np.uint8)
        # print(self.observation_space.sample())

    '''robot operating'''

    def load_robot(self):
        robot_urdf_path = "/ur_e_description/urdf/ur5e.urdf"
        flags = pybullet.URDF_USE_SELF_COLLISION
        table = pybullet.loadURDF("table/table.urdf", [0.4, 0, -0.6300],
                                  pybullet.getQuaternionFromEuler([0, 0, math.pi / 2]))
        robot = pybullet.loadURDF(robot_urdf_path, [0, 0, 0], [0, 0, 0, 1], flags=flags)
        plate1 = pybullet.loadURDF("tray/traybox.urdf", [0.5, 0.35, 0], [0, 0, 0, 1], useMaximalCoordinates=True)
        plate2 = pybullet.loadURDF("tray/traybox.urdf", [0.5, -0.35, 0], [0, 0, 0, 1], useMaximalCoordinates=True)
        return robot

    def load_ball(self):
        ball = []
        for i in range(5):
            single_ball = pybullet.loadURDF("ur_e_description/urdf/block_joint.urdf",
                                            basePosition=[(random() - 0.5) * 0.2 + 0.5,
                                                          (random() - 0.5) * 0.2 - 0.35,
                                                          0.1])
            ball.append(single_ball)
        return ball

    # 设置各关节角
    def set_joint_angles(self, joint_angles):
        poses = []
        indexes = []
        forces = []
        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        pybullet.setJointMotorControlArray(
            self.ur5, indexes,
            pybullet.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0] * len(poses),
            positionGains=[0.04] * len(poses), forces=forces
        )

    # 返回当前各关节角，列表
    def get_joint_angles(self):
        j = pybullet.getJointStates(self.ur5, list(range(1, 6)))  # tuple
        joints = list([i[0] for i in j])
        return joints

    # 检测是否有碰撞，返回true/false
    def check_collisions(self):
        collisions = pybullet.getContactPoints()
        if len(collisions) > 0:
            # print("[Collision detected!] {}".format(datetime.now()))
            return True
        return False

    # 逆运动学求解，返回各关节角
    def calculate_ik(self, position, orientation):
        quaternion = pybullet.getQuaternionFromEuler(orientation)
        lower_limits = [-math.pi] * 6
        upper_limits = [math.pi] * 6
        joint_ranges = [2 * math.pi] * 6
        rest_poses = [0, -math.pi / 2, -math.pi / 2, -math.pi / 2, -math.pi / 2, 0]
        joints_angles = pybullet.calculateInverseKinematics(
            self.ur5, self.end_effector_index, position, quaternion,
            jointDamping=[0.01] * 6, upperLimits=upper_limits,
            lowerLimits=lower_limits, jointRanges=joint_ranges,
            restPoses=rest_poses
        )
        return joints_angles

    def add_gui_sliders(self):
        self.sliders = []
        self.sliders.append(pybullet.addUserDebugParameter("X", 0, 1, 0.4))
        self.sliders.append(pybullet.addUserDebugParameter("Y", -1, 1, 0))
        self.sliders.append(pybullet.addUserDebugParameter("Z", 0.1, 1, 0.45))
        self.sliders.append(pybullet.addUserDebugParameter("Rx", -math.pi / 2, math.pi / 2, 0))
        self.sliders.append(pybullet.addUserDebugParameter("Ry", -math.pi / 2, math.pi / 2, math.pi / 2))
        self.sliders.append(pybullet.addUserDebugParameter("Rz", -math.pi / 2, math.pi / 2, 0))

    def read_gui_sliders(self):
        x = pybullet.readUserDebugParameter(self.sliders[0])
        y = pybullet.readUserDebugParameter(self.sliders[1])
        z = pybullet.readUserDebugParameter(self.sliders[2])
        rx = pybullet.readUserDebugParameter(self.sliders[3])
        ry = pybullet.readUserDebugParameter(self.sliders[4])
        rz = pybullet.readUserDebugParameter(self.sliders[5])
        return [x, y, z, rx, ry, rz]

    # 返回当前机械臂末端位姿
    def get_current_end(self):
        linkstate = pybullet.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
        end_position = np.array(linkstate[0], dtype=np.float32)
        end_orientation = np.array(pybullet.getEulerFromQuaternion(linkstate[1]), dtype=np.float32)
        return end_position, end_orientation

    '''camera'''

    def set_stable_camera(self):
        camera_pos = np.array(self.camera_pos)  # [-0.2, 0., 1.0]
        target_pos = np.array(self.target_pos)  # [0.1, -0., -0.9]
        view_matrix = pybullet.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=np.array([0, 0, 1]),
            physicsClientId=0
        )
        projection_matrix = pybullet.computeProjectionMatrixFOV(
            fov=90.0,  # 摄像头的视线夹角
            aspect=1.0,
            nearVal=0.01,  # 摄像头焦距下限
            farVal=10,  # 摄像头能看上限
            physicsClientId=0
        )
        width, height, rgba_img, depth_img, seg_img = pybullet.getCameraImage(
            width=self.fig_width, height=self.fig_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            physicsClientId=0
        )
        return width, height, rgba_img, depth_img, seg_img

    '''gym.Env'''

    def get_target_position(self):
        positions = []
        for i in self.ball:
            ball_state = pybullet.getBasePositionAndOrientation(i)  # tuple
            ball_position = np.array(ball_state[0], dtype=np.float32)
            positions.append(ball_position)
        return positions

    def get_state(self):
        pass

    def get_end2target_distance(self, mode, target_point):
        pass

    def robot_action(self, act):
        pass

    def get_reward_and_done(self, reward_mode="shaped"):
        pass

    def step(self, act):
        pass

    def reset_robot(self):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        pass


def demo_simulation():
    """ Demo program showing how to use the sim """
    sim = UR5eCatch()
    sim.add_gui_sliders()
    # x, y, z, Rx, Ry, Rz = sim.read_gui_sliders()
    # joint_angles = sim.calculate_ik([x, y, z], [Rx, Ry, Rz])
    # sim.set_joint_angles(joint_angles)
    # # createConstraint:(父机器人编号，父机器人的连杆序号，子机器人编号，子机器人连杆编号（-1指base），关节类型，关节转轴，连接的位置，父框架位置，子框架位置)
    # jointId2 = pybullet.createConstraint(sim.ur5, sim.end_effector_index + 1, sim.ball, -1, pybullet.JOINT_FIXED,
    #                                      [0, 0, 0], [0., 0., 0.025], [0, 0, 0], childFrameOrientation=[0, 1, 0, 1])
    while True:
        x, y, z, Rx, Ry, Rz = sim.read_gui_sliders()
        joint_angles = sim.calculate_ik([x, y, z], [Rx, Ry, Rz])
        sim.set_joint_angles(joint_angles)
        sim.check_collisions()
        print(sim.get_target_position())


def track_oneball():
    sim = UR5eCatch()
    while True:
        balls = sim.get_target_position()
        ball1 = balls[0]
        x = ball1[0]
        y = ball1[1]
        joint_angles = sim.calculate_ik([x, y, 0.3], [0, math.pi / 2, 0])
        sim.set_joint_angles(joint_angles)
        sim.check_collisions()


def touch_allballs():
    sim = UR5eCatch()
    for i in range(10):
        joint_angles = sim.calculate_ik([0.5, -0.35, 0.3], [0, math.pi / 2, 0])
        sim.set_joint_angles(joint_angles)
        for i in range(240):
            pybullet.stepSimulation()
    sleep(1)
    mode = 0
    newjoint = None
    for i in range(len(sim.ball)):
        while mode < 5000:
            balls = sim.get_target_position()
            target_ball = balls[i]

            if mode <= 50:
                joint_angles = sim.calculate_ik([target_ball[0], target_ball[1], 0.3], [0, math.pi / 2, 0])
                sim.set_joint_angles(joint_angles)
                sleep(0.001)

            elif 1000 < mode <= 2000:
                joint_angles = sim.calculate_ik([target_ball[0], target_ball[1], 0.1], [0, math.pi / 2, 0])
                sim.set_joint_angles(joint_angles)
                sleep(0.001)
                if mode == 2000:
                    newjoint = pybullet.createConstraint(sim.ur5, sim.end_effector_index + 1, sim.ball[i], -1, pybullet.JOINT_FIXED,
                                                         [0, 0, 0], [0., 0., 0.025], [0, 0, 0], childFrameOrientation=[0, 1, 0, 1])

            elif 3000 < mode <= 3500:
                joint_angles = sim.calculate_ik([target_ball[0], target_ball[1], 0.3], [0, math.pi / 2, 0])
                sim.set_joint_angles(joint_angles)
                sleep(0.001)

            elif 3500 < mode <= 4000:
                joint_angles = sim.calculate_ik([0.5, 0.35, 0.3], [0, math.pi / 2, 0])
                sim.set_joint_angles(joint_angles)
                sleep(0.001)
                if mode == 4000:
                    pybullet.removeConstraint(newjoint)

            mode += 1

        mode = 0


if __name__ == "__main__":
    touch_allballs()
