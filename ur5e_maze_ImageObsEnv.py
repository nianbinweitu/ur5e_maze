import gym
from gym import spaces
import math
import numpy as np
import pybullet
from random import *
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict
from time import sleep
from stable_baselines3.common.env_checker import check_env
import maze


class ImageObsUR5Sim(gym.Env):
    def __init__(self, method=pybullet.DIRECT, mode="train"):
        """parameters config"""
        self.method = method
        self.mode = mode
        '''pybullet parameters'''
        pybullet.connect(self.method)
        pybullet.setRealTimeSimulation(True)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
        # pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        # pybullet.loadURDF("plane.urdf", [0, 0, -0.6300])
        # pybullet.setGravity(0, 0, -9.8)
        pybullet.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=90,
            cameraPitch=-60,
            cameraTargetPosition=[0.3, 0, 0.2]
        )
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

        '''ur5 parameters'''
        self.ur5 = self.load_robot()
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
                pybullet.setJointMotorControl2(self.ur5, info.id, pybullet.POSITION_CONTROL, targetVelocity=0, force=10_000)
            self.joints[info.name] = info
            # print(info)

        '''puzzle parameters'''
        self.block_size = 0.05  # 方块长宽，高为0.01
        self.center_x, self.center_y, self.center_z = 0.5, 0, 0.3  # puzzle的中心坐标
        self.nx, self.ny = 8, 8  # x,y方向方块阵列数

        self.m = None
        self.puzzle = None
        self.puzzle_wall = None
        self.start_point = None  # 机械臂操作任务末端的初始位置
        self.current_block_center = None  # 记录end所在方块的中心
        self.middle_point = None  # 机械臂到达操作初始位置的过渡点，避免碰撞
        self.target_point = None  # 机械臂操作任务末端的目标位置
        self.block = self.load_block()  # puzzle形状设置，并且获取上述几个坐标

        '''camera parameters'''
        self.camera_pos = [0.51, 0., 0.6]
        self.target_pos = [0.5, 0., 0.3]
        self.fig_width = 84
        self.fig_height = 84

        '''gym.Env parameters'''
        # 记录停留在一块的timestep
        self.last_block_reward = 0
        self.last_point_reward = 0
        self.one_block_step = 0

        # 记录时间步数
        self.step_count = 0

        # 动作为x,y轴方向的移动
        self.end_act_high = 0.02
        act_high = np.array([self.end_act_high] * 2, dtype=np.float32)
        self.action_space = spaces.Box(-act_high, act_high)

        # 观测来源于camera

        # # 1.depth
        # self.observation_space = spaces.Box(low=0.9, high=1.0000001,
        #                                     shape=(self.fig_width, self.fig_height),
        #                                     dtype=np.float32)

        # 2.rgb/rgba
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.fig_width, self.fig_height, 3),
                                            dtype=np.uint8)
        # print(self.observation_space.sample())

    '''robot operating'''

    def load_robot(self):
        # table_urdf_path = os.path.join(pybullet_data.getDataPath(), "table/table.urdf")
        robot_urdf_path = "ur_e_description/urdf/ur5e_with_stick.urdf"
        flags = pybullet.URDF_USE_SELF_COLLISION
        # table = pybullet.loadURDF(table_urdf_path, [0.5, 0, -0.6300], [0, 0, 0, 1])
        robot = pybullet.loadURDF(robot_urdf_path, [0, 0, 0], [0, 0, 0, 1], flags=flags)
        return robot

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

    def load_block(self):
        block_size = self.block_size
        center_x = self.center_x
        center_y = self.center_y
        center_z = self.center_z
        nx = self.nx
        ny = self.ny
        self.m = maze.Maze(nx, ny)
        puzzle, puzzle_wall = self.m.maze, self.m.maze_shape
        self.puzzle = puzzle
        self.puzzle_wall = puzzle_wall
        # puzzle左上角block中心的世界坐标
        y0 = center_y - block_size * (ny - 1) / 2
        x0 = center_x - block_size * (nx - 1) / 2
        # self.puzzle矩阵中入口和出口的序号
        start_0 = np.where(puzzle_wall[0] == 0)
        target_0 = np.where(puzzle_wall[-1] == 0)
        if len(start_0[0]) == 1:
            start_0 = start_0[0][0]
            if len(target_0[0]) == 1:
                target_0 = target_0[0][0]
            else:
                raise AssertionError("puzzle出口数不唯一")
        else:
            raise AssertionError("puzzle入口数不唯一")
        # puzzle中入口block和出口block的中心坐标
        y_start = y0
        x_start = x0 + block_size * start_0
        y_target = center_y + block_size * (ny - 1) / 2
        x_target = x0 + block_size * target_0
        # 机械臂末端end的任务起始点，到达任务起始点的过渡点，目标点
        self.start_point = np.array([x_start, y_start, center_z + 0.025])
        self.middle_point = np.array([x_start, y_start - block_size * 2, center_z + 0.025])
        self.target_point = np.array([x_target, y_target + block_size, center_z + 0.025])
        self.current_block_center = self.start_point
        block_urdf_path = "ur_e_description/urdf/block.urdf"
        blocks = list([])
        for i in range(nx):
            for j in range(ny):
                if puzzle[j, i] == -1:
                    single_block = pybullet.loadURDF(block_urdf_path,
                                                     [x0 + block_size * i, y0 + block_size * j, center_z],
                                                     pybullet.getQuaternionFromEuler([0, 0, 0]),
                                                     useFixedBase=True)
                    blocks.append(single_block)
        return blocks

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

    def check_end_lock(self):
        lock_state = pybullet.getLinkState(self.ur5, 8, computeForwardKinematics=True)
        plate_position = np.array(lock_state[0], dtype=np.float32)
        plate_z = plate_position[2]  # plate的z坐标
        end_position, end_orientation = self.get_current_end()
        end_z = end_position[2]  # end_effector的z坐标
        end_angle = abs(end_orientation[1] - math.pi / 2)  # end_effector和z轴夹角
        return True if plate_z < self.center_z < end_z and end_angle < math.pi / 18 else False

    def get_state(self):
        _, _, rgba, depth, seg = self.set_stable_camera()
        rgb = rgba[:, :, :3]
        return rgb

    def get_end2target_distance(self, mode, target_point):
        end_position, _ = self.get_current_end()
        if mode == 1:
            return sum([(target_point[i] - end_position[i]) ** 2 for i in range(2)])  # xy平面投影点，直线距离的平方
        elif mode == 2:
            return np.amax(np.abs([target_point[i] - end_position[i] for i in range(2)]))  # xy平面投影点，沿x轴或y轴的最大距离
        elif mode == 3:
            return target_point[1] - end_position[1]  # xy平面投影点，y轴坐标差
        else:
            pass

    def robot_action(self, act):
        end_position, _ = self.get_current_end()
        if act[0] == 0 and act[1] == 0:
            return end_position
        else:
            new_end_position = [end_position[0] + act[0], end_position[1] + act[1], self.start_point[2]]
            new_joints_angles = self.calculate_ik(new_end_position, [0, math.pi / 2, 0])
            self.set_joint_angles(new_joints_angles)
            if self.mode == "test":
                sleep(0.1)
            else:
                for _ in range(30):
                    pybullet.stepSimulation()

    def get_reward_and_done(self, reward_mode="shaped"):
        end_position, _ = self.get_current_end()
        pos = [end_position[0] - (self.center_x - self.block_size * self.nx / 2),
               end_position[1] - (self.center_y - self.block_size * self.ny / 2)]  # end相对迷宫的左上角的xy坐标
        in_puzzle = bool(0 < pos[0] < self.block_size * self.nx and 0 < pos[1] < self.block_size * self.ny)
        in_target_block = bool(self.get_end2target_distance(2, self.target_point) < self.block_size / 2)
        out_of_workplace = not (in_puzzle or in_target_block)
        out_of_lock = not self.check_end_lock()
        in_accident = out_of_workplace or out_of_lock

        if in_puzzle:
            block_reward, point_reward = self.m.read_reward(pos=pos, block_size=self.block_size)
        elif in_target_block:
            block_reward, point_reward = 1., 1.
        else:
            block_reward, point_reward = 0., 0.

        if reward_mode == "sparse":
            reward = 200 if in_target_block else 0  # sparse reward
        elif reward_mode == "shaped":
            reward_progress = (point_reward - self.last_point_reward) if not in_accident else 0
            reward_success = 1. * in_target_block
            reward_accident = -1. * in_accident
            reward = reward_progress + reward_success + reward_accident
        else:
            raise AssertionError("请选择奖励模式:sparse/shaped！")

        done = bool(self.step_count >= 500
                    or in_target_block
                    or in_accident)

        self.last_point_reward = 0 if done else point_reward

        return reward, done

    def step(self, act):
        self.robot_action(act)
        self.step_count += 1
        state = self.get_state()
        reward, done = self.get_reward_and_done()
        return state, reward, done, {}

    def reset_robot(self):
        for i in range(20):  # large_scale movement
            middle_joint_angles = self.calculate_ik(self.middle_point, [0, math.pi / 2, 0])
            self.set_joint_angles(middle_joint_angles)
            for _ in range(30):
                pybullet.stepSimulation()
        for i in range(20):
            reset_joint_angles = self.calculate_ik(self.start_point, [0, math.pi / 2, 0])
            self.set_joint_angles(reset_joint_angles)
            for _ in range(30):
                pybullet.stepSimulation()

    def reset(self):
        # 去除方块
        for single_block in self.block:
            pybullet.removeBody(single_block)
        self.block = []
        # 机械臂避开方块区域
        self.set_joint_angles([0.0] * 6)
        for _ in range(960):
            pybullet.stepSimulation()
        # 重置方块
        self.block = self.load_block()
        # 重置机械臂
        self.reset_robot()
        # episode时间步数清零
        self.step_count = 0
        return self.get_state()

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        pass


if __name__ == "__main__":
    sim = ImageObsUR5Sim(method=pybullet.GUI)
    check_env(sim)
    print("check over!!!")
