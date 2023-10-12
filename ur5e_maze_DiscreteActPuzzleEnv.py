import gym
from gym.spaces import Box, Discrete
from math import pi, sin, cos
import numpy as np
import pybullet
import os
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict
from time import sleep
from stable_baselines3.common.env_checker import check_env
import maze


class DiscreteActPuzzle(gym.Env):
    def __init__(self, method=pybullet.DIRECT, mode="train"):
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
                pybullet.setJointMotorControl2(self.ur5, info.id, pybullet.VELOCITY_CONTROL, targetVelocity=0, force=0)
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

        '''gym.Env parameters'''
        # 记录上一步所在block的reward
        self.last_block_reward = 0

        # 记录时间步数
        self.step_count = 0

        # 动作为 +-xy 轴方向的 1 block 长度的移动
        self.action_space = Discrete(4)

        # 观测:puzzle_block矩阵，0空，1墙，2end
        self.observation_space = Box(low=0., high=2.,
                                     shape=(self.ny, self.nx),
                                     dtype=np.float32)

    '''robot operating'''

    def load_robot(self):
        table_urdf_path = os.path.join(pybullet_data.getDataPath(), "table/table.urdf")
        robot_urdf_path = "/ur_e_description/urdf/ur5e_with_stick.urdf"
        flags = pybullet.URDF_USE_SELF_COLLISION
        table = pybullet.loadURDF(table_urdf_path, [0.5, 0, -0.6300], [0, 0, 0, 1])
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
        lower_limits = [-pi] * 6
        upper_limits = [pi] * 6
        joint_ranges = [2 * pi] * 6
        rest_poses = [0, -pi / 2, -pi / 2, -pi / 2, -pi / 2, 0]
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
        self.sliders.append(pybullet.addUserDebugParameter("Rx", -pi / 2, pi / 2, 0))
        self.sliders.append(pybullet.addUserDebugParameter("Ry", -pi / 2, pi / 2, pi / 2))
        self.sliders.append(pybullet.addUserDebugParameter("Rz", -pi / 2, pi / 2, 0))

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
        self.target_point = np.array([x_target, y_target, center_z + 0.025])
        self.current_block_center = self.start_point
        block_urdf_path = "/ur_e_description/urdf/block.urdf"
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

    '''gym.Env'''

    def check_end_lock(self):
        lock_state = pybullet.getLinkState(self.ur5, 8, computeForwardKinematics=True)
        plate_position = np.array(lock_state[0], dtype=np.float32)
        plate_z = plate_position[2]  # plate的z坐标
        end_position, end_orientation = self.get_current_end()
        end_z = end_position[2]  # end_effector的z坐标
        end_angle = abs(end_orientation[1] - pi / 2)  # end_effector和z轴夹角
        return True if plate_z < self.center_z < end_z and end_angle < pi / 18 else False

    def check_in_puzzle(self):
        end_position, _ = self.get_current_end()
        pos = [end_position[0] - (self.center_x - self.block_size * self.nx / 2),
               end_position[1] - (self.center_y - self.block_size * self.ny / 2)]  # end相对迷宫的左上角的xy坐标
        return pos, bool(0 < pos[0] < self.block_size * self.nx and 0 < pos[1] < self.block_size * self.ny)

    def get_block_coordinates(self):
        end_position, _ = self.get_current_end()
        end_x_block = int((end_position[0] - (self.center_x - self.block_size * self.nx / 2)) // self.block_size)
        end_y_block = int((end_position[1] - (self.center_y - self.block_size * self.ny / 2)) // self.block_size)
        return end_x_block, end_y_block

    def get_state(self):
        state = self.puzzle_wall.copy()
        end_x, end_y = self.get_block_coordinates()
        if 0 <= end_x < self.nx and 0 <= end_y < self.ny:
            state[end_y][end_x] = 2
        # print(state)
        return state

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
        x_block, y_block = self.get_block_coordinates()
        x_block_moved = int(sin(act * pi / 2)) + x_block
        y_block_moved = int(-cos(act * pi / 2)) + y_block
        # 如果动作会发生碰撞或离开工作区，则不进行动作
        if 0 <= x_block_moved < self.nx and 0 <= y_block_moved < self.ny:
            if self.puzzle_wall[y_block_moved][x_block_moved] == 0:
                y_end_move = -cos(act * pi / 2) * self.block_size
                x_end_move = sin(act * pi / 2) * self.block_size
                new_end_position = [self.current_block_center[0] + x_end_move,
                                    self.current_block_center[1] + y_end_move,
                                    self.current_block_center[2]]
                new_joints_angles = self.calculate_ik(new_end_position, [0, pi / 2, 0])
                self.set_joint_angles(new_joints_angles)
                if self.mode == "test":
                    sleep(0.15)
                else:
                    for _ in range(30):
                        pybullet.stepSimulation()
                self.current_block_center = new_end_position

    def get_reward_and_done(self, reward_mode="shaped"):
        pos, in_puzzle = self.check_in_puzzle()
        in_target_block = bool(self.get_end2target_distance(2, self.target_point) < self.block_size / 2)
        out_timestep = bool(self.step_count >= 200)

        if in_puzzle:
            block_reward, point_reward = self.m.read_reward(pos=pos, block_size=self.block_size)
        else:
            block_reward, point_reward = 0., 0.
        in_stay = bool(block_reward == self.last_block_reward)
        in_progress = bool(block_reward - self.last_block_reward > 0)
        in_regress = bool(block_reward - self.last_block_reward < 0)
        if reward_mode == "sparse":
            reward = 200 if in_target_block else 0  # sparse reward
        elif reward_mode == "shaped":
            reward_success = 1. * in_target_block
            reward_progress = 1. * in_progress - 1. * in_regress - 1. * in_stay
            reward_toolong = -2. * out_timestep
            reward = reward_progress + reward_success + reward_toolong
        else:
            raise AssertionError("请选择奖励模式:sparse/shaped！")

        done = bool(out_timestep or in_target_block)

        self.last_block_reward = 0 if done else block_reward

        return reward, done

    def step(self, act):
        self.robot_action(act)
        self.step_count += 1
        state = self.get_state()
        reward, done = self.get_reward_and_done()
        return state, reward, done, {}

    def reset_robot(self):
        for i in range(20):  # large_scale movement
            middle_joint_angles = self.calculate_ik(self.middle_point, [0, pi / 2, 0])
            self.set_joint_angles(middle_joint_angles)
            for _ in range(30):
                pybullet.stepSimulation()
        for i in range(20):
            reset_joint_angles = self.calculate_ik(self.start_point, [0, pi / 2, 0])
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
        for _ in range(60):
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
    sim = DiscreteActPuzzle(method=pybullet.GUI)
    check_env(sim)
    print("check over!!!")
