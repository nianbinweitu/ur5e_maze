from ur5e_maze_FlattenMatrixObsEnv import FlattenMatrixObsUR5Sim
from ur5e_maze_DiscreteActPuzzleEnv import DiscreteActPuzzle
import pybullet
import pickle
import datetime
import numpy as np
from time import sleep
from imitation.data.types import TrajectoryWithRew
from imitation.data import rollout
from ur5e_maze_StickControl import usartread
import os

now = datetime.datetime.now().strftime("%y-%m-%d %H:%M")


'''
连续动作环境：FlattenMatrixObsEnv/MatrixObsEnv/ImageObsEnv
'''


def traj_connect1(n_rollouts):
    '''imu串口初始化'''
    signal = usartread()
    '''环境初始化'''
    env = FlattenMatrixObsUR5Sim(method=pybullet.GUI)
    # if env.method != pybullet.GUI:
    #     raise AssertionError("请将env.method设置为pybullet.GUI")
    # if not isinstance(env, FlattenMatrixPuzzleObsUR5Sim):
    #     raise AssertionError("请选择连续动作环境：FlattenMatrixObsEnv")
    rollouts_path = 'ur5e_puzzle_rollouts/FlattenMatrixObsEnv/rollouts{} {}.pkl'.format(n_rollouts, now)

    rollouts = []
    epi_step = []
    epi_rew = []

    while len(rollouts) < n_rollouts:
        reset_state = env.reset()
        done = False
        episode_reward = 0

        if not signal.usart_port.is_open:
            signal.usart_port.open()
        signal.usart_port.read_until(expected=b'\r\n')

        # 过滤掉开始的杂信号
        data = signal.read_usart_data()
        signal_x, signal_y = data[0], data[1]
        while not (45 < signal_x < 55 and 45 < signal_y < 55):
            data = signal.read_usart_data()
            signal_x, signal_y = data[0], data[1]
        print("-------------------------------------------------------------------------------------")

        s = reset_state
        a = np.array([])
        r = np.array([])
        action = np.array([])
        state = np.array([])

        while not done:
            data = signal.read_usart_data()
            signal_x, signal_y = data[0], data[1]
            x_move = 0 if 47 < signal_x < 53 else (signal_x - 50) / 50 * 0.02  # [0.004, 0.02]
            y_move = 0 if 47 < signal_y < 53 else (signal_y - 50) / 50 * 0.02
            action = np.array(([x_move, y_move]), dtype=np.float32)
            if not np.all(action == 0.):
                env.robot_action(action)
                state = env.get_state()
                reward, done = env.get_reward_and_done()
                env.step_count += 1
                a = np.append(a, action)
                r = np.append(r, reward)
                s = np.append(s, state)
                episode_reward += reward
        signal.usart_port.close()
        n = r.size
        if episode_reward > 1.95:
            a = a.reshape(n, action.size)
            s = s.reshape(n + 1, state.size)
            traj = TrajectoryWithRew(obs=s, acts=a, rews=r, infos=None, terminal=True)
            rollouts.append(traj)
            epi_step.append(n)
            epi_rew.append(episode_reward)
        print(n, "steps         ", "episode_reward =", episode_reward, "         ", n_rollouts - len(rollouts), "episodes left")
        sleep(2)

    # 存储rollouts
    # print(rollouts)
    with open(rollouts_path, 'wb') as f:
        pickle.dump(rollouts, f)
    print("rollouts saved!")
    return rollouts


# 读取rollouts
def read_rollouts(path):
    with open(path, 'rb') as f:
        rollouts = pickle.load(f)
    print(rollouts)
    transitions = rollout.flatten_trajectories(rollouts)
    return transitions


def combine_rollouts():
    folder_path = '../ur5e_puzzle_rollouts/DiscreteActEnv'
    file_names = os.listdir(folder_path)
    all_rollouts = []
    for file in file_names:
        path = os.path.join(folder_path, file)
        with open(path, 'rb') as f:
            rollouts = pickle.load(f)
        for traj in rollouts:
            all_rollouts.append(traj)
        print(len(all_rollouts))
    len_r = len(all_rollouts)

    # 存储rollouts
    rollouts_store_path = os.path.join(folder_path, 'combined_rollouts{}.pkl'.format(len_r))
    with open(rollouts_store_path, 'wb') as f:
        pickle.dump(all_rollouts, f)
    print("combined_rollouts have been saved!")


'''
离散动作环境：DiscreteActPuzzle

'''


def traj_connect2(n_rollouts):
    '''imu串口初始化'''
    signal = usartread()
    '''环境初始化'''
    env = DiscreteActPuzzle(method=pybullet.GUI, mode="test")
    # if env.method != pybullet.GUI or env.mode != "test":
    #     raise AssertionError("请将env.method设置为pybullet.GUI，env.mode设置为test")
    # if not isinstance(env, DiscreteActPuzzle):
    #     raise AssertionError("请选择离散动作环境：DiscreteActPuzzle")
    rollouts_path = 'ur5e_puzzle_rollouts/DiscreteActPuzzle/rollouts{} {}.pkl'.format(n_rollouts, now)

    rollouts = []
    epi_rew = []

    while len(rollouts) < n_rollouts:
        reset_state = env.reset()
        done = False
        episode_reward = 0

        if not signal.usart_port.is_open:
            signal.usart_port.open()
        signal.usart_port.read_until(expected=b'\r\n')

        # 过滤掉开始的杂信号
        data = signal.read_usart_data()
        signal_x, signal_y = data[0], data[1]
        while not (47 < signal_x < 53 and 45 < signal_y < 55):
            data = signal.read_usart_data()
            signal_x, signal_y = data[0], data[1]
        print("-------------------------------------------------------------------------------------")

        s = reset_state
        a = np.array([])
        r = np.array([])
        action = np.array([])
        state = np.array([])

        while not done:
            if not signal.usart_port.is_open:
                signal.usart_port.open()
            signal.usart_port.read_until(expected=b'\r\n')
            data = signal.read_usart_data()
            signal_x, signal_y = data[0], data[1]
            x_move = 0 if 30 < signal_x < 70 else (signal_x-50)/50/100   # [0.001, 0.01]
            y_move = 0 if 30 < signal_y < 70 else (signal_y-50)/50/100
            act = -1
            if abs(x_move) >= abs(y_move):
                if x_move < 0:
                    act = 3
                elif x_move > 0:
                    act = 1
            elif abs(y_move) > abs(x_move):
                if y_move < 0:
                    act = 0
                elif y_move > 0:
                    act = 2
            if act == -1:
                pass
            else:
                env.robot_action(act)
                state = env.get_state()
                env.step_count += 1
                reward, done = env.get_reward_and_done()

                a = np.append(a, action)
                r = np.append(r, reward)
                s = np.append(s, state)

                print(reward, env.step_count)
                episode_reward += reward

            signal.usart_port.close()
            sleep(0.1)
        n = r.size
        if episode_reward >= 1.9:
            a = a.reshape(n, action.size)
            s = s.reshape(n + 1, state.size)
            traj = TrajectoryWithRew(obs=s, acts=a, rews=r, infos=None, terminal=True)
            rollouts.append(traj)
            epi_rew.append(episode_reward)
        print(n, "steps     ", "episode_reward =", episode_reward, "     ", n_rollouts - len(rollouts), "episodes left")
        sleep(2)

    # 存储rollouts
    with open(rollouts_path, 'wb') as f:
        pickle.dump(rollouts, f)
    print("rollouts saved!")
    return rollouts


if __name__ == "__main__":
    rollouts = traj_connect1(100)

