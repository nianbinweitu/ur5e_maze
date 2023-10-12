from ur5e_maze_DiscreteActPuzzleEnv import DiscreteActPuzzle
from ur5e_maze_ImageObsEnv import ImageObsUR5Sim
from time import sleep
import pybullet
import serial
import serial.tools.list_ports


class usartread():
    def __init__(self):
        self.usart_port = serial.Serial()
        self.usart_port.port = "/dev/ttyACM0"
        self.usart_port.baudrate = 115200
        self.usart_port.bytesize = 8
        self.usart_port.parity = serial.PARITY_NONE
        self.usart_port.stopbits = 1
        # if not self.usart_port.is_open:
        #     self.usart_port.open()
        # self.usart_port.read_until(expected=b'\r\n')

    def read_usart_data(self):
        data = self.usart_port.read_until(expected=b'\r\n')  # tepe(data) = <class 'bytes'>
        data = data.decode("ascii")
        data = data.split(',')
        if data[0] == '88' and len(data) == 3:
            data = data[1:]
            for i in range(len(data)):
                try:
                    data[i] = float(data[i])
                except ValueError:
                    return self.read_usart_data()
                except IndexError:
                    return self.read_usart_data()
            return data
        else:
            return self.read_usart_data()


def stick_control_regular_env():
    '''imu串口初始化'''
    signal = usartread()
    '''环境初始化'''
    env = ImageObsUR5Sim(method=pybullet.GUI)
    env.reset()
    done = False
    episode_reward = 0
    for i in range(5):
        if not signal.usart_port.is_open:
            signal.usart_port.open()
        signal.usart_port.read_until(expected=b'\r\n')
        while not done:
            data = signal.read_usart_data()
            signal_x, signal_y = data[0], data[1]
            x_move = 0 if 40 < signal_x < 60 else (signal_x-50)/50/100   # [0.001, 0.01]
            y_move = 0 if 40 < signal_y < 60 else (signal_y-50)/50/100
            env.robot_action([x_move, y_move])
            reward, done = env.get_reward_and_done()
            env.step_count += 1
            print(reward, env.step_count)
            episode_reward += reward
        signal.usart_port.close()
        env.reset()
        done = False
        print("episode_reward=", episode_reward)
        episode_reward = 0
        sleep(2)


def stick_control_discrete_act_env():
    '''imu串口初始化'''
    signal = usartread()
    '''环境初始化'''
    env = DiscreteActPuzzle(method=pybullet.GUI, mode="test")  #

    for i in range(5):
        env.reset()
        done = False
        episode_reward = 0

        if not signal.usart_port.is_open:
            signal.usart_port.open()
        signal.usart_port.read_until(expected=b'\r\n')

        # 过滤掉开始的杂信号
        data = signal.read_usart_data()
        signal_x, signal_y = data[0], data[1]
        while not (20 < signal_x < 80 and 20 < signal_y < 80):
            data = signal.read_usart_data()
            signal_x, signal_y = data[0], data[1]
        print("------------------------------")
        while not done:
            if not signal.usart_port.is_open:
                signal.usart_port.open()
            signal.usart_port.read_until(expected=b'\r\n')
            data = signal.read_usart_data()
            signal_x, signal_y = data[0], data[1]
            x_move = 0 if 20 < signal_x < 80 else (signal_x-50)/50/100   # [0.001, 0.01]
            y_move = 0 if 20 < signal_y < 80 else (signal_y-50)/50/100
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
                state, reward, done, _ = env.step(act)
                print(reward, env.step_count)
                episode_reward += reward
            signal.usart_port.close()
            sleep(0.1)
        print("episode_reward=", episode_reward)
        sleep(1)


if __name__ == "__main__":
    stick_control_discrete_act_env()
    # stick_control_regular_env()




    
    
    
    
