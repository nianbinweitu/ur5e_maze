import math

import numpy as np
from queue import Queue


class Maze:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.cul = 0
        self.que = Queue()
        # self.moveque=Queue()
        self.single_reward = 0
        # 随机确定入口
        self.point = np.array([0, np.random.randint(1, rows - 2)])
        # 每次只能上下左右试探
        self.step_set = np.array([[1, 0],
                                  [-1, 0],
                                  [0, 1],
                                  [0, -1]])
        maze = np.ones(shape=(cols, rows))
        maze_move = np.zeros(shape=(2, cols, rows), dtype=np.int8)
        self.maze = maze
        self.maze_move = maze_move
        maze[self.point[0]][self.point[1]] = 0
        self.maze, _ = self.find_next_step(self.maze, self.point)
        # maze:带分数的矩阵  maze_shape：形状矩阵
        self.maze, self.maze_shape, self.maze_move = self.get_maze()

    def find_next_step(self, maze, point):
        # 用递归实现深度优先搜索
        done = False
        step_set = np.random.permutation(self.step_set)
        for next_step in step_set:
            next_point = point + next_step * 2
            x, y = next_point
            if 0 < x < self.cols - 1 and 1 <= y < self.rows - 1:  # 在格子内

                if maze[x][y] == 1:  # 如果还没打通，就打通
                    maze[point[0]][point[1]] = 2
                    maze[(point + next_step)[0]][(point + next_step)[1]] = 2

                    maze, done = self.find_next_step(maze, next_point)  # 深度优先搜索
                    if done:
                        maze[point[0]][point[1]] = 0
                        maze[(point + next_step)[0]][(point + next_step)[1]] = 0
                        self.que.put([(point + next_step)[0], (point + next_step)[1]])
                        self.que.put([point[0], point[1]])
                        # self.moveque.put(next_step)
                        self.cul += 1
                        return maze, done
            elif x >= self.cols - 1 and 1 <= y < self.rows:
                maze[point[0]][point[1]] = 0
                maze[(point + next_step)[0]][(point + next_step)[1]] = 0

                if x < self.cols:
                    maze[next_point[0]][next_point[1]] = 0
                    self.que.put([next_point[0], next_point[1]])
                self.que.put([(point + next_step)[0], (point + next_step)[1]])
                self.que.put([point[0], point[1]])
                # self.moveque.put(next_step)
                done = True
                return maze, done

        # 全部遍历后，还是找不到，就是这个叶节点没有下一步了，返回即可
        return maze, done

    def get_maze(self, reward=1.0):
        maze = self.maze
        tmp = (self.cul * 2.0 + 2.0)  # self.cul通道block数
        single_reward = reward / tmp
        self.single_reward = single_reward

        for i in range(self.cols):
            for j in range(self.rows):
                if maze[i][j] >= 1:
                    maze[i][j] = -1
        maze_shape = np.negative(maze).astype(np.int8)

        maze_move = self.maze_move
        next_x, next_y = None, None
        for i in range(self.que.qsize()):
            x, y = self.que.get()
            maze[x][y] = single_reward * (tmp - i - 1)
            if x == self.cols - 1:
                maze_move[1, x, y] = 2
            else:
                if next_y - y == 0:
                    if next_x - x == 1:
                        maze_move[0, next_x, next_y] = 1
                        maze_move[1, x, y] = 2
                    if next_x - x == -1:
                        maze_move[0, next_x, next_y] = 2
                        maze_move[1, x, y] = 1
                elif next_x - x == 0:
                    if next_y - y == 1:
                        maze_move[0, next_x, next_y] = 3
                        maze_move[1, x, y] = 4
                    if next_y - y == -1:
                        maze_move[0, next_x, next_y] = 4
                        maze_move[1, x, y] = 3
                else:
                    raise AssertionError("WRONG MAZE!")
                if x == 0:
                    maze_move[0, x, y] = 1
            next_x, next_y = x, y

        self.step_set = np.array([[1, 0],
                                  [-1, 0],
                                  [0, 1],
                                  [0, -1]])
        return maze, maze_shape, maze_move

    def read_reward(self, pos, block_size):
        """
        :param pos: 输入一个末端的位置，这个位置需要处理：假如你生成的迷宫的左上角在空间的位置是（x,y）需要你减去这个项
        :param block_size: 方块的大小，默认是正方形
        :return: 返回奖励
        """
        y, x = pos[0], pos[1]
        block_x, block_y = int(x // block_size), int(y // block_size)
        in_block_x, in_block_y = float(x % block_size), float(y % block_size)
        rew = self.maze[block_x][block_y]
        dr = self.single_reward
        move_in = self.maze_move[0, block_x, block_y]
        move_out = self.maze_move[1, block_x, block_y]
        if [move_in, move_out] == [0, 0]:
            rew_ = rew
        elif [move_in, move_out] == [1, 2]:
            rew_ = rew + dr * in_block_x / block_size
        elif [move_in, move_out] == [2, 1]:
            rew_ = rew + dr * (1 - in_block_x / block_size)
        elif [move_in, move_out] == [3, 4]:
            rew_ = rew + dr * in_block_y / block_size
        elif [move_in, move_out] == [4, 3]:
            rew_ = rew + dr * (1 - in_block_y / block_size)
        elif [move_in, move_out] == [1, 3]:
            rew_ = rew + dr * math.atan(in_block_x / in_block_y) * 2 / math.pi
        elif [move_in, move_out] == [1, 4]:
            rew_ = rew + dr * math.atan(in_block_x / (block_size - in_block_y)) * 2 / math.pi
        elif [move_in, move_out] == [2, 3]:
            rew_ = rew + dr * math.atan((block_size - in_block_x) / in_block_y) * 2 / math.pi
        elif [move_in, move_out] == [2, 4]:
            rew_ = rew + dr * math.atan((block_size - in_block_x) / (block_size - in_block_y)) * 2 / math.pi
        elif [move_in, move_out] == [3, 1]:
            rew_ = rew + dr * math.atan(in_block_y / in_block_x) * 2 / math.pi
        elif [move_in, move_out] == [3, 2]:
            rew_ = rew + dr * math.atan(in_block_y / (block_size - in_block_x)) * 2 / math.pi
        elif [move_in, move_out] == [4, 1]:
            rew_ = rew + dr * math.atan((block_size - in_block_y) / in_block_x) * 2 / math.pi
        elif [move_in, move_out] == [4, 2]:
            rew_ = rew + dr * math.atan((block_size - in_block_y) / (block_size - in_block_x)) * 2 / math.pi
        else:
            raise AssertionError("maze_move生成的有问题！move_in/move_out为{}/{}".format(move_in, move_out))
        return rew, rew_


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    while 1:
        maze_generator = Maze(12, 10)
        mymaze, mymaze_shape, mymaze_move = maze_generator.get_maze()
        print(mymaze)
        print(maze_generator.read_reward([0.3, 0.2], 0.05))
        plt.imshow(mymaze)
        plt.show()
