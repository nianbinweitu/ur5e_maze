from gym.spaces import Box, Dict
import numpy as np
import pybullet
from ur5e_maze_ImageObsEnv import ImageObsUR5Sim
from stable_baselines3.common.env_checker import check_env


class MatrixPuzzleObsUR5Sim(ImageObsUR5Sim):
    def __init__(self, method=pybullet.DIRECT, mode="train"):
        super().__init__(method, mode)

        # 观测:puzzle_block矩阵 + end的xy坐标
        end_xy_low = np.array([0., -1.], dtype=np.float32)
        end_xy_high = np.array([1., 1.], dtype=np.float32)
        self.observation_space = Dict({
            'puzzle_block': Box(low=0., high=2.,
                                shape=(self.ny, self.nx),
                                dtype=np.float32),
            'end_position': Box(end_xy_low, end_xy_high)
        })
        # print(self.observation_space.sample())

    def get_state(self):
        end_position, _ = self.get_current_end()
        end_position_xy = end_position[:2]
        puzzle_block = np.array(self.puzzle_wall, dtype=np.float32)
        return dict({'puzzle_block': puzzle_block,
                     'end_position': end_position_xy})


if __name__ == "__main__":
    sim = MatrixPuzzleObsUR5Sim(method=pybullet.GUI)
    check_env(sim)
    print("check over!!!")
