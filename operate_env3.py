from ur5e_maze_FlattenMatrixObsEnv import FlattenMatrixObsUR5Sim
from operate_env import OperateRobot
import pybullet


if __name__ == "__main__":
    '''导入环境'''
    # ur = FlattenMatrixObsUR5Sim()
    ur = FlattenMatrixObsUR5Sim(method=pybullet.GUI)
    '''测试环境'''
    # check_env(ur)
    # print("check over !!!")

    '''操作环境'''
    trainer = OperateRobot(ur)

    '''rl train'''
    # trainer.rl_train("sac")

    # trainer.venv_rl_train("sac")
    '''rl retrain'''
    # trainer.rl_retrain(model_path='ur5e_puzzle_result/rl/MatrixObsEnv/23-06-22/15:04/eventual_model.zip',
    #                    store_path='ur5e_puzzle_result/rl/MatrixObsEnv/23-06-22/15:04',
    #                    rl_algorithm="ppo")

    '''test rl model'''
    # test_model_path = 'ur5e_puzzle_result/rl/FlattenMatrixObsEnv/23-07-04/11:16:16/eval/best_model.zip'
    # trainer.test_rl_model("sac", test_model_path)

    '''il'''
    # transitions = trainer.rollouts_to_transitions(
    #     'ur5e_puzzle_rollouts/FlattenMatrixObsEnv/rollouts100 23-06-26 22:06.pkl')
    # trainer.bc_pretrain(transitions)

    '''rl after il'''
    # trainer.rl_train_after_pretrain()

