import pybullet as p
import pybullet_data
from UR5Sim import UR5Sim
ur = UR5Sim()
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
ballId = p.loadURDF("ur_e_description/urdf/ball_joint.urdf", basePosition=[-0.3, 0, 0])
#createConstraint:(父机器人编号，父机器人的连杆序号，子机器人编号，子机器人连杆编号（-1指base），关节类型，关节转轴，连接的位置，父框架位置，子框架位置)
jointId2 = p.createConstraint(ur.ur5, ur.end_effector_index+1, ballId, -1, p.JOINT_FIXED, [0, 0, 0], [0., -0., -0.], [0, 0, 0],childFrameOrientation=[0,1,0,1])

#移除关节
p.removeConstraint(jointId2)
#生成一个盒子
square = 1  # 边长
boxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[square / 2, square / 2, 0.01])
boxId2 = p.createVisualShape(p.GEOM_BOX, halfExtents=[square / 2, square / 2, 0.01], rgbaColor=[1, 0, 0, 1])
# boxcolorId=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.01])
a, b, height = 1, 1, 1
p.createMultiBody(baseMass=0,
                  basePosition=[a, b, height],
                  baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                  baseCollisionShapeIndex=boxId,
                  baseVisualShapeIndex=boxId2)
