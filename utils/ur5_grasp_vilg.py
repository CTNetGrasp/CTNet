import time
import glob
import pybullet as pb
import pybullet_data
import numpy as np
from operator import itemgetter
from scipy.spatial.transform import Rotation as R
import sys


class UR5_Environment:
    def __init__(self):

        self.home_joints = np.array([0, -0.8, 0.5, -0.2, -0.5, 0]) * np.pi 

        self.ik_rest_joints = np.array([0, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi   
        self.drop_joints0 = np.array([0.5, -0.8, 0.5, -0.2, -0.5, 0]) * np.pi   
        self.drop_joints1 = np.array([1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi   
        
        time_step=1 / 240
        gui=True 
        self._client_id = pb.connect(pb.GUI if gui else pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setTimeStep(time_step)


        # Load UR5e
        self.ur5e = pb.loadURDF(
            "./utils/assets/ur5e/ur5e.urdf", basePosition=(0, 0, 0), useFixedBase=True,
        )
        self.ur5e_joints = []
        for i in range(pb.getNumJoints(self.ur5e)):
            info = pb.getJointInfo(self.ur5e, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_name == "ee_fixed_joint":
                self.ur5e_ee_id = joint_id
            if joint_type == pb.JOINT_REVOLUTE:
                self.ur5e_joints.append(joint_id)
        pb.enableJointForceTorqueSensor(self.ur5e, self.ur5e_ee_id, 1)
        
        self.setup_gripper()

        # Move robot to home joint configuration.
        success = self.go_home()
        self.close_gripper()
        self.open_gripper()

        if not success:
            print("Simulation is wrong!")
            exit()

        if gui:
            target = pb.getDebugVisualizerCamera()[11]
            pb.resetDebugVisualizerCamera(
                cameraDistance=1.5, cameraYaw=90, cameraPitch=-25, cameraTargetPosition=target,
            )   

    @property
    def is_static(self):
        """Return true if objects are no longer moving."""
        v = [
            np.linalg.norm(pb.getBaseVelocity(i, physicsClientId=self._client_id)[0])
            for i in self.obj_ids["rigid"]
        ]
        return all(np.array(v) < 5e-3)

    @property
    def info(self):
        """Environment info variable with object poses, dimensions, and colors."""

        info = {}  # object id : (position, rotation, dimensions)
        for obj_ids in self.obj_ids.values():
            for obj_id in obj_ids:
                pos, rot = pb.getBasePositionAndOrientation(
                    obj_id, physicsClientId=self._client_id
                )
                dim = pb.getVisualShapeData(obj_id, physicsClientId=self._client_id)[0][3]
                info[obj_id] = (pos, rot, dim)
        return info

    def obj_info(self, obj_id):
        """Environment info variable with object poses, dimensions, and colors."""

        pos, rot = pb.getBasePositionAndOrientation(
            obj_id, physicsClientId=self._client_id
        )
        dim = pb.getVisualShapeData(obj_id, physicsClientId=self._client_id)[0][3]
        info = (pos, rot, dim)
        return info    


    def wait_static(self, timeout=3):
        """Step simulator asynchronously until objects settle."""
        pb.stepSimulation()
        t0 = time.time()
        while (time.time() - t0) < timeout:
            if self.is_static:
                return True
            pb.stepSimulation()
        print(f"Warning: Wait static exceeded {timeout} second timeout. Skipping.")
        return False


    def setup_gripper(self):
        """Load end-effector: gripper"""
        ee_position, _ = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
        self.ee = pb.loadURDF(
            "./utils/assets/ur5e/gripper/robotiq_2f_85.urdf",
            # "E:\\downloads\\manipulation_kinova_gen3-2ff5f812cc002b606abe705489164dc0920edeea\\manipulation_\\pybullet\\gen3_robotiq_2f_140.urdf",

            ee_position,
            pb.getQuaternionFromEuler((0, -np.pi / 2, 0)),
        )
        self.ee_tip_z_offset = 0.1625
        # self.gripper_angle_open = 0.03
        self.gripper_angle_open = 0.001
        self.gripper_angle_close = 0.95
        self.gripper_angle_close_threshold = 0.73
        self.gripper_mimic_joints = {
            "left_inner_finger_joint": -1,
            "left_inner_knuckle_joint": -1,
            "right_outer_knuckle_joint": -1,
            "right_inner_finger_joint": -1,
            "right_inner_knuckle_joint": -1,
        }
        for i in range(pb.getNumJoints(self.ee)):
            info = pb.getJointInfo(self.ee, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_name == "finger_joint":
                self.gripper_main_joint = joint_id
            elif joint_name == "dummy_center_fixed_joint":
                self.ee_tip_id = joint_id
            elif "finger_pad_joint" in joint_name:
                pb.changeDynamics(
                    self.ee, joint_id, lateralFriction=0.9
                )
                self.ee_finger_pad_id = joint_id
            elif joint_type == pb.JOINT_REVOLUTE:
                self.gripper_mimic_joints[joint_name] = joint_id
                # Keep the joints static
                pb.setJointMotorControl2(
                    self.ee, joint_id, pb.VELOCITY_CONTROL, targetVelocity=0, force=0,
                )
        self.ee_constraint = pb.createConstraint(
            parentBodyUniqueId=self.ur5e,
            parentLinkIndex=self.ur5e_ee_id,
            childBodyUniqueId=self.ee,
            childLinkIndex=-1,
            jointType=pb.JOINT_FIXED,
            jointAxis=(0, 0, 1),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, -0.02),
            childFrameOrientation=pb.getQuaternionFromEuler((0, -np.pi / 2, 0)),
            physicsClientId=self._client_id,
        )
        pb.changeConstraint(self.ee_constraint, maxForce=10000)
        pb.enableJointForceTorqueSensor(self.ee, self.gripper_main_joint, 1)

        # Set up mimic joints in robotiq gripper: left
        c = pb.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["left_inner_finger_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=1, erp=0.8, maxForce=10000)
        c = pb.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["left_inner_knuckle_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=10000)
        # Set up mimic joints in robotiq gripper: right
        c = pb.createConstraint(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            self.ee,
            self.gripper_mimic_joints["right_inner_finger_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=1, erp=0.8, maxForce=10000)
        c = pb.createConstraint(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            self.ee,
            self.gripper_mimic_joints["right_inner_knuckle_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=10000)
        # Set up mimic joints in robotiq gripper: connect left and right
        c = pb.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            physicsClientId=self._client_id,
        )
        pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=1000)

    def step1_grasp(self, pose, grasp_angle, grasp_width): 
        """Execute action with specified primitive.

        Args:
            action: action to execute.

        Returns:
            obs, done
        """
        pose = pose 
        grasp_angle = grasp_angle
        orn = pb.getQuaternionFromEuler([np.pi,0.,grasp_angle+np.pi])   
        pose.extend(list(orn))
        target_pose = pose
        if target_pose is not None:
            success = self.grasp(target_pose, 0.001)
    
        return success
    
    
    def step2_place(self,flag):   
        
        if flag: 
            success = self.move_joints(self.drop_joints1) 
            self.open_gripper(is_slow=True) 
 
        
        return success



    def seed(self, seed=None):
        self._random = np.random.RandomState(seed)
        return seed

    def __del__(self):
        pb.disconnect()

    def get_link_pose(self, body, link):
        result = pb.getLinkState(body, link)
        return result[4], result[5]

    def get_true_object_pose(self, obj_id):
        pos, ort = pb.getBasePositionAndOrientation(obj_id)
        position = np.array(pos).reshape(3, 1)
        rotation = pb.getMatrixFromQuaternion(ort)
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        return transform   

    def get_true_object_poses(self):
        transforms = dict()
        for obj_id in self.obj_ids["rigid"]:
            transform = self.get_true_object_pose(obj_id)
            transforms[obj_id] = transform
        return transforms

    # ---------------------------------------------------------------------------
    # Robot Movement Functions
    # ---------------------------------------------------------------------------

    def go_home(self):
        return self.move_joints(self.home_joints)
    
    def go_rotate_observe(self):
        return self.move_joints(self.rotate_observe_joints)


    def move_joints(self, target_joints, speed=0.01, timeout=3):
        """Move UR5e to target joint configuration."""
        t0 = time.time()
        while (time.time() - t0) < timeout:
            current_joints = np.array(
                [
                    pb.getJointState(self.ur5e, i, physicsClientId=self._client_id)[0]
                    for i in self.ur5e_joints
                ]
            )
            pos, _ = self.get_link_pose(self.ee, self.ee_tip_id)
            if pos[2] < 0.005:
                print(f"Warning: move_joints tip height is {pos[2]}. Skipping.")
                return False
            diff_joints = target_joints - current_joints
            if all(np.abs(diff_joints) < 0.05):
                # give time to stop
                for _ in range(5):
                    pb.stepSimulation()
                return True

            # Move with constant velocity
            norm = np.linalg.norm(diff_joints)
            v = diff_joints / norm if norm > 0 else 0
            step_joints = current_joints + v * speed
            pb.setJointMotorControlArray(
                bodyIndex=self.ur5e,
                jointIndices=self.ur5e_joints,
                controlMode=pb.POSITION_CONTROL,
                targetPositions=step_joints,
                positionGains=np.ones(len(self.ur5e_joints)),
            )
            pb.stepSimulation()
        print(f"Warning: move_joints exceeded {timeout} second timeout. Skipping.")
        return False

    def move_ee_pose(self, pose, speed=0.01):
        """Move UR5e to target end effector pose."""
        target_joints = self.solve_ik(pose)
        return self.move_joints(target_joints, speed)

    def solve_ik(self, pose):
        """Calculate joint configuration with inverse kinematics."""
        joints = pb.calculateInverseKinematics(
            bodyUniqueId=self.ur5e,
            endEffectorLinkIndex=self.ur5e_ee_id,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=[-6.283, -6.283, -3.141, -6.283, -6.283, -6.283],
            upperLimits=[6.283, 6.283, 3.141, 6.283, 6.283, 6.283],
            jointRanges=[12.566, 12.566, 6.282, 12.566, 12.566, 12.566],
            restPoses=np.float32(self.ik_rest_joints).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5,
        )
        joints = np.array(joints, dtype=np.float32)

        return joints

    def straight_move(self, pose0, pose1, rot, speed=0.01, max_force=300, detect_force=False, is_push=False):
        """Move every 1 cm, keep the move in a straight line instead of a curve. Keep level with rot"""
        step_distance = 0.01  # every 1 cm
        vec = np.float32(pose1) - np.float32(pose0)
        length = np.linalg.norm(vec)
        vec = vec / length
        n_push = np.int32(np.floor(length / step_distance))  # every 1 cm
        success = True
        for n in range(n_push):
            target = pose0 + vec * n * step_distance
            success &= self.move_ee_pose((target, rot), speed)
            if detect_force:
                force = np.sum(
                    np.abs(np.array(pb.getJointState(self.ur5e, self.ur5e_ee_id)[2]))
                )
                if force > max_force:
                    target = target - vec * 2 * step_distance
                    self.move_ee_pose((target, rot), speed)
                    print(f"Force is {force}, exceed the max force {max_force}")
                    return False    
        if is_push:
            speed /= 5
        success &= self.move_ee_pose((pose1, rot), speed)
        return success

    def grasp(self, pose, speed=0.005):
        """Execute grasping primitive.

        Args:
            pose: SE(3) grasping pose.

        Returns:
            success: robot movement success if True.
        """
        
        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9, spinningFriction=0.1
        )

        pose = np.array(pose, dtype=np.float32)
        rot = pose[-4:] 
        pos = pose[:3]
        transform = np.eye(4)
        transform[:3, :3] = R.from_quat(rot).as_matrix()
        transform[:3, 3] = pos

        # ee link in tip
        ee_tip_transform = np.array([[0, 0, -1, 0],
                                    [0, 1, 0, 0],
                                    [1, 0, 0, -self.ee_tip_z_offset],
                                    [0, 0, 0, 1]])


        ee_transform = transform @ ee_tip_transform

        pos = (ee_transform[:3, 3]).T

        over = np.array((pos[0]-0.1, pos[1], pos[2] + 0.3))
        rot = R.from_matrix(ee_transform[:3, :3]).as_quat()

    
        
        self.open_gripper()
        success = self.move_joints(self.ik_rest_joints)  

        if success:
            success = self.move_ee_pose((over, rot), speed=speed) 
        if success:
            success = self.straight_move(over, pos, rot, speed, detect_force=True) 
   
            self.close_gripper()
            success = self.straight_move(pos, over, rot, speed)
            success &= self.is_gripper_closed

        

        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9
        )
        return success
        


    def open_gripper(self, is_slow=False):
        self._move_gripper(self.gripper_angle_open, is_slow=is_slow)

    def close_gripper(self, is_slow=True):
        self._move_gripper(self.gripper_angle_close, is_slow=is_slow)

    @property
    def is_gripper_closed(self):
        gripper_angle = pb.getJointState(
            self.ee, self.gripper_main_joint, physicsClientId=self._client_id
        )[0]
        return gripper_angle < self.gripper_angle_close_threshold

    def _move_gripper(self, target_angle, timeout=3, is_slow=False):
        t0 = time.time()
        prev_angle = pb.getJointState(
            self.ee, self.gripper_main_joint, physicsClientId=self._client_id
        )[0]   ### open时，0.01569258  0.54110
        
        if is_slow:
            pb.setJointMotorControl2(
                self.ee,
                self.gripper_main_joint,
                pb.VELOCITY_CONTROL,
                targetVelocity=1 if target_angle > 0.5 else -1,
                maxVelocity=1 if target_angle > 0.5 else -1,
                force=3,
                physicsClientId=self._client_id,
            )


            pb.setJointMotorControl2(
                self.ee,
                self.gripper_mimic_joints["right_outer_knuckle_joint"],
                pb.VELOCITY_CONTROL,
                targetVelocity=1 if target_angle > 0.5 else -1,
                maxVelocity=1 if target_angle > 0.5 else -1,
                force=3,
                physicsClientId=self._client_id,
            )
            for _ in range(10):
                pb.stepSimulation()
            while (time.time() - t0) < timeout:
                current_angle = pb.getJointState(self.ee, self.gripper_main_joint)[0]
                diff_angle = abs(current_angle - prev_angle)
                if diff_angle < 1e-4:
                    break
                prev_angle = current_angle
                for _ in range(10):
                    pb.stepSimulation()
        # maintain the angles
        pb.setJointMotorControl2(
            self.ee,
            self.gripper_main_joint,
            pb.POSITION_CONTROL,
            targetPosition=target_angle,
            force=3.1,
        ) 
        pb.setJointMotorControl2(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            pb.POSITION_CONTROL,
            targetPosition=target_angle,
            force=3.1,
        )

        for _ in range(10):
            pb.stepSimulation()


