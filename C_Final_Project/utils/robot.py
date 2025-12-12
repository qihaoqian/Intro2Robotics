"""
Robot class - Eye-in-hand robot configuration
"""

import numpy as np
import pybullet as p


class EyeInHandRobot:
    """Eye-in-hand robot class - Camera mounted on robot end-effector"""

    def __init__(self, robot_id, initial_joint_pos, camera_offset=0.1):
        """
        Initialize robot

        Args:
            robot_id: Robot ID in PyBullet
            initial_joint_pos: List of initial joint positions
            camera_offset: Camera offset in z-direction (to avoid gripper)
        """
        self.robot_id = robot_id
        self.eeFrameId = []
        self.camera_offset = camera_offset

        # Get joint information
        self._numLinkJoints = p.getNumJoints(self.robot_id)  # Includes passive joints
        jointInfo = [p.getJointInfo(self.robot_id, i) for i in range(self._numLinkJoints)]

        # Get joint positions (some joints are passive)
        self._active_joint_indices = []
        for i in range(self._numLinkJoints):
            if jointInfo[i][2] == p.JOINT_REVOLUTE:
                self._active_joint_indices.append(jointInfo[i][0])
        self.numActiveJoints = len(self._active_joint_indices)  # Exact number of active joints

        # Reset joints
        for i in range(self._numLinkJoints):
            p.resetJointState(self.robot_id, i, initial_joint_pos[i])

    def get_ee_position(self):
        """
        Get end-effector position and orientation

        Returns:
            endEffectorPos: End-effector position (3D vector)
            endEffectorOrn: End-effector orientation (3x3 rotation matrix)
        """
        endEffectorIndex = self.numActiveJoints
        endEffectorState = p.getLinkState(self.robot_id, endEffectorIndex)
        endEffectorPos = np.array(endEffectorState[0])
        endEffectorOrn = np.array(p.getMatrixFromQuaternion(endEffectorState[1])).reshape(3, 3)

        # Add offset to clear the gripper
        endEffectorPos += self.camera_offset * endEffectorOrn[:, 2]
        return endEffectorPos, endEffectorOrn

    def get_current_joint_angles(self):
        """
        Get current joint angles

        Returns:
            joint_angles: Array of joint angles
        """
        joint_angles = np.zeros(self.numActiveJoints)
        for i in range(self.numActiveJoints):
            joint_state = p.getJointState(self.robot_id, self._active_joint_indices[i])
            joint_angles[i] = joint_state[0]
        return joint_angles

    def get_jacobian_at_current_position(self):
        """
        Get robot Jacobian at current position

        Returns:
            Jacobian: 6xN Jacobian matrix (N is number of active joints)
        """
        mpos, mvel, mtorq = self.get_active_joint_states()
        zero_vec = [0.0] * len(mpos)
        linearJacobian, angularJacobian = p.calculateJacobian(
            self.robot_id, self.numActiveJoints, [0, 0, self.camera_offset], mpos, zero_vec, zero_vec
        )
        # Only return Jacobian for active joints
        Jacobian = np.vstack((linearJacobian, angularJacobian))
        return Jacobian[:, : self.numActiveJoints]

    def set_joint_position(self, desired_joint_positions, kp=1.0, kv=0.3):
        """
        Set robot joint angle positions

        Args:
            desired_joint_positions: Desired joint positions
            kp: Position gain
            kv: Velocity gain
        """
        zero_vec = [0.0] * self._numLinkJoints
        allJointPositionObjectives = [0.0] * self._numLinkJoints
        for i in range(desired_joint_positions.shape[0]):
            idx = self._active_joint_indices[i]
            allJointPositionObjectives[idx] = desired_joint_positions[i]

        p.setJointMotorControlArray(
            self.robot_id,
            range(self._numLinkJoints),
            p.POSITION_CONTROL,
            targetPositions=allJointPositionObjectives,
            targetVelocities=zero_vec,
            positionGains=[kp] * self._numLinkJoints,
            velocityGains=[kv] * self._numLinkJoints,
        )

    def get_active_joint_states(self):
        """
        Get active joint states of robot (positions, velocities, and torques)

        Returns:
            joint_positions: List of joint positions
            joint_velocities: List of joint velocities
            joint_torques: List of joint torques
        """
        joint_states = p.getJointStates(self.robot_id, range(self._numLinkJoints))
        joint_infos = [p.getJointInfo(self.robot_id, i) for i in range(self._numLinkJoints)]
        joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques
