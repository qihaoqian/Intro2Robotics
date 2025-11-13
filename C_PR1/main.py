import time
import numpy as np
import pybullet as p
import pybullet_data
from scipy.optimize import minimize
from PIL import Image, ImageDraw, ImageFont

if __name__ == '__main__': 
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0., 0., -9.81)
    plane = p.loadURDF('plane.urdf')
    robot = p.loadURDF('three-link-robot.urdf', useFixedBase=True)

    num_joints = p.getNumJoints(robot)
    movable_joints = []
    end_effector_link_idx = None
    for idx in range(num_joints): 
        info = p.getJointInfo(robot, idx)
        print('Joint {}: {}'.format(idx, info))

        joint_type = info[2]
        if joint_type != p.JOINT_FIXED: 
            movable_joints.append(idx)

        link_name = info[12].decode('utf-8')
        if link_name == 'end_effector': 
            end_effector_link_idx = idx

    L1 = 0.5 
    L2 = 0.5  
    L3 = 0.2 
    L4 = 0.2  
    
    # Define Kinematics Functions
    def forward_kinematics(joint_angles):
        q1, q2, q3 = joint_angles
        x_ee = L1 * np.cos(q1) + L2 * np.cos(q1 + q2) + L3 * np.cos(q1 + q2 + q3) + L4 * np.cos(q1 + q2 + q3)
        y_ee = L1 * np.sin(q1) + L2 * np.sin(q1 + q2) + L3 * np.sin(q1 + q2 + q3) + L4 * np.sin(q1 + q2 + q3)
        z_ee = 0.0
        theta = q1 + q2 + q3
        return np.array([x_ee, y_ee, z_ee]), theta
    
    def inverse_kinematics_openloop(target_pos, target_theta=0.0, initial_guess=[0.0, 0.0, 0.0]):
        
        def objective(joint_angles):
            current_pos, current_theta = forward_kinematics(joint_angles)
            pos_error = np.linalg.norm(current_pos[:2] - target_pos[:2])
            theta_error = abs(current_theta - target_theta)
            return pos_error + 0.1 * theta_error
        
        result = minimize(objective, initial_guess, method='BFGS', options={'maxiter': 100})
        return result.x
    
    def inverse_kinematics_closedloop(target_pos, initial_guess=[0.0, 0.0, 0.0]):
        def objective(joint_angles):
            q1, q2, q3 = joint_angles
            current_pos, _ = forward_kinematics(joint_angles)
            pos_error = np.linalg.norm(current_pos[:2] - target_pos[:2])
            nullspace_cost = -(abs(q2) + abs(q3))
            return 1000.0 * pos_error + nullspace_cost
        
        result = minimize(objective, initial_guess, method='BFGS', options={'maxiter': 100})
        return result.x
    
    def smooth_trajectory(waypoints, total_time, current_time):
        n_segments = len(waypoints) - 1
        segment_time = total_time / n_segments
        segment_idx = int(current_time / segment_time)
        if segment_idx >= n_segments:
            segment_idx = n_segments - 1
        t_in_segment = current_time - segment_idx * segment_time
        t_normalized = t_in_segment / segment_time
        start = np.array(waypoints[segment_idx])
        end = np.array(waypoints[(segment_idx + 1) % len(waypoints)])
        s = 10 * t_normalized**3 - 15 * t_normalized**4 + 6 * t_normalized**5
        position = start + s * (end - start)
        return position
    
    print("---------------Q3---------------")
    
    configurations_q3 = [
        [0, 0, np.pi/2],
        [0, np.pi, 0],
        [np.pi/2, np.pi/2, 0],
        [np.pi/3, np.pi/2, 0]
    ]
    
    p.resetDebugVisualizerCamera(
        cameraDistance=2.5,
        cameraYaw=45,
        cameraPitch=-20,
        cameraTargetPosition=[0.5, 0, 0.2]
    )
    
    for i, config in enumerate(configurations_q3):
        pos, theta = forward_kinematics(config)
        
        for j, joint_idx in enumerate(movable_joints):
            p.resetJointState(robot, joint_idx, config[j])
        
        for _ in range(30):
            p.stepSimulation()
            time.sleep(1./240.)
        
        width_img, height_img = 1920, 1080
        view_matrix_img = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.5, 0, 0.2],
            distance=2.5,
            yaw=45,
            pitch=-20,          
            roll=0,
            upAxisIndex=2
        )
        proj_matrix_img = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width_img/height_img,
            nearVal=0.1,
            farVal=100.0
        )
        img_arr = p.getCameraImage(width_img, height_img, view_matrix_img, proj_matrix_img, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        
        rgb_array = np.array(img_arr[2], dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (height_img, width_img, 4))[:, :, :3]
        img = Image.fromarray(rgb_array)
        filename = f"q3_configuration_{i+1}.png"
        img.save(filename)
        print(f"  ✓ Q3 png saved\n")
        time.sleep(0.5)
    
    print("--------------- Q3 Complete ---------------\n")
    
    print("---------------Q5---------------")
    
    waypoints_q5 = [
        [1.2, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.4, 0.0, 0.0],
        [1.2, 0.0, 0.0]
    ]
    
    total_time_q5 = 10.0
    fps = 30
    num_frames_q5 = int(total_time_q5 * fps)
    dt_q5 = total_time_q5 / num_frames_q5
    
    # Setup camera
    width, height = 800, 600
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.6, 0, 0.2],
        distance=2.5,
        yaw=45,
        pitch=-25,
        roll=0,
        upAxisIndex=2
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=width/height,
        nearVal=0.1,
        farVal=100.0
    )
    
    for i, joint_idx in enumerate(movable_joints):
        p.resetJointState(robot, joint_idx, 0.0)
    
    frames_q5 = []
    print("\nGenerating Q5 gif.")
    
    for frame in range(num_frames_q5):
        t = frame * dt_q5
        target_state = smooth_trajectory(waypoints_q5, total_time_q5, t)
        target_pos = target_state[:2]
        target_theta = target_state[2]
        
        joint_states = p.getJointStates(robot, movable_joints)
        current_angles = [state[0] for state in joint_states]
        
        new_angles = inverse_kinematics_openloop(
            np.array([target_pos[0], target_pos[1], 0.0]), 
            target_theta, 
            current_angles
        )
        
        for i, joint_idx in enumerate(movable_joints):
            p.resetJointState(robot, joint_idx, new_angles[i])
        
        p.stepSimulation()
        
        img_arr = p.getCameraImage(width, height, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(img_arr[2], dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (height, width, 4))[:, :, :3]
        img = Image.fromarray(rgb_array)
        
        draw = ImageDraw.Draw(img)
        text = f"Q5 Open-loop | Time: {t:.2f}s | Target: ({target_pos[0]:.2f}, {target_pos[1]:.2f})"
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except:
            font = ImageFont.load_default()
        draw.text((10, 10), text, fill=(255, 255, 255), font=font)
        
        frames_q5.append(img)
        
        if (frame + 1) % 30 == 0:
            print(f"  Q5: {frame + 1}/{num_frames_q5} frames ({(frame+1)/num_frames_q5*100:.1f}%)")
    
    print("--------------- Q5 complete ---------------")
    frames_q5[0].save(
        'q5_openloop_animation.gif',
        save_all=True,
        append_images=frames_q5[1:],
        duration=int(1000/fps),
        loop=0
    )
    print(f"✓ Q5 gif saved\n")
    
    # Q6
    print("---------------Q6---------------")
    
    waypoints_q6 = [
        [1.2, 0.0],
        [0.5, 0.5],
        [0.5, -0.5],
        [0.4, 0.0],
        [1.2, 0.0]
    ]
    
    total_time_q6 = 10.0
    num_frames_q6 = int(total_time_q6 * fps)
    dt_q6 = total_time_q6 / num_frames_q6
    
    for i, joint_idx in enumerate(movable_joints):
        p.resetJointState(robot, joint_idx, 0.0)
    
    waypoints_with_theta = [[wp[0], wp[1], 0.0] for wp in waypoints_q6]
    frames_q6 = []
    joint_angle_sums = []
    
    
    for frame in range(num_frames_q6):
        t = frame * dt_q6
        target_state = smooth_trajectory(waypoints_with_theta, total_time_q6, t)
        target_pos_xy = target_state[:2]
        
        joint_states = p.getJointStates(robot, movable_joints)
        current_angles = [state[0] for state in joint_states]
        
        new_angles = inverse_kinematics_closedloop(target_pos_xy, current_angles)
        
        for i, joint_idx in enumerate(movable_joints):
            p.resetJointState(robot, joint_idx, new_angles[i])
        
        q1, q2, q3 = new_angles
        joint_angle_sum = abs(q2) + abs(q3)
        joint_angle_sums.append(joint_angle_sum)
        
        p.stepSimulation()
        
        img_arr = p.getCameraImage(width, height, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(img_arr[2], dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (height, width, 4))[:, :, :3]
        img = Image.fromarray(rgb_array)
        
        draw = ImageDraw.Draw(img)
        text1 = f"Q6 Closed-loop (Nullspace) | Time: {t:.2f}s"
        text2 = f"Target: ({target_pos_xy[0]:.2f}, {target_pos_xy[1]:.2f}) | |θ2|+|θ3|={joint_angle_sum:.2f}rad"
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)

        draw.text((10, 10), text1, fill=(255, 255, 255), font=font)
        draw.text((10, 35), text2, fill=(255, 255, 0), font=font)
        
        frames_q6.append(img)
        
        if (frame + 1) % 30 == 0:
            print(f"  Q6: {frame + 1}/{num_frames_q6} frames ({(frame+1)/num_frames_q6*100:.1f}%)")
    
    frames_q6[0].save(
        'q6_closedloop_animation.gif',
        save_all=True,
        append_images=frames_q6[1:],
        duration=int(1000/fps),
        loop=0
    )
    print(f"✓ Q6 gif saved\n")
    print(f"--------------- Q6 complete ---------------")
    
    for _ in range(1200): 
        p.stepSimulation()
        time.sleep(1./240.)

    p.disconnect()
