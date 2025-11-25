import pybullet as p
import pybullet_data
import time
from useful_code import *
import random


def check_node_collision(robot_id, object_ids, joint_position):
    """
    Checks for collisions between a robot and an object in PyBullet.

    Args:
        robot_id (int): The ID of the robot in PyBullet.
        object_id (int): The ID of the object in PyBullet.
        joint_position (list): List of joint positions.

    Returns:
        bool: True if a collision is detected, False otherwise.
    """
    # set joint positions
    for joint_index, joint_pos in enumerate(joint_position):
        p.resetJointState(robot_id, joint_index, joint_pos)

    # Perform collision check for all links
    for object_id in object_ids:    # Check for each object
        for link_index in range(0, p.getNumJoints(robot_id)): # Check for each link of the robot
            contact_points = p.getClosestPoints(
                bodyA=robot_id, bodyB=object_id, distance=0.01, linkIndexA=link_index
            )
            if contact_points:  # If any contact points exist, a collision is detected
                return True # exit early
    return False

#################################################
#### YOUR CODE HERE: COLLISION EDGE CHECKING ####
#################################################
def check_edge_collision(robot_id, object_ids, joint_position_start, joint_position_end, discretization_step=0.01):
    """
    Checks for collision between two joint positions of a robot in PyBullet.
    Args:
        robot_id (int): The ID of the robot in PyBullet.
        object_ids (list): List of IDs of the objects in PyBullet.
        joint_position_start (list): List of joint positions to start from.
        joint_position_end (list): List of joint positions to get to.
        discretization_step (float): maximum interpolation distance before a new collision check is performed.
    Returns:
        bool: True if a collision is detected, False otherwise.
    """
    start_pos = np.array(joint_position_start)
    end_pos = np.array(joint_position_end)

    # Calculate Euclidean distance between start and end joint positions
    distance = np.linalg.norm(end_pos - start_pos)

    # Determine number of steps based on discretization_step
    if distance < 1e-6:
        num_steps = 1
    else:
        num_steps = int(np.ceil(distance / discretization_step))

    # Check collision for each intermediate point
    for i in range(num_steps + 1):
        t = i / num_steps
        intermediate_joint_position = start_pos + t * (end_pos - start_pos)
        if check_node_collision(robot_id, object_ids, intermediate_joint_position):
            return True

    return False


# Provided
class Node:
    def __init__(self, joint_angles):
        self.joint_angles = np.array(joint_angles)  # joint angles of the node in n-dimensional space
        self.parent = None
        self.cost = 0.0


######################################################################
##################### YOUR CODE HERE: RRT CLASS ######################
######################################################################
class RRT:
    def __init__(self, q_start, q_goal, robot_id, obstacle_ids, q_limits, max_iter=10000, step_size=0.5):
        """
        RRT Initialization.

        Parameters:
        - q_start: List of starting joint angles [x1, x2, ..., xn].
        - q_goal: List of goal joint angles [x1, x2, ..., xn].
        - obstacle_ids: List of obstacles, each as a tuple ([center1, center2, ..., centern], radius).
        - q_limits: List of tuples [(min_x1, max_x1), ..., (min_xn, max_xn)] representing the limits in each dimension.
        - max_iter: Maximum number of iterations.
        - step_size: Maximum step size to expand the tree.
        """
        self.q_start = Node(q_start)
        self.q_goal = Node(q_goal)
        self.obstacle_ids = obstacle_ids
        self.robot_id = robot_id
        self.q_limits = q_limits
        self.max_iter = max_iter
        self.step_size = step_size
        self.node_list = [self.q_start]

    def step(self, from_node, to_joint_angles):
        """Step from "from_node" to "to_joint_angles", that should
         (a) return the to_joint_angles if it is within the self.step_size or
         (b) only step so far as self.step_size, returning the new node within that distance"""
        diff = to_joint_angles - from_node.joint_angles
        distance = np.linalg.norm(diff)

        if distance <= self.step_size:
            new_joint_angles = to_joint_angles
        else:
            new_joint_angles = from_node.joint_angles + (diff / distance) * self.step_size

        new_node = Node(new_joint_angles)
        new_node.parent = from_node
        return new_node

    def get_nearest_node(self, random_point):
        """Find the nearest node in the tree to a given point."""
        distances = [np.linalg.norm(node.joint_angles - random_point) for node in self.node_list]
        nearest_index = np.argmin(distances)
        return self.node_list[nearest_index]

    def plan(self):
        """Run the RRT algorithm to find a path of dimension Nx3. Limit the search to only max_iter iterations."""
        for _ in range(self.max_iter):
            if random.random() < 0.1:
                random_point = self.q_goal.joint_angles
            else:
                random_point = np.array([random.uniform(lim[0], lim[1]) for lim in self.q_limits])

            nearest_node = self.get_nearest_node(random_point)

            new_node = self.step(nearest_node, random_point)

            if not check_edge_collision(self.robot_id, self.obstacle_ids, nearest_node.joint_angles, new_node.joint_angles):
                self.node_list.append(new_node)

                if np.linalg.norm(new_node.joint_angles - self.q_goal.joint_angles) < 1e-2:
                    path = []
                    current = new_node
                    while current is not None:
                        path.append(current.joint_angles)
                        current = current.parent
                    return np.array(path[::-1])

        return None

    def plan2(self):
        """Run the RRT* algorithm to find a path of dimension Nx3. Limit the search to only max_iter iterations."""
        rewire_radius = self.step_size * 2.0

        for _ in range(self.max_iter):
            if random.random() < 0.1:
                random_point = self.q_goal.joint_angles
            else:
                random_point = np.array([random.uniform(lim[0], lim[1]) for lim in self.q_limits])

            nearest_node = self.get_nearest_node(random_point)

            new_node = self.step(nearest_node, random_point)

            if check_edge_collision(self.robot_id, self.obstacle_ids, nearest_node.joint_angles, new_node.joint_angles):
                continue

            nearby_nodes = []
            for node in self.node_list:
                if np.linalg.norm(node.joint_angles - new_node.joint_angles) <= rewire_radius:
                    nearby_nodes.append(node)

            min_cost = nearest_node.cost + np.linalg.norm(new_node.joint_angles - nearest_node.joint_angles)
            best_parent = nearest_node

            for node in nearby_nodes:
                dist = np.linalg.norm(new_node.joint_angles - node.joint_angles)
                cost = node.cost + dist
                if cost < min_cost:
                    if not check_edge_collision(self.robot_id, self.obstacle_ids, node.joint_angles, new_node.joint_angles):
                        min_cost = cost
                        best_parent = node

            new_node.parent = best_parent
            new_node.cost = min_cost
            self.node_list.append(new_node)

            for node in nearby_nodes:
                if node == best_parent:
                    continue

                dist = np.linalg.norm(new_node.joint_angles - node.joint_angles)
                new_cost = new_node.cost + dist

                if new_cost < node.cost:
                    if not check_edge_collision(self.robot_id, self.obstacle_ids, new_node.joint_angles, node.joint_angles):
                        node.parent = new_node
                        node.cost = new_cost

            if np.linalg.norm(new_node.joint_angles - self.q_goal.joint_angles) < 1e-2:
                path = []
                current = new_node
                while current is not None:
                    path.append(current.joint_angles)
                    current = current.parent
                return np.array(path[::-1])

        return None

#####################################################
##################### MAIN CODE #####################
#####################################################

if __name__ == "__main__":

    #######################
    #### PROBLEM SETUP ####
    #######################

    # Initialize PyBullet with even window dimensions for video encoding
    p.connect(p.GUI, options="--width=1920 --height=1080")
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) # For default URDFs
    p.setGravity(0, 0, -9.8)

    # Load the plane and robot arm
    ground_id = p.loadURDF("plane.urdf")
    arm_id = p.loadURDF("robot.urdf", [0, 0, 0], useFixedBase=True)

    # Add Collision Objects
    collision_ids = [ground_id] # add the ground to the collision list
    collision_positions = [[0.3, 0.5, 0.251], [-0.3, 0.3, 0.101], [-1, -0.15, 0.251], [-1, -0.15, 0.752], [-0.5, -1, 0.251], [0.5, -0.35, 0.201], [0.5, -0.35, 0.602]]
    collision_orientations =  [[0, 0, 0.5], [0, 0, 0.2], [0, 0, 0],[0, 0, 1], [0, 0, 0], [0, 0, .25], [0, 0, 0.5]]
    collision_scales = [0.5, 0.25, 0.5, 0.5, 0.5, 0.4, 0.4]
    for i in range(len(collision_scales)):
        collision_ids.append(p.loadURDF("cube.urdf",
            basePosition=collision_positions[i],  # Position of the cube
            baseOrientation=p.getQuaternionFromEuler(collision_orientations[i]),  # Orientation of the cube
            globalScaling=collision_scales[i]  # Scale the cube to half size
        ))

    # Goal Joint Positions for the Robot
    goal_positions = [[-2.54, 0.15, -0.15], [-1.79,0.15,-0.15],[0.5, 0.15,-0.15], [1.7,0.2,-0.15],[-2.54, 0.15, -0.15]]

    # Joint Limits of the Robot
    joint_limits = [[-np.pi, np.pi], [0, np.pi], [-np.pi, np.pi]]

    # A3xN path array that will be filled with waypoints through all the goal positions
    path_saved = np.array([[-2.54, 0.15, -0.15]]) # Start at the first goal position

    ####################################################################################################
    #### YOUR CODE HERE: RUN RRT MOTION PLANNER FOR ALL goal_positions (starting at goal position 1) ###
    ####################################################################################################

    print("Starting RRT Planning for all segments...")
    for i in range(len(goal_positions) - 1):
        start_pos = goal_positions[i]
        end_pos = goal_positions[i+1]
        print(f"Planning segment {i+1}: {start_pos} -> {end_pos}")

        solve_times = []
        best_path = None

        # Run 10 times for statistics as requested
        for run in range(10):
            rrt = RRT(
                q_start=start_pos,
                q_goal=end_pos,
                robot_id=arm_id,
                obstacle_ids=collision_ids,
                q_limits=joint_limits,
                max_iter=10000,
                step_size=0.5
            )

            t_start = time.time()
            path = rrt.plan() #plan2()
            t_end = time.time()

            if path is not None:
                solve_times.append(t_end - t_start)
                if best_path is None:
                    best_path = path
            else:
                print(f"  Run {run+1} failed to find path.")

        if solve_times:
            avg_time = np.mean(solve_times)
            std_time = np.std(solve_times)
            print(f"  Segment {i+1} Stats (10 runs): Avg Time = {avg_time:.4f}s, Std Dev = {std_time:.4f}s")
        else:
            print(f"  Segment {i+1} failed all 10 runs.")

        if best_path is not None:
            path_saved = np.vstack((path_saved, best_path[1:]))
        else:
            print("  Stopping execution due to planning failure.")
            break

    np.savetxt('path_saved.npy', path_saved)
    print(f"steps taken: {len(path_saved)}")


    ################################################################################
    ####  RUN THE SIMULATION AND MOVE THE ROBOT ALONG PATH_SAVED ###################
    ################################################################################

    for joint_index, joint_pos in enumerate(goal_positions[0]):
        p.resetJointState(arm_id, joint_index, joint_pos)
    log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, f"Q4_RRT_result.mp4")
    for waypoint in path_saved:
        for joint_index, joint_pos in enumerate(waypoint):
            while True:
                goal_positions = [p.getJointState(arm_id, i)[0] for i in range(3)]
                displacement_to_waypoint = waypoint-goal_positions
                max_speed = 0.05
                if(np.linalg.norm(displacement_to_waypoint) < max_speed):
                    break
                else:
                    velocities = np.min((np.linalg.norm(displacement_to_waypoint), max_speed))*displacement_to_waypoint/np.linalg.norm(displacement_to_waypoint)
                    for joint_index, joint_step in enumerate(velocities):
                        p.setJointMotorControl2(
                            bodyIndex=arm_id,
                            jointIndex=joint_index,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=joint_step,
                        )

                p.stepSimulation()
        time.sleep(1.0 / 240.0)

    p.stopStateLogging(log_id)
    # Disconnect from PyBullet
    time.sleep(100) # Remove this line -- it is just to keep the GUI open when you first run this starter code
    p.disconnect()