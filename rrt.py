import time

import numpy as np
import pybullet
import tqdm

import util


class NodeArray:
    BASE_LEN = 8  # parent_id (1), pos (3), orn (4), + config (dynamic)
    """
    Utilising numpy array to speedup nearest neighbour computations etc.
    :param q_len: int, length of a single configuration (dimension of the configuration space)
    :param expected_nodes: optional, the number of nodes we expect to be stored
    :param q_weights: currently unused, weighting factors the individual elements in q for distance calculation 
    """
    def __init__(self, q_len, expected_nodes=10000, q_weights=None):
        self._q_len = q_len
        # initialise internally used array
        self._array = np.empty((expected_nodes, NodeArray.BASE_LEN + q_len), dtype=float)
        # keep track of actual number of nodes
        self._n_actual_nodes = 0

        # weights for distance computation
        if q_weights is not None:
            assert len(q_weights) == q_len, 'unexpected length of q_weights'
            q_weights = np.array(q_weights).reshape(1, q_len)
        self._q_weights = q_weights

    def __len__(self):
        return self._n_actual_nodes

    def __getitem__(self, item):
        assert isinstance(item, (int, np.integer)), f'unsupported index type. got: {type(item)}'
        assert item < len(self), f'index {item} out of bounds for len {len(self)}'
        return Node(self._array[item], item)

    def _ensure_capacity(self):
        """ doubles the capacity of the internal array in case no capacity left """
        if self._n_actual_nodes < len(self._array):
            return
        new_array = np.empty((2 * len(self._array), NodeArray.BASE_LEN + self._q_len), dtype=float)
        new_array[:len(self._array)] = self._array
        print(f'NodeArray: Increasing capacity from {len(self._array)} to {len(new_array)}')
        self._array = new_array

    def add_node(self, q, parent_id, pos=None, orn=None):
        """ adds a node to this NodeArray and returns the node_id """
        self._ensure_capacity()
        assert len(q) == self._q_len, 'unexpected length of given configuration q'
        if parent_id is None:
            parent_id = -1
        node_id = self._n_actual_nodes
        self._array[node_id, 0] = parent_id
        if pos is not None:
            self._array[node_id, 1:4] = pos
        if orn is not None:
            self._array[node_id, 4:8] = orn
        self._array[node_id, 8:] = q
        self._n_actual_nodes += 1
        return node_id

    def get_nearest_node_in_c_space(self, query_q):
        """ calculates distances to all nodes and returns the nearest node """
        assert len(query_q) == self._q_len, 'unexpected length of configuration'
        query_q = np.asarray(query_q).reshape(1, self._q_len)
        q_array = self._array[:self._n_actual_nodes, 8:]
        # apply weights to distance calculation
        if self._q_weights is not None:
            distances = np.linalg.norm(self._q_weights * (q_array - query_q), axis=-1)
        else:
            distances = np.linalg.norm(q_array - query_q, axis=-1)
        nearest_node_id = np.argmin(distances)
        return self[nearest_node_id]

    def get_path_to(self, node_id):
        waypoints = []
        while node_id >= 0:
            node = self[node_id]
            waypoints.append(node.q)
            node_id = node.parent_id

        waypoints.reverse()
        return waypoints


class Node:
    def __init__(self, array, node_id):
        """ don't use this constructor yourself, it's used by NodeArray (add_node) """
        self._array = array
        self._id = node_id

    @property
    def node_id(self):
        return int(self._id)

    @property
    def parent_id(self):
        return int(self._array[0])

    @property
    def pos(self):
        return self._array[1:4]

    @property
    def orn(self):
        return self._array[4:8]

    @property
    def q(self):
        return self._array[8:]


class RRT:
    # basic RRT
    def __init__(self, start_config, target_config, robot, max_iter=10000, step_size_q=0.005, goal_prob=0.2,
                 seed=42, goal_thresh=0.001, max_time=200):

        self.target_config = target_config
        self.robot = robot
        self.joint_limits = self.robot.arm_joint_limits()
        # use joint ranges as weight / scale factor... it kind of normalises the joints to their range
        # -> inverse for calculating distance (i.e. the larger the range, the smaller the weight)
        # -> actual joint range for steps (i.e. for large range, take larger steps)
        joint_range = self.joint_limits[:, 1] - self.joint_limits[:, 0]
        self.q_weights = np.array(joint_range)

        start_config = np.array(start_config)
        assert len(start_config) == len(self.q_weights), 'shape of weights/start config not matching'
        self.nodes = NodeArray(len(start_config), expected_nodes=max_iter, q_weights=1/joint_range)
        start_node_id = self.nodes.add_node(start_config, parent_id=None, pos=None, orn=None)

        self.closest_node_to_goal = start_node_id
        self.closest_dist_to_goal = self.distance_to_target(start_config)

        self.max_iter = max_iter
        self.step_size_q = step_size_q
        self.goal_prob = goal_prob
        self.goal_thresh = goal_thresh
        self.max_time = max_time

        self.rng = np.random.default_rng(seed=seed)
        self.timer = util.Timer()

    def plan(self):
        """
        :return: tuple(bool, list[waypoint]) - bool indicates success, waypoints are the best current solution
        """
        self.timer.start('total')
        start_time = time.time()
        success = False
        for _ in tqdm.tqdm(range(self.max_iter)):
            if time.time() - start_time > self.max_time:
                break

            if self.rng.uniform() < self.goal_prob:
                self.extend_to_goal(steps=2)
            else:
                self.extend_randomly(steps=5)

            if self.closest_dist_to_goal < self.goal_thresh:
                # found a solution!
                print('success!')
                success = True
                break

        self.timer.stop('total')
        self.timer.print()
        print('****************************************************************')
        print(f'planning ended after {time.time()-start_time:.2f} seconds.')
        print(f'{len(self.nodes)} nodes were created.')
        print(f'The closest node is {self.closest_dist_to_goal:.2f} away from the goal.')
        return success, self.nodes.get_path_to(self.closest_node_to_goal)

    def check_and_add_node(self, new_q, parent_node):
        """
        Checks whether a new configuration is valid; if yes, will be added to the tree.

        :param new_q: new joint config
        :param parent_node: parent node of the potential new node
        :return: the new node, or None if new_q was not valid
        """
        if self.config_in_joint_limits(new_q) and self.config_collision_free(new_q):
            # make it a node and add it to the path
            new_node_id = self.nodes.add_node(new_q, parent_node.node_id)

            # check distance from goal and update the best candidate node
            dist = self.distance_to_target(new_q)
            if dist < self.closest_dist_to_goal:
                self.closest_dist_to_goal = dist
                self.closest_node_to_goal = new_node_id
                print(f'new closest node. dist: {dist:.2f}')

            return self.nodes[new_node_id]

        # configuration not valid
        return None

    def distance_to_target(self, config):
        distance = np.linalg.norm((self.target_config - config) / self.q_weights)
        return distance

    def extend_to_goal(self, steps=1):
        self.timer.start('extend_to_goal')
        # we keep track of the closest node, so no need to do nearest neighbours search
        nearest_node = self.nodes[self.closest_node_to_goal]
        while steps > 0 and nearest_node is not None:
            q_new = self.steer_q(nearest_node, self.target_config)
            nearest_node = self.check_and_add_node(q_new, nearest_node)
            steps -= 1

        self.timer.stop('extend_to_goal')

    def extend_randomly(self, steps=1):
        self.timer.start('extend_randomly')
        # get a collision-free random configuration from within joint limits
        q_rand = self.random_config()
        while not self.config_collision_free(q_rand):
            q_rand = self.random_config()

        # find the closest node in tree
        self.timer.start('nearest_neighbor')
        nearest_node = self.nodes.get_nearest_node_in_c_space(q_rand)
        self.timer.stop('nearest_neighbor')

        # generate new joint configs towards the random config
        while steps > 0 and nearest_node is not None:
            q_new = self.steer_q(nearest_node, q_rand)
            nearest_node = self.check_and_add_node(q_new, nearest_node)
            steps -= 1
            # do not need to take any further steps if we are already at q_rand
            if np.allclose(q_rand, q_new):
                break

        self.timer.stop('extend_randomly')

    def steer_q(self, node_from, q_to):
        # provides a new joint config one step closer towards q_to
        # steps are taken per joint, i.e. self.step_size_q is for each joint individually (normalised by joint range)
        q_from = node_from.q

        q_new = np.empty_like(q_to)
        # get difference relative to joint range (weight)
        weighted_q_diff = (q_to - q_from) / self.q_weights
        # if joint is less than one step away, we just go to target immediately
        near_joints = np.abs(weighted_q_diff) <= self.step_size_q
        q_new[near_joints] = q_to[near_joints]

        # move remaining joints by step_size_q towards target
        step = ~near_joints
        q_new[step] = q_from[step] + np.sign(weighted_q_diff[step]) * self.step_size_q * self.q_weights[step]

        return q_new

    def random_config(self):
        q_rand = self.rng.uniform(low=self.joint_limits[:, 0], high=self.joint_limits[:, 1])
        return q_rand

    def config_in_joint_limits(self, config):
        low = self.joint_limits[:, 0]
        high = self.joint_limits[:, 1]
        return np.all(config > low) and np.all(config < high)

    def config_collision_free(self, config):
        self.timer.start('collision_check')
        self.timer.count('collision_check')
        self.robot.reset_arm_joints(config)
        if self.robot.in_self_collision():
            self.timer.stop('collision_check')
            return False
        if self.robot.in_collision():
            self.timer.stop('collision_check')
            return False
        self.timer.stop('collision_check')
        return True


class Jplus_RRT:
    # an RRT variant that uses the pseudo-inverse of the jacobian to steer towards a target in the world space rather
    # than in the joint space. will grow the tree randomly and then try to extend towards the goal pose using pinv(J)
    # similar to this here:
    # J+RRT from Vahrenkamp et al., 2009 "Humanoid Motion Planning for Dual-Arm Manipulation and Re-Grasping Tasks"
    def __init__(self, start_config, goal_pos, goal_orn, robot, max_iter=10000, step_size_q=0.005, goal_prob=0.2,
                 seed=42, goal_thresh=15., max_time=200):

        self.goal_pos, self.goal_orn = goal_pos, goal_orn
        self.robot = robot
        self.joint_limits = self.robot.arm_joint_limits()
        # use joint ranges as weight / scale factor... it kind of normalises the joints to their range
        # -> inverse for calculating distance (i.e. the larger the range, the smaller the weight)
        # -> actual joint range for steps (i.e. for large range, take larger steps)
        joint_range = self.joint_limits[:, 1] - self.joint_limits[:, 0]
        self.q_weights = np.array(joint_range)

        start_config = np.array(start_config)
        assert len(start_config) == len(self.q_weights), 'shape of weights/start config not matching'
        pos, orn = self.robot.forward_kinematics(start_config)
        self.nodes = NodeArray(len(start_config), expected_nodes=max_iter, q_weights=1/joint_range)
        start_node_id = self.nodes.add_node(start_config, parent_id=None, pos=pos, orn=orn)

        self.closest_node_to_goal = start_node_id
        self.closest_dist_to_goal = self.p_distance(pos, orn, self.goal_pos, self.goal_orn)

        self.max_iter = max_iter
        self.step_size_q = step_size_q
        self.goal_prob = goal_prob
        self.goal_thresh = goal_thresh
        self.max_time = max_time

        self.rng = np.random.default_rng(seed=seed)
        self.timer = util.Timer()

    def plan(self):
        self.timer.start('total')
        success = False
        start_time = time.time()
        for _ in tqdm.tqdm(range(self.max_iter)):
            if time.time() - start_time > self.max_time:
                break

            if self.rng.uniform() < self.goal_prob:
                self.extend_to_goal(steps=2)
            else:
                self.extend_randomly(steps=5)

            if self.closest_dist_to_goal < self.goal_thresh:
                # found a solution!
                print('success!')
                success = True
                break

        self.timer.stop('total')
        self.timer.print()
        print('****************************************************************')
        print(f'planning ended after {time.time()-start_time:.2f} seconds.')
        print(f'{len(self.nodes)} nodes were created.')
        print(f'The closest node is {self.closest_dist_to_goal:.2f} away from the goal.')
        return success, self.nodes.get_path_to(self.closest_node_to_goal)

    def check_and_add_node(self, new_q, parent_node):
        """
        Checks whether a new configuration is valid; if yes, will be added to the tree.

        :param new_q: new joint config
        :param parent_node: parent node of the potential new node
        :return: the new node, or None if new_q was not valid
        """
        if self.config_in_joint_limits(new_q) and self.config_collision_free(new_q):
            # make it a node and add it to the path
            new_pos, new_orn = self.robot.forward_kinematics(new_q)
            new_node_id = self.nodes.add_node(new_q, parent_node.node_id, new_pos, new_orn)

            # check distance from goal and update the best candidate node
            dist = self.p_distance(new_pos, new_orn, self.goal_pos, self.goal_orn)
            if dist < self.closest_dist_to_goal:
                self.closest_dist_to_goal = dist
                self.closest_node_to_goal = new_node_id
                print(f'new closest node. dist: {dist:.2f}')

            return self.nodes[new_node_id]

        # configuration not valid
        return None

    def extend_to_goal(self, steps=1):
        self.timer.start('extend_to_goal')
        # we keep track of the closest node in task space, so no need to do nearest neighbours search
        nearest_node = self.nodes[self.closest_node_to_goal]
        while steps > 0 and nearest_node is not None:
            q_new = self.steer_p(nearest_node, self.goal_pos, self.goal_orn)
            nearest_node = self.check_and_add_node(q_new, nearest_node)
            steps -= 1

        self.timer.stop('extend_to_goal')

    def extend_randomly(self, steps=1):
        self.timer.start('extend_randomly')
        # get a collision-free random configuration from within joint limits
        q_rand = self.random_config()
        while not self.config_collision_free(q_rand):
            q_rand = self.random_config()

        # find the closest node in tree
        self.timer.start('nearest_neighbor')
        nearest_node = self.nodes.get_nearest_node_in_c_space(q_rand)
        self.timer.stop('nearest_neighbor')

        # generate new joint configs towards the random config
        while steps > 0 and nearest_node is not None:
            q_new = self.steer_q(nearest_node, q_rand)
            nearest_node = self.check_and_add_node(q_new, nearest_node)
            steps -= 1
            # do not need to take any further steps if we are already at q_rand
            if np.allclose(q_rand, q_new):
                break

        self.timer.stop('extend_randomly')

    def steer_q(self, node_from, q_to):
        # provides a new joint config one step closer towards q_to
        # steps are taken per joint, i.e. self.step_size_q is for each joint individually (normalised by joint range)
        q_from = node_from.q

        q_new = np.empty_like(q_to)
        # get difference relative to joint range (weight)
        weighted_q_diff = (q_to - q_from) / self.q_weights
        # if joint is less than one step away, we just go to target immediately
        near_joints = np.abs(weighted_q_diff) <= self.step_size_q
        q_new[near_joints] = q_to[near_joints]

        # move remaining joints by step_size_q towards target
        step = ~near_joints
        q_new[step] = q_from[step] + np.sign(weighted_q_diff[step]) * self.step_size_q * self.q_weights[step]

        return q_new

    def steer_p(self, node_from, target_pos, target_orn, step_size_pos=0.1, step_size_orn=10):
        # from "node_from", we make one step towards target_pos and target_orn
        # returns joint angles of new configuration

        # step_size_orn is in degree
        step_size_orn = np.deg2rad(step_size_orn)
        current_pos = node_from.pos
        current_orn = node_from.orn
        current_q = node_from.q

        # calculate delta p
        delta_pos = target_pos - current_pos
        pos_error = np.linalg.norm(delta_pos)

        q_diff = pybullet.getDifferenceQuaternion(target_orn, current_orn)
        delta_orn = np.asarray(pybullet.getEulerFromQuaternion(q_diff))
        orn_error = 2 * np.arccos(np.abs(q_diff[3]))

        # calculate jacobian and its inverse
        q_with_gripper = list(current_q) + [0.04, 0.04]  # jacobian wants all movable joints, including fingers
        zero_vec = [0.0] * len(q_with_gripper)
        link_com = [0.0, 0.0, 0.0]
        jac_t, jac_r = self.robot.bullet_client.calculateJacobian(
            self.robot.body_id, self.robot.end_effector_link_id,
            link_com, list(q_with_gripper), zero_vec, zero_vec)

        jac_t, jac_r = np.asarray(jac_t), np.asarray(jac_r)
        pinv_jac_t, pinv_jac_r = np.linalg.pinv(jac_t), np.linalg.pinv(jac_r)

        if np.linalg.matrix_rank(pinv_jac_t) < 3 or np.linalg.matrix_rank(pinv_jac_r) < 3:
            print('pseudo-inverse Jacobian does not have full rank')
            print('rank pinv jac_t', np.linalg.matrix_rank(pinv_jac_t))
            print('rank pinv jac_r', np.linalg.matrix_rank(pinv_jac_r))

        # restrict steps to certain step size
        if pos_error > step_size_pos:
            delta_pos *= step_size_pos/pos_error
        if orn_error > step_size_orn:
            delta_orn *= step_size_orn/orn_error

        # make update
        new_joint_pos = q_with_gripper + pinv_jac_t @ delta_pos
        new_joint_pos += -pinv_jac_r @ delta_orn
        new_joint_pos = new_joint_pos[:-2]  # remove finger joints again

        # make q-step - so both types of steps will be similarly large
        q_new = self.steer_q(node_from, new_joint_pos)
        if np.allclose(q_new, new_joint_pos):
            self.timer.count('p-step smaller than q-step')
        else:
            self.timer.count('q-step smaller than p-step')
        return q_new

    def random_config(self):
        q_rand = self.rng.uniform(low=self.joint_limits[:, 0], high=self.joint_limits[:, 1])
        return q_rand

    def config_in_joint_limits(self, config):
        low = self.joint_limits[:, 0]
        high = self.joint_limits[:, 1]
        return np.all(config > low) and np.all(config < high)

    def config_collision_free(self, config):
        self.timer.start('collision_check')
        self.timer.count('collision_check')
        self.robot.reset_arm_joints(config)
        if self.robot.in_self_collision():
            self.timer.stop('collision_check')
            return False
        if self.robot.in_collision():
            self.timer.stop('collision_check')
            return False
        self.timer.stop('collision_check')
        return True

    @staticmethod
    def p_distance(pos1, orn1, pos2, orn2):
        # calculates distance in task space, balancing translational and rotational error such that 1mm = 1degree
        pos_dist = np.linalg.norm(pos2-pos1)
        diff_quat = pybullet.getDifferenceQuaternion(orn1, orn2)
        angle = np.rad2deg(2 * np.arccos(np.abs(diff_quat[3])))
        p_distance = 1000 * pos_dist + angle  # 1mm = 1deg
        return p_distance


def test_self_collisions():
    from simulation import GraspingSimulator, FrankaRobot
    import burg_toolkit as burg
    burg.log_to_console()

    sim = GraspingSimulator(verbose=True)
    pose = np.eye(4)
    pose[2, 3] = 0.05
    robot = FrankaRobot(sim, pose, with_platform=False)
    rrt = Jplus_RRT(robot.arm_joints_pos(), None, None, robot)

    for i in range(10):
        q = rrt.random_config()
        robot.reset_arm_joints(q)
        print('**** checking self collisions')
        print(robot.in_self_collision())
        print('**** checking other collisions')
        print(robot.in_collision())
        input('enter to proceed')


def test_rrt():
    from simulation import GraspingSimulator, FrankaRobot
    import burg_toolkit as burg

    sim = GraspingSimulator(verbose=False)
    pose = np.eye(4)
    pose[2, 3] = 0.05
    robot = FrankaRobot(sim, pose, with_platform=True)
    target_ee = np.eye(4)
    target_ee[:3, 3] = [1, 0.5, 0.2]
    pos, orn = burg.util.position_and_quaternion_from_tf(target_ee @ robot.tf_grasp2ee, convention='pybullet')
    rrt = Jplus_RRT(robot.arm_joints_pos(), pos, orn, robot, seed=2, goal_thresh=15, step_size_q=0.5)
    print('starting to plan')
    success, waypoints = rrt.plan()
    print('finished planning')
    print(f'used {len(rrt.nodes)} nodes to plan a trajectory with {len(waypoints)} waypoints')

    sim = GraspingSimulator(verbose=True)
    robot = FrankaRobot(sim, pose, with_platform=True)

    for i in range(len(waypoints)):
        robot.reset_arm_joints(waypoints[i])
        input(f'{i}/{len(waypoints)}: enter to proceed')


def test_rrt2():
    import burg_toolkit as burg
    from simulation import GraspingSimulator, FrankaRobot
    import visualization

    scene_fn = 'scenes/sugarbox_verticalgap_20cm.yaml'

    sim = GraspingSimulator(verbose=False)
    sim.add_scene_from_file(scene_fn)
    robot_pose = np.eye(4)
    robot_pose[2, 3] = 0.05
    robot = FrankaRobot(sim, robot_pose, with_platform=True)
    target_ee = robot.get_ee_pose_for_grasp(util.get_fake_grasp(pos=[1.0, 1.0, 0.2]))
    pos, orn = burg.util.position_and_quaternion_from_tf(target_ee, convention='pybullet')

    rrt = Jplus_RRT(robot.arm_joints_pos(), pos, orn, robot, seed=3, goal_thresh=15, max_time=30, max_iter=35000,
                    step_size_q=0.02)
    print('starting to plan')
    success, waypoints = rrt.plan()
    print('finished planning')
    print(f'used {len(rrt.nodes)} nodes to plan a trajectory with {len(waypoints)} waypoints')

    print('waypoints')
    print(waypoints)

    visualization.visualize_waypoints(waypoints, scene_fn, target_ee, robot_pose)


if __name__ == '__main__':
    test_rrt2()
