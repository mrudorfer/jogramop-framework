import time

import tqdm
import numpy as np
import pybullet
import burg_toolkit as burg

import util
from rrt import NodeArray


class MultiTarget_Jplus_RRT:
    # an RRT variant that uses the pseudo-inverse of the jacobian to steer towards multiple targets in the world space
    # rather than in the joint space.
    # will grow the tree randomly and then try to extend towards all the goal poses using pinv(J)
    # similar to this here:
    # J+RRT from Vahrenkamp et al., 2009 "Humanoid Motion Planning for Dual-Arm Manipulation and Re-Grasping Tasks"
    def __init__(self, start_config, goal_poses, robot, max_iter=10000, step_size_q=0.005, goal_prob=0.2,
                 seed=42, goal_thresh=15., max_time=200, pick_goals_uniformly=True, dist_thresh_for_orientation=100):

        # setup robot, joint limits, and weights for q-space distance calculations (weighted by joint range)
        self.robot = robot
        self.joint_limits = self.robot.arm_joint_limits()
        # use joint ranges as weight / scale factor... it kind of normalises the joints to their range
        # -> inverse for calculating distance (i.e. the larger the range, the smaller the weight)
        # -> actual joint range for steps (i.e. for large range, take larger steps)
        joint_range = self.joint_limits[:, 1] - self.joint_limits[:, 0]
        self.q_weights = np.array(joint_range)

        # setup starting node and tree
        start_config = np.array(start_config)
        assert len(start_config) == len(self.q_weights), 'shape of weights/start config not matching'
        start_pos, start_orn = self.robot.forward_kinematics(start_config)
        self.nodes = NodeArray(len(start_config), expected_nodes=max_iter, q_weights=1/joint_range)
        start_node_id = self.nodes.add_node(start_config, parent_id=None, pos=start_pos, orn=start_orn)

        # setup targets: for each goal, in addition to tf we also remember position and orientation as quaternion
        self.goal_poses = goal_poses
        self.goal_pos, self.goal_orn = [], []
        for i in range(len(goal_poses)):
            pos, orn = burg.util.position_and_quaternion_from_tf(goal_poses[i], convention='pybullet')
            self.goal_pos.append(pos)
            self.goal_orn.append(orn)

        self.goal_pos = np.asarray(self.goal_pos)
        self.goal_orn = np.asarray(self.goal_orn)

        # calculate distance from start node to all the goals
        self.goal_dist = self.p_distance(start_pos, start_orn)
        self.closest_node = np.asarray(len(self.goal_dist) * [start_node_id])

        print('initiating multitarget-rrt')
        print(f'goal poses: {self.goal_poses.shape}')
        print(f'starting node has a distance of {self.goal_dist[0]}')
        print('weights (= joint ranges):', self.q_weights)

        # general config
        self.max_iter = max_iter
        self.step_size_q = step_size_q
        self.goal_prob = goal_prob
        self.goal_thresh = goal_thresh
        self.max_time = max_time
        self.pick_goals_uniformly = pick_goals_uniformly
        self.dist_thresh_for_orientation = dist_thresh_for_orientation

        self.rng = np.random.default_rng(seed=seed)
        self.timer = util.Timer()

    def plan(self):
        self.timer.start('total')
        success = False
        start_time = time.time()
        for i in tqdm.tqdm(range(self.max_iter)):
            if i % 1000 == 0:
                closest_grasp = np.argmin(self.goal_dist)
                print(f'{i}. {time.time() - start_time:.2f}s. {len(self.nodes)} nodes. '
                      f'closest dist: {self.goal_dist[closest_grasp]:.2f}, grasp idx {closest_grasp}')
                closest_node = self.nodes[self.closest_node[closest_grasp]]
                self.p_distance(closest_node.pos, closest_node.orn, verbose=True)
            if time.time() - start_time > self.max_time:
                break

            if self.rng.uniform() < self.goal_prob:
                self.extend_to_goal(steps=5)
            else:
                self.extend_randomly(steps=20)

            if np.min(self.goal_dist) < self.goal_thresh:
                # found a solution!
                print('success!')
                success = True
                break

        self.timer.stop('total')
        self.timer.print()
        print('****************************************************************')
        print(f'planning ended after {time.time()-start_time:.2f} seconds.')
        print(f'{len(self.nodes)} nodes were created.')
        print(f'The closest node is {np.min(self.goal_dist):.2f} away from the goal.')
        idx = np.argmin(self.goal_dist)
        return success, self.nodes.get_path_to(self.closest_node[idx])

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
            self.timer.start('calc distance update')
            distances = self.p_distance(new_pos, new_orn)
            min_indices = distances < self.goal_dist
            self.goal_dist[min_indices] = distances[min_indices]
            self.closest_node[min_indices] = new_node_id
            self.timer.stop('calc distance update')
            return self.nodes[new_node_id]

        # configuration not valid
        return None

    def extend_to_goal(self, steps=1):
        self.timer.start('extend_to_goal')
        # first, choose a random grasp to go to... however, not uniformly, but weighted by closest distance, so that
        # the grasps which have nodes nearby are picked more likely
        if self.pick_goals_uniformly:
            index = self.rng.choice(len(self.goal_dist))
        else:
            weights = 1. / self.goal_dist
            probabilities = weights / np.sum(weights)
            index = self.rng.choice(len(self.goal_dist), p=probabilities)

        # we keep track of the closest node in task space, so no need to do nearest neighbours search
        # todo: would there be some value to not choosing the closest node always, but somewhat randomly?
        nearest_node = self.nodes[self.closest_node[index]]
        while steps > 0 and nearest_node is not None:
            target_orn = None
            if self.goal_dist[index] < self.dist_thresh_for_orientation:
                target_orn = self.goal_orn[index]
            q_new = self.steer_p(nearest_node, self.goal_pos[index], target_orn)
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

    @staticmethod
    def get_closest_symmetric_grasp(rot_mat_grasp, rot_mat_ee):
        # grasps are symmetric: y-axis (in panda EE frame) can point in either direction
        # this function gets the grasp R and the ee R and returns the grasp R that is closest to ee R
        y_grasp = rot_mat_grasp[:3, 1]
        y_ee = rot_mat_ee[:3, 1]
        closest_grasp = rot_mat_grasp
        if np.dot(y_grasp, y_ee) < 0:
            # need to flip both y and x axis
            closest_grasp[:3, :2] = -closest_grasp[:3, :2]

        return closest_grasp

    @staticmethod
    def project_approach_direction(rot_mat_grasp, rot_mat_ee):
        # projects the EE's approach vector onto the grasp's approach plane, and returns the corresponding
        # grasp target
        z_ee = rot_mat_ee[:3, 2]
        y_grasp = rot_mat_grasp[:3, 1]
        z_projected = z_ee - np.dot(z_ee, y_grasp) * y_grasp
        norm = np.linalg.norm(z_projected)
        if norm == 0:
            # ee approach vector is exactly orthogonal to grasp approach plane, we can just keep original grasp approach
            return rot_mat_grasp
        z_new = z_projected / norm

        # the new grasp frame will be the original y-axis (as this is the main grasp axis), the projected z-axis, and
        # a corresponding x-axis
        x_new = np.cross(y_grasp, z_new)
        new_grasp_rot_mat = np.column_stack([x_new, y_grasp, z_new])
        return new_grasp_rot_mat

    def steer_p(self, node_from, target_pos, target_orn=None):
        # from "node_from", we make one step towards target_pos and target_orn
        # returns joint angles of new configuration

        current_pos = node_from.pos
        current_orn = node_from.orn
        current_q = node_from.q

        # calculate jacobians
        q_with_gripper = list(current_q) + [0.04, 0.04]  # jacobian wants all movable joints, including fingers
        zero_vec = [0.0] * len(q_with_gripper)
        link_com = [0.0, 0.0, 0.0]
        jac_t, jac_r = self.robot.bullet_client.calculateJacobian(
            self.robot.body_id, self.robot.end_effector_link_id,
            link_com, list(q_with_gripper), zero_vec, zero_vec)

        # calculate pseudo inverses
        jac_t, jac_r = np.asarray(jac_t), np.asarray(jac_r)
        pinv_jac_t, pinv_jac_r = np.linalg.pinv(jac_t), np.linalg.pinv(jac_r)

        # make position update
        delta_pos = target_pos - current_pos
        new_joint_pos = q_with_gripper + pinv_jac_t @ delta_pos

        # orientation update is optional
        if target_orn is not None:
            # first, check grasp symmetry and get the closest of the two but keep approach direction
            R_grasp = np.asarray(pybullet.getMatrixFromQuaternion(target_orn)).reshape(3, 3)
            R_ee = np.asarray(pybullet.getMatrixFromQuaternion(current_orn)).reshape(3, 3)
            R_grasp = self.get_closest_symmetric_grasp(R_grasp, R_ee)
            target_orn_with_approach = util.quaternion_from_rotation_matrix(R_grasp)

            # now, we also determine a target that has the closest approach direction, this will be main target
            R_closest_target = self.project_approach_direction(R_grasp, R_ee)
            target_orn = util.quaternion_from_rotation_matrix(R_closest_target)

            # get the difference errors
            q_diff_nearest = pybullet.getDifferenceQuaternion(target_orn, current_orn)
            delta_orn_nearest = np.asarray(pybullet.getEulerFromQuaternion(q_diff_nearest))
            q_diff_approach = pybullet.getDifferenceQuaternion(target_orn_with_approach, current_orn)
            delta_orn_approach = np.asarray(pybullet.getEulerFromQuaternion(q_diff_approach))
            alpha = 0.7  # how much to prefer the closest grasp over the original approach
            delta_orn = alpha * delta_orn_nearest + (1 - alpha) * delta_orn_approach  # todo: should probably do the weighting in quaternions
            new_joint_pos += -pinv_jac_r @ delta_orn

        # remove finger joints again
        new_joint_pos = new_joint_pos[:-2]

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

    def p_distance(self, pos1, orn1, verbose=False):
        # calculates distance between given grasp and all the goal grasps
        # this distance metric should reflect the following:
        # - we want to reach the contact points as accurately as possible (pos error should have high weight)
        # - angle between grasp axis of gripper and target
        # - angle between approach axis and approach plane
        # calculates distance in task space, balancing translational and rotational error such that 1mm = 1degree

        pos_dist = np.linalg.norm(self.goal_poses[:, :3, 3] - pos1, axis=-1)
        tf = burg.util.tf_from_pos_quat(pos1, orn1, convention='pybullet')
        grasp_angle, approach_angle = self.orientation_alignment(tf)
        p_distance = 1000 * pos_dist + grasp_angle + approach_angle  # 1mm = 1deg
        if verbose:
            i = np.argmin(p_distance)
            print(f'dist: {p_distance[i]} = {1000 * pos_dist[i]}mm + {grasp_angle[i]}deg + {approach_angle[i]}deg')
        return p_distance

    def orientation_alignment(self, gripper_ee_tf):
        """
        given the gripper EE transform, we calculate two angles:
        1. angle between the grasp axis of the gripper and the grasp target (aligns contact force direction)
        2. angle between the approach direction and the approach plane
        :param gripper_ee_tf: (4, 4) ndarray
        :return: (float, float) angles in degrees
        """
        gripper_y_vec = gripper_ee_tf[:3, 1]
        gripper_z_vec = gripper_ee_tf[:3, 2]
        grasp_y_vec = self.goal_poses[:, :3, 1]

        # 1. angle between the grasp axis of the gripper and the grasp target (aligns contact force direction)
        # we use einsum to calculate the dot product fast
        # then absolute value (since grasps are symmetric)
        # then minimum of that and 1, to avoid numerical issues when dot product is 1.000000002
        # then arccos and convert to degree
        # the smaller the angle the better
        grasp_angle = np.rad2deg(np.arccos(np.minimum(np.abs(np.einsum('j,ij->i', gripper_y_vec, grasp_y_vec)), 1)))

        # 2. angle between the approach direction and the approach plane
        # grasp_y_vec is orthogonal to approach plane, so maximum angle to that is the closest distance to the approach
        # plane
        # again use einsum to calculate dot product fast
        # abs because of symmetry, minimum for numerical reasons
        approach_angle = np.rad2deg(np.arccos(np.minimum(np.abs(np.einsum('j,ij->i', gripper_z_vec, grasp_y_vec)), 1)))
        # turn angle to y_vec into angle to plane
        approach_angle = 90 - approach_angle

        return grasp_angle, approach_angle


