import logging

import numpy as np
import burg_toolkit as burg
import pybullet

import util

_log = logging.getLogger(__name__)


class GraspingSimulator(burg.sim.SimulatorBase):
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)
        self._reset(plane_and_gravity=True)

        if self.verbose:
            # adjust camera
            look_at = [1, 0.5, 0.2]
            yaw = 110  # left/right degree
            pitch = -45  # up/down degree
            distance = 1.5  # [m]
            self._p.resetDebugVisualizerCamera(distance, yaw, pitch, look_at)
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI, 0)

    def add_frame(self, pos=None, orn=None, tf=None, scale=0.1):
        if tf is None:
            pos = pos if pos is not None else [0, 0, 0]
            orn = orn if orn is not None else [0, 0, 0, 1]
            tf = burg.util.tf_from_pos_quat(pos, orn, convention='pybullet')
        # pos, orn = burg.util.position_and_quaternion_from_tf(tf)

        body_id = self.bullet_client.loadURDF('robots/frame_vis/frame_vis.urdf',
                                              # basePosition=pos,
                                              # baseOrientation=orn,
                                              useFixedBase=True,
                                              globalScaling=scale
                                              )

        com = np.array(self.bullet_client.getDynamicsInfo(body_id, -1)[3])
        tf_burg2py = np.eye(4)
        tf_burg2py[0:3, 3] = com
        start_pose = tf @ tf_burg2py
        pos, quat = burg.util.position_and_quaternion_from_tf(start_pose, convention='pybullet')
        self.bullet_client.resetBasePositionAndOrientation(body_id, pos, quat)

        return body_id

    def add_sphere(self, pos=None, orn=None, tf=None, scale=0.1):
        if tf is None:
            pos = pos if pos is not None else [0, 0, 0]
            orn = orn if orn is not None else [0, 0, 0, 1]
            tf = burg.util.tf_from_pos_quat(pos, orn, convention='pybullet')
        # pos, orn = burg.util.position_and_quaternion_from_tf(tf)

        body_id = self.bullet_client.loadURDF('robots/frame_vis/sphere_vis.urdf',
                                              # basePosition=pos,
                                              # baseOrientation=orn,
                                              useFixedBase=True,
                                              globalScaling=scale
                                              )

        com = np.array(self.bullet_client.getDynamicsInfo(body_id, -1)[3])
        tf_burg2py = np.eye(4)
        tf_burg2py[0:3, 3] = com
        start_pose = tf @ tf_burg2py
        pos, quat = burg.util.position_and_quaternion_from_tf(start_pose, convention='pybullet')
        self.bullet_client.resetBasePositionAndOrientation(body_id, pos, quat)

        return body_id




    def remove(self, body_id):
        self.bullet_client.removeBody(body_id)

    def add_scene_from_file(self, scene_fn):
        scene, _, _ = burg.Scene.from_yaml(scene_fn)
        self.add_scene(scene)

    def links_in_collision(self, body1, link1, body2, link2, threshold=-0.001):
        # based on body id and link id

        # in contrast to getContactPoints, this also works before stepSimulation or performCollisionDetection
        distance = 0.01  # do not return any points for objects that are farther apart than this
        points = self._p.getClosestPoints(body1, body2, distance, linkIndexA=link1, linkIndexB=link2)
        _log.debug(f'found {len(points)} points that are close between links {link1} and {link2}')

        for point in points:
            distance = point[8]
            if distance < threshold:
                _log.debug(f'collision detected between links {link1} and {link2}')
                return True
        return False

    def body_in_collision(self, query_body):
        """
        Checks collisions of query_body with moving objects and environment
        Returns true if in collision.
        """
        # checking collisions against environment objects (ground plane)
        _log.debug('checking collisions with environment bodies')
        for body_key, body_id in self._env_bodies.items():
            if query_body == body_id:
                continue
            if self.are_in_collision(query_body, body_id):
                _log.debug(f'body in collision with {body_key} ({body_id})')
                return True

        # checking collisions with other scene objects
        _log.debug('checking collisions with other bodies')
        for body_key, body_id in self._moving_bodies.items():
            if query_body == body_id:
                continue
            if self.are_in_collision(query_body, body_id):
                _log.debug(f'body in collision with {body_key} ({body_id})')
                return True

        _log.debug('COLLISION CHECKS PASSED')
        return False


class FrankaRobot(burg.robots.RobotBase):
    tf_grasp2ee = np.asarray([  # transformation from grasp to end-effector frame
        0, 1, 0, 0,
        1, 0, 0, 0,
        0, 0, -1, 0,
        0, 0, 0, 1
    ]).reshape(4, 4)

    def __init__(self, simulator, pose, with_platform=False):
        super().__init__(simulator)
        self._bullet_client = self._simulator.bullet_client

        # load robot
        urdf_fn = 'robots/franka_panda/panda.urdf'
        if with_platform:
            urdf_fn = 'robots/franka_panda/mobile_panda.urdf'

        pos, quat = burg.util.position_and_quaternion_from_tf(pose, convention='pybullet')
        self._body_id, self.robot_joints = self._simulator.load_robot(
            urdf_fn, position=pos, orientation=quat, fixed_base=True
        )

        if pos[2] < 0.05:
            print('WARNING: consider a larger z-value for your robot, to avoid false positives during collision '
                  'detection with the plane. current pos: ', pos)

        # display joint infos
        if simulator.verbose:
            for joint, info in self.robot_joints.items():
                print(joint, info)

        # joint id defs
        self.with_platform = with_platform
        if with_platform:
            self.end_effector_id = 14
            self.arm_joint_ids = [0, 1, 3, 4, 5, 6, 7, 8, 9]  # includes base X/Y mobile platform
            self.finger_joint_ids = [12, 13]
        else:
            self.end_effector_id = 11
            self.arm_joint_ids = [0, 1, 2, 3, 4, 5, 6]
            self.finger_joint_ids = [9, 10]

        self.home_conf = [-0.017792060227770554, -0.7601235411041661, 0.019782607023391807, -2.342050140544315,
                          0.029840531355804868, 1.5411935298621688, 0.7534486589746342]
        if with_platform:
            self.home_conf = [0.0, 0.0] + self.home_conf

        # gripper
        self.finger_open_distance = 0.04
        self.finger_force = 100
        self.grasp_speed = 0.1
        self.configure_gripper_friction()
        self._simulator.register_step_func(self.gripper_constraints)  # both fingers act symmetrically

        self.reset_home_pose()

    @property
    def bullet_client(self):
        return self._bullet_client

    def reset_home_pose(self):
        self.reset_arm_joints()
        self.reset_gripper()

    def get_ee_pose_for_grasp(self, grasp):
        raise DeprecationWarning('this is being done during loading of scenario. dont use directly anymore')
        # based on a grasp from BURG toolkit, returns the target pose for the end effector
        # this is due to the end effector using a different frame than the grasp convention
        ee_pose = grasp.pose @ self.tf_grasp2ee
        return ee_pose

    @property
    def end_effector_link_id(self):
        return self.end_effector_id

    def end_effector_pose(self):
        pos, quat, *_ = self._simulator.bullet_client.getLinkState(
            self.body_id,
            self.end_effector_link_id
        )
        return burg.util.tf_from_pos_quat(pos, quat, convention='pybullet')

    def reset_arm_joints(self, joint_config=None):
        # just the 7dof robot arm
        if joint_config is None:
            joint_config = self.home_conf
        assert len(joint_config) == len(self.arm_joint_ids)
        for joint_id, q in zip(self.arm_joint_ids, joint_config):
            self._bullet_client.resetJointState(self.body_id, joint_id, q)

    def get_arm_joint_conf_from_motor_joint_conf(self, ik_joint_config):
        """
        pybullet ik returns all movable joints including gripper, excluding fixed joints.
        this function gives the arm joints based on ik joints.
        """
        # arm joints are always the first n joints, as we count the platform as part of the arm
        arm_joints = np.asarray(ik_joint_config)[:len(self.arm_joint_ids)]
        return arm_joints

    def forward_kinematics(self, joint_conf, as_matrix=False):
        """
        gets the end effector pose for the given robot in the given arm joint angles.
        returns position and quaternion, unless as_matrix=True, then it will return 4x4 matrix

        caution: resets the robots joints. do not use in active simulation.
        """
        if len(joint_conf) == len(self.arm_joint_ids):
            arm_joint_conf = joint_conf
        elif len(joint_conf) == len(self.arm_joint_ids) + len(self.finger_joint_ids):
            arm_joint_conf = self.get_arm_joint_conf_from_motor_joint_conf(joint_conf)
        elif len(joint_conf) == self.end_effector_link_id:
            arm_joint_conf = np.asarray(joint_conf)[self.arm_joint_ids]
        else:
            raise ValueError('cannot match joint conf to arm/motor/all joints; unexpected length.')

        # set robot joint angles
        self.reset_arm_joints(arm_joint_conf)
        # retrieve EE pose
        pos, quat, *_ = self.bullet_client.getLinkState(
            self.body_id,
            self.end_effector_link_id,
            computeForwardKinematics=True
        )
        if as_matrix:
            return burg.util.tf_from_pos_quat(pos, quat, convention='pybullet')
        return np.asarray(pos), np.asarray(quat)

    def inverse_kinematics(self, pos, orn=None, combined_threshold=0.015, null_space_control=False):
        """
        checks that given solution is indeed close to the desired pos/orn

        :param pos: 3d position
        :param orn: orientation as quaternion, optional
        :param null_space_control: whether to use null space control (will try to stay close to home conf then)
        :param combined_threshold: combined threshold for position and orientation to accept IK solution (deg=mm), in m
        :return: joint configuration, or None if no solution found
        """
        iterations = 100
        threshold = 0.001

        if null_space_control:
            lower_limits = self.arm_joint_limits()[:, 0]
            upper_limits = self.arm_joint_limits()[:, 1]
            joint_ranges = upper_limits - lower_limits
            rest_poses = self.home_conf

            # however, for IK we also need to use finger joints
            lower_limits = lower_limits.tolist() + [0.0, 0.0]
            upper_limits = upper_limits.tolist() + [0.04, 0.04]
            joint_ranges = joint_ranges.tolist() + [0.04, 0.04]
            rest_poses = rest_poses + [0.04, 0.04]

            if orn is None:
                joint_positions = self.bullet_client.calculateInverseKinematics(
                    self.body_id, self.end_effector_link_id, pos,
                    lowerLimits=lower_limits, upperLimits=upper_limits, jointRanges=joint_ranges, restPoses=rest_poses,
                    maxNumIterations=iterations, residualThreshold=threshold)
            else:
                joint_positions = self.bullet_client.calculateInverseKinematics(
                    self.body_id, self.end_effector_link_id, pos, orn,
                    lowerLimits=lower_limits, upperLimits=upper_limits, jointRanges=joint_ranges, restPoses=rest_poses,
                    maxNumIterations=iterations, residualThreshold=threshold)
        else:
            if orn is None:
                joint_positions = self.bullet_client.calculateInverseKinematics(
                    self.body_id, self.end_effector_link_id, pos,
                    maxNumIterations=iterations, residualThreshold=threshold)
            else:
                joint_positions = self.bullet_client.calculateInverseKinematics(
                    self.body_id, self.end_effector_link_id, pos, orn,
                    maxNumIterations=iterations, residualThreshold=threshold)

        joint_positions = self.get_arm_joint_conf_from_motor_joint_conf(joint_positions)

        # ensure solution has reached the desired pose
        actual_pos, actual_orn = self.forward_kinematics(joint_positions)
        pos_diff = np.linalg.norm(pos - actual_pos)
        if orn is not None:
            orn_diff = util.angle_between_quaternions(orn, actual_orn, as_degree=True)
        else:
            orn_diff = 0

        if pos_diff + orn_diff/1000.0 <= combined_threshold:
            return joint_positions

        # print(f'diff too big: {pos_diff}m, {orn_diff}deg')
        return None

    def reset_gripper(self, open_scale=1.0):
        # adjusts both fingers. 0==closed, 1==open
        assert 0.0 <= open_scale <= 1.0, 'open_scale is out of range'
        for joint_id in self.finger_joint_ids:
            self._bullet_client.resetJointState(self.body_id, joint_id, self.finger_open_distance * open_scale)

    def configure_gripper_friction(self, lateral_friction=1.0, spinning_friction=1.0, rolling_friction=0.0001,
                                   friction_anchor=True):
        # configures the friction properties of all gripper joints
        for joint in self.finger_joint_ids:
            self._bullet_client.changeDynamics(self.body_id, joint,
                                               lateralFriction=lateral_friction, spinningFriction=spinning_friction,
                                               rollingFriction=rolling_friction, frictionAnchor=friction_anchor)

    def gripper_constraints(self):
        # first finger joint is "leader", second finger joint simply mirrors the position
        pos = self._bullet_client.getJointState(self.body_id, self.finger_joint_ids[0])[0]
        self._bullet_client.setJointMotorControl2(
            self.body_id,
            self.finger_joint_ids[1],
            self._bullet_client.POSITION_CONTROL,
            targetPosition=pos,
            force=self.finger_force,
            targetVelocity=2 * self.grasp_speed,
            positionGain=1.8
        )
        return pos

    def open(self, open_scale=1.0):
        # opens the gripper, blocks for 2 seconds
        joint_ids = [i for i in self.finger_joint_ids]
        target_states = [self.finger_open_distance * open_scale, self.finger_open_distance * open_scale]

        self._bullet_client.setJointMotorControlArray(
            self.body_id,
            joint_ids,
            self._bullet_client.POSITION_CONTROL,
            targetPositions=target_states,
            forces=[self.finger_force] * len(joint_ids)
        )
        self._simulator.step(seconds=2)

    def close(self):
        # closes the gripper, blocks for 2 seconds
        self._bullet_client.setJointMotorControl2(
            self.body_id,
            self.finger_joint_ids[0],
            self._bullet_client.VELOCITY_CONTROL,
            targetVelocity=-self.grasp_speed,
            force=self.finger_force,
        )
        self._simulator.step(seconds=2)

    def joint_pos(self):
        # super().joint_pos() returns all joints, including fixed. this one only returns movable joints.
        joint_pos = super().joint_pos()
        movable_joints = self.arm_joint_ids + self.finger_joint_ids
        return joint_pos[movable_joints]

    def arm_joints_pos(self):
        return super().joint_pos()[self.arm_joint_ids]

    def arm_joint_limits(self):
        # includes the platform as well (if present)
        # will return n, 2 array with [min, max] for each joint
        joint_limits = np.empty(shape=(len(self.arm_joint_ids), 2))
        i = 0
        for key, joint_info in self.robot_joints.items():
            if joint_info['id'] in self.arm_joint_ids:
                joint_limits[i, 0] = joint_info['lower_limit']
                joint_limits[i, 1] = joint_info['upper limit']
                i += 1

        assert i == len(self.arm_joint_ids), 'joint limits should be equal to number of joints'
        return joint_limits

    def move_to(self, target_joint_config):
        # given target joint values for the arm, robot moves there, blocks until target reached

        # make a joint trajectory
        trajectory = TrajectoryPlanner.ptp(self.arm_joints_pos(), target_joint_config)

        # execute a joint trajectory
        self.execute_joint_trajectory(trajectory)

    def set_target_pos_and_vel(self, target_joint_pos, target_joint_vel):
        """ set target joint positions and target velocities for velocity control """
        joints = self.arm_joint_ids
        self._simulator.bullet_client.setJointMotorControlArray(
            self.body_id,
            joints,
            self._simulator.bullet_client.VELOCITY_CONTROL,
            targetPositions=list(target_joint_pos),
            targetVelocities=list(target_joint_vel),
            forces=[500]*len(joints),
        )

    def execute_joint_trajectory(self, joint_trajectory):
        """
        Executes the commands from a JointTrajectory.

        :param joint_trajectory: JointTrajectory object

        :return: bool, True if all joints arrived at target configuration
        """
        start_time = self._simulator.simulated_seconds
        for time_step, dt, target_pos, target_vel in joint_trajectory:
            # set intermediate target
            self.set_target_pos_and_vel(target_pos, target_vel)

            # simulate until we reach next timestep
            step_end_time = start_time + time_step + dt
            while self._simulator.simulated_seconds < step_end_time:
                self._simulator.step()
            _log.debug(f'expected vs. actual joint pos after time step\n\t{target_pos}\n\t{self.arm_joints_pos()}')

        # should have arrived at the final target now, check if true
        arrived = joint_trajectory.joint_pos[-1] - self.arm_joints_pos() < 0.001
        if np.all(arrived):
            _log.debug('finished trajectory execution. arrived at goal position.')
            return True
        else:
            _log.warning(f'trajectory execution terminated but target configuration not reached:'
                         f'\n\tjoint pos diff is: {joint_trajectory.joint_pos[-1] - self.arm_joints_pos()}'
                         f'\n\tend-effector pos is: {self.end_effector_pos()}')
            return False

    def in_self_collision(self):
        """
        checks if the robot is in collision with itself
        :return: True if in collision
        """
        # ignore links: w/platform: [0, 1, 10, 14]; w/o platform: [7, 11] - they do not have a collision shape.
        # as first link, we only need to consider up to panda_link7 (6/9), after that there's only the hand which is
        # supposed to be in contact with the fingers.
        # disable collisions for link0, link1 (with platform), link8 (7/10), grasp_target (11/14)
        if self.with_platform:
            max_link = 14
            ignore_links = [0, 1, 10, 14]
            first_links = [2, 3, 4, 5, 6, 7, 8]
        else:
            max_link = 11
            ignore_links = [7, 11]
            first_links = [1, 2, 3, 4, 5]  # 6 cannot collide with the fingers due to kinematics

        for first_link in first_links:
            # skip next link (supposed to be in contact) plus all the ignore links
            check_links = [link for link in np.arange(first_link+2, max_link+1) if link not in ignore_links]
            for check_link in check_links:
                collision = self._simulator.links_in_collision(self.body_id, first_link, self.body_id, check_link)
                if collision:
                    return True
        return False

    def in_collision(self):
        """
        checks if the robot is in collision with the scene
        does not check for self collisions
        :return:  True if in collision
        """
        return self._simulator.body_in_collision(self.body_id)


class TrajectoryPlanner:
    def lin(self):
        pass

    @staticmethod
    def ptp(start_q, target_q, v_max=0.7, a_max=1.5, dt=1./20):
        # will create trapezoidal velocity profile for each joint by using waypoints
        # synchronised ptp: joint that needs to do the largest distance will determine the trajectory length
        # for all other joints we will reduce v and a accordingly
        # see TrajectoryPlanner in BURG Toolkit (for LIN motion)
        assert len(target_q) == len(start_q)
        if np.allclose(start_q, target_q):  # already at target
            time_steps = np.asarray([0, dt])
            waypoints = np.asarray([start_q, target_q])
            return burg.robots.JointTrajectory(time_steps, waypoints)

        n_joints = len(start_q)

        distances = np.abs(target_q-start_q)
        max_dist = np.max(distances)

        if max_dist < v_max**2/a_max:
            # the distance is too short for the trapezoidal profile when using max vel and acc
            # reduce velocity to ensure the trapezoidal form (at least full acceleration/deceleration phase, no top)
            # acceleration phase: t_a = v/a
            # corresponding distance in both phases = 2 * 1/2 * a * t_a**2 = a * (v/a)**2 = v**2/a
            v_max = np.sqrt(max_dist*a_max)

        trajectory_time = (max_dist * a_max + v_max ** 2) / (v_max * a_max)
        trajectory_steps = int(trajectory_time // dt) + 1

        # determine v and a for every joint, based on trajectory time and distance
        # assuming a trapezoidal form, we can see that 2d > vT > 1d
        # hence we can choose v = 1.5 d/T
        v = np.minimum(1.5 * distances / trajectory_time, v_max)

        # having chosen v, d, T, we can determine a as a = v**2 / (vT - d)
        a = v**2 / (v*trajectory_time - distances)

        time_steps = np.linspace(0, trajectory_time, trajectory_steps)
        directions = (target_q - start_q) / np.abs(target_q - start_q)
        waypoints = np.zeros(shape=(trajectory_steps, n_joints))

        # compute waypoints
        for i, t in enumerate(time_steps):
            # separately for each joint, as they might be in different phases of the trapezoid
            for j in range(n_joints):
                if t <= v[j] / a[j]:  # acceleration period
                    distance_from_start = 1 / 2 * a[j] * t ** 2
                elif t <= trajectory_time - v[j] / a[j]:  # max velocity period
                    distance_from_start = v[j] * t - v[j] ** 2 / (2 * a[j])
                else:  # deceleration period
                    distance_from_start = (2 * a[j] * v[j] * trajectory_time - 2 * v[j] ** 2 - a[j] ** 2 * (
                                t - trajectory_time) ** 2) / (2 * a[j])
                waypoints[i][j] = start_q[j] + directions[j] * distance_from_start

        assert np.allclose(waypoints[-1], target_q), f'waypoint interpolation went wrong, target pose is' \
                                                     f'{target_q} but last waypoint is {waypoints[-1]}'

        trajectory = burg.robots.JointTrajectory(time_steps, waypoints)
        return trajectory
