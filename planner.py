import numpy as np
import burg_toolkit as burg

from scenario import Scenario
from rrt import RRT, Jplus_RRT
from rrt_multitarget import MultiTarget_Jplus_RRT
import util
import visualization


class NaiveRRTPlanner:
    """
    The Naive Planner works as follows:
    - while grasps available:
        - randomly select one grasp from the set (with removal)
        - perform IK to get a joint configuration
        - if joint configuration colliding, continue with next iteration
        - use planner to get to target joint configuration
        - if success:
            - execute grasp
            - return
    - no plans found for any of the grasps
    """
    def __init__(self, seed=None, planner_time=30):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.dist_threshold = 0.005  # 5 mm
        self.angle_threshold = 5  # 5 degree
        self.planner_time = planner_time
        self.timer = util.Timer()

    def plan(self, scenario):
        self.timer.start('plan')
        grasps = scenario.grasp_poses
        robot, sim = scenario.get_robot_and_sim(with_gui=False)

        indices = self.rng.permutation(len(grasps))
        for i in indices:
            self.timer.count('iterations')
            target_ee_pose = grasps[i].pose

            # perform inverse kinematics
            self.timer.start('ik')
            position, orientation = burg.util.position_and_quaternion_from_tf(target_ee_pose, convention='pybullet')
            robot.reset_arm_joints()  # reset to home position
            joint_positions = sim.bullet_client.calculateInverseKinematics(
                robot.body_id, robot.end_effector_link_id, position, orientation,
                maxNumIterations=100, residualThreshold=0.001)
            target_conf = robot.get_arm_joint_conf_from_motor_joint_conf(joint_positions)
            self.timer.stop('ik')

            # make sure the IK worked
            # check robot reaches the target_ee_pose with reasonable accuracy
            pos, ori = robot.forward_kinematics(target_conf)
            dist, angle = util.get_translation_and_angle(position, orientation, pos, ori)
            if dist > self.dist_threshold or angle > self.angle_threshold:
                self.timer.count('IK did not reach target')
                continue

            # check it's collision-free
            if robot.in_self_collision() or robot.in_collision():
                self.timer.count('IK solution in collision')

                visualization.visualize_waypoints(
                    [target_conf],
                    scene_fn=scenario.scene_fn,
                    target_pose=target_ee_pose,
                    robot_base_pose=scenario.robot_pose,
                    step_time=0,
                    repeat=False
                )
                continue

            # target configuration is valid; plan a motion to get there!
            planner = RRT(
                start_config=robot.home_conf,
                target_config=target_conf,
                robot=robot,
                seed=self.seed,
                max_iter=50000,
                max_time=self.planner_time
            )
            self.timer.start('RRT')
            success, waypoints = planner.plan()
            self.timer.stop('RRT')
            if success:
                self.timer.stop('plan')
                return success, waypoints
            self.timer.count('RRT did not find solution')

        # all grasps considered, but no plans found
        self.timer.stop('plan')
        return False, None


class NaiveJplusRRTPlanner:
    """
    The Naive J+RRT Planner works as follows:
    - while grasps available:
        - randomly select one grasp from the set (with removal)
        - use J+RRT to get to target pose
        - if success:
            - execute grasp
            - return
    - no plans found for any of the grasps
    """
    def __init__(self, seed=None, planner_time=30):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.timer = util.Timer()
        self.planner_time = planner_time

    def plan(self, scenario):
        self.timer.start('plan')
        grasps = scenario.grasp_poses
        robot, sim = scenario.get_robot_and_sim(with_gui=False)

        indices = self.rng.permutation(len(grasps))
        for i in indices:
            self.timer.count('iterations')
            target_ee_pose = grasps[i].pose

            # initialise RRT planner
            pos, orn = burg.util.position_and_quaternion_from_tf(target_ee_pose, convention='pybullet')
            rrt = Jplus_RRT(robot.home_conf, pos, orn, robot, seed=self.seed,
                            max_time=self.planner_time,
                            max_iter=50000,
                            step_size_q=0.02
                            )
            success, waypoints = rrt.plan()
            if success:
                self.timer.stop('plan')
                return success, waypoints
            self.timer.count('RRT did not find solution')

        # all grasps considered, but no plans found
        self.timer.stop('plan')
        return False, None


class MultiTargetJplusRRTPlanner:
    """
    The MultiTarget J+RRT Planner only plans once, uses RRT that extends to all possible grasp targets
    """
    def __init__(self, seed=None, planner_time=300):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.timer = util.Timer()
        self.planner_time = planner_time

    def plan(self, scenario):
        self.timer.start('plan')
        grasps = scenario.grasp_poses
        robot, sim = scenario.get_robot_and_sim(with_gui=False)

        rrt = MultiTarget_Jplus_RRT(robot.home_conf, grasps, robot, seed=self.seed,
                                    max_time=self.planner_time,
                                    max_iter=100000,
                                    step_size_q=0.02,
                                    goal_prob=1,
                                    pick_goals_uniformly=False,
                                    dist_thresh_for_orientation=500
                                    )
        success, waypoints = rrt.plan()
        self.timer.stop('plan')

        return success, waypoints


def main():
    seed = 3
    scenario = Scenario(1, with_platform=False)
    #scenario.select_n_grasps(n=20, seed=seed)
    planner = MultiTargetJplusRRTPlanner(seed=seed, planner_time=3)  # seems overwhelmed with 9 DoF... IK solutions are a bit stupid
    success, waypoints = planner.plan(scenario)
    print('planner success:', success)
    planner.timer.print()

    print()
    print(waypoints)
    print()

    visualization.visualize_waypoints(
        scenario,
        waypoints,
        target_pose=None,
    )


if __name__ == '__main__':
    main()
