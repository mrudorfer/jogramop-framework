# plan:
# create environment as BURG scene
# one target object, several obstacle objects
# use antipodal grasp sampler to get some grasps of the object (w/ collision detect)
# use planner to find trajectory to a grasp pose
# if unsuccessful, use next grasp
# until either success, or all grasps considered
import math

import burg_toolkit as burg
import numpy as np
import matplotlib.pyplot as plt

import simulation
from util import get_fake_grasp, angle_between_quaternions, make_sphere

# burg.visualization.configure_visualizer_mode(burg.visualization.VisualizerMode.CONFIGURABLE_VIEWER)


def sample_grasps(scene, visualize=True):
    """
    uses an antipodal grasp sampler to come up with a bunch of candidate grasps.
    performs simple collision checking against an abstract gripper model and the other objects in the scene.

    returns a burg.GraspSet with the sampled, collision-free grasp candidates
    """

    target_object = scene.objects[0]
    gripper = burg.gripper.TwoFingerGripperVisualisation()
    sampler = burg.sampling.AntipodalGraspSampler()

    candidate_grasps, _ = sampler.sample(target_object, n=100)
    print(f'sampled {len(candidate_grasps)} grasps.')
    if visualize:
        burg.visualization.show_grasp_set([scene], candidate_grasps, gripper=gripper)

    # check collisions with scene as well
    collisions = sampler.check_collisions(candidate_grasps, scene, gripper_mesh=gripper.mesh)
    collision_free_grasps = candidate_grasps[collisions == 0]
    print(f'collision-free grasps: {len(collision_free_grasps)}.')
    if visualize:
        candidate_grasps.scores = collisions
        burg.visualization.show_grasp_set([scene], candidate_grasps, gripper=gripper,
                                          score_color_func=lambda s: [s, 1-s, 0])

    return collision_free_grasps


def forward_kinematics(robot, joint_conf, as_matrix=False):
    """
    gets the end effector pose for the given robot in the given arm joint angles.
    returns position and quaternion, unless as_matrix=True, then it will return 4x4 matrix

    caution: resets the robots joints. do not use in active simulation.
    """
    if len(joint_conf) == len(robot.arm_joint_ids):
        arm_joint_conf = joint_conf
    elif len(joint_conf) == len(robot.arm_joint_ids) + len(robot.finger_joint_ids):
        arm_joint_conf = robot.get_arm_joint_conf_from_motor_joint_conf(joint_conf)
    elif len(joint_conf) == robot.end_effector_link_id:
        arm_joint_conf = np.asarray(joint_conf)[robot.arm_joint_ids]
    else:
        raise ValueError('cannot match joint conf to arm/motor/all joints; unexpected length.')

    # set robot joint angles
    robot.reset_arm_joints(arm_joint_conf)
    # retrieve EE pose
    pos, quat, *_ = robot.bullet_client.getLinkState(
        robot.body_id,
        robot.end_effector_link_id,
        computeForwardKinematics=True
    )
    if as_matrix:
        return burg.util.tf_from_pos_quat(pos, quat, convention='pybullet')
    return np.asarray(pos), np.asarray(quat)


def do_inverse_kinematics_stuff(visualize=True):
    # set up simulators and robots
    robot_base_pose = np.eye(4)

    sim_gui = simulation.GraspingSimulator(verbose=True)
    robot_gui = simulation.FrankaRobot(sim_gui, robot_base_pose, with_platform=True)

    sim_direct = simulation.GraspingSimulator(verbose=False)  # for IK and resetting states
    robot_direct = simulation.FrankaRobot(sim_direct, robot_base_pose, with_platform=True)

    # set up target
    target_pos = [0.2, 0.4, 0.4]
    target_ee_pose = robot_gui.get_ee_pose_for_grasp(get_fake_grasp(pos=target_pos))
    green = [0, 1, 0, 1]
    red = [1, 0, 0, 1]
    blue = [0, 0, 1, 1]
    make_sphere(sim_gui, target_ee_pose, green)

    # do the pybullet IK
    target_pos, target_orn = burg.util.position_and_quaternion_from_tf(target_ee_pose, convention='pybullet')
    joint_positions = sim_gui.bullet_client.calculateInverseKinematics(
        robot_gui.body_id, robot_gui.end_effector_link_id, target_pos, # target_orn,
        maxNumIterations=100, residualThreshold=0.001)

    # ik finds solution including all movable joints (incl. gripper), we are only interested in arm joints
    print(f'IK found {len(joint_positions)} joint positions: {joint_positions}')
    arm_joint_targets = robot_gui.get_arm_joint_conf_from_motor_joint_conf(joint_positions)
    print(f'got {len(arm_joint_targets)} arm joint targets: {arm_joint_targets}')

    # check that IK found a good solution
    actual_pos, actual_orn = forward_kinematics(robot_direct, arm_joint_targets)
    print(f'IK vs. target pos difference: {np.linalg.norm(target_pos - actual_pos)*1000:.03} mm')
    print(f'IK vs. target orn difference: {angle_between_quaternions(target_orn, actual_orn, as_degree=True):.03} degree')

    # visualise solution
    robot_gui.reset_arm_joints(arm_joint_targets)
    make_sphere(sim_gui, robot_gui.end_effector_pose(), red)

    robot_gui.reset_arm_joints()
    input('press enter to start with our own IK')

    # do own IK with direct robot:
    # reset robot to init pose
    # calculate error: target_pos - fkin(joint_pos)
    # set new joint pos: joint_pos + pinv(jacobian) * error
    # until error small enough
    robot_direct.reset_home_pose()
    cur_joint_pos = robot_direct.joint_pos()
    home_joint_pos = robot_direct.joint_pos()

    epsilon = 0.001
    it = 0
    max_iter = 1000
    errors = []
    while it < max_iter:
        # calculate position delta and error
        current_pos, current_orn = forward_kinematics(robot_direct, cur_joint_pos)
        delta_pos = target_pos - current_pos
        pos_error = np.linalg.norm(delta_pos)

        # calculate rotation error
        q_diff = sim_direct.bullet_client.getDifferenceQuaternion(target_orn, current_orn)
        delta_orn = sim_direct.bullet_client.getEulerFromQuaternion(q_diff)
        orn_error = 2 * math.acos(np.abs(q_diff[3]))

        print('euler:', delta_orn)
        print('combined angle:', orn_error)

        print(f'{it}:\n\tdelta_pos: {delta_pos}; error: {pos_error}\n\tdelta_orn: {delta_orn}; error: {orn_error}')
        print(f'\t{cur_joint_pos}')
        error = pos_error + orn_error
        # error = orn_error
        errors.append(error)
        if error < epsilon:
            print(f'ending IK after {it} updates. combined error is {error}.')
            break

        # calculate jacobian and its inverse
        zero_vec = [0.0] * len(cur_joint_pos)
        link_com = [0.0, 0.0, 0.0]
        jac_t, jac_r = sim_direct.bullet_client.calculateJacobian(
            robot_direct.body_id, robot_direct.end_effector_link_id,
            link_com, list(cur_joint_pos), zero_vec, zero_vec)

        jac_t, jac_r = np.asarray(jac_t), np.asarray(jac_r)
        pinv_jac_t, pinv_jac_r = np.linalg.pinv(jac_t), np.linalg.pinv(jac_r)

        if np.linalg.matrix_rank(pinv_jac_t) < 3 or np.linalg.matrix_rank(pinv_jac_r) < 3:
            print('pseudo-inverse Jacobian does not have full rank')
            print('rank pinv jac_t', np.linalg.matrix_rank(pinv_jac_t))
            print('rank pinv jac_r', np.linalg.matrix_rank(pinv_jac_r))

        # make update
        cur_joint_pos += pinv_jac_t @ delta_pos
        cur_joint_pos += -pinv_jac_r @ delta_orn

        # for null-space control, we need the combined Jacobian matrix full_jac
        # null_space_mat will then ensure that every control vector is mapped into the nullspace of combined J
        full_jac = np.vstack([jac_t, jac_r])
        full_jac_pinv = np.linalg.pinv(full_jac)
        null_space_mat = np.eye(full_jac.shape[1]) - full_jac_pinv @ full_jac

        # define null-space controller
        # currently only proof-of-concept controller that keeps the base at a desired position
        # TODO: need to ensure range of joints, by repelling the end configurations in null-space controller
        # TODO: need to avoid obstacles, also repelling in joint space... how to do this?
        target_base_pos = [0.5, 0]
        u_null_space = np.zeros(len(cur_joint_pos))
        u_null_space[0] = cur_joint_pos[0] - target_base_pos[0]  # only control first two axes
        u_null_space[1] = cur_joint_pos[1] - target_base_pos[1]  # only control first two axes
        cur_joint_pos += -5 * null_space_mat @ u_null_space
        it += 1

    print('visualising IK result')
    arm_joints = robot_direct.get_arm_joint_conf_from_motor_joint_conf(cur_joint_pos)
    robot_gui.reset_arm_joints(arm_joints)
    make_sphere(sim_gui, robot_gui.end_effector_pose(), blue)

    plt.plot(errors)
    plt.show()

    input('press any key to exit')
    input()

    # get rid of simulators
    sim_gui.dismiss()
    sim_direct.dismiss()


def simulate_grasp(scene, grasp, visualize=False):
    if visualize:
        burg.visualization.show_grasp_set([scene], grasp, gripper=burg.gripper.TwoFingerGripperVisualisation())

    sim = simulation.GraspingSimulator(verbose=visualize)
    sim.add_scene(scene)
    pose = np.asarray([
        0, -1, 0, 1,
        1, 0, 0, 0.4,
        0, 0, 1, 0,
        0, 0, 0, 1
    ]).reshape(4, 4)
    robot = simulation.FrankaRobot(sim, pose, with_platform=True)

    ee_target_pose = robot.get_ee_pose_for_grasp(grasp)

    position, orientation = burg.util.position_and_quaternion_from_tf(ee_target_pose, convention='pybullet')
    # Calculate the inverse kinematics to get joint configuration for the given pose
    # this can be inaccurate, but should give a configuration CLOSE to the grasp
    # for actual control, we will need to account for those inaccuracies using iterative control
    # see: https://github.com/bulletphysics/bullet3/issues/1380
    joint_positions = sim.bullet_client.calculateInverseKinematics(
        robot.body_id, robot.end_effector_link_id, position, orientation,
        maxNumIterations=100, residualThreshold=0.001)

    target_conf = robot.get_arm_joint_conf_from_motor_joint_conf(joint_positions)
    robot.move_to(target_conf)

    make_sphere(sim, grasp.pose, [1, 0, 0, 1])  # red: the grasp pose
    make_sphere(sim, robot.end_effector_pose(), [0, 0, 1, 1])  # blue: where we end up

    robot.close()
    robot.move_to(robot.home_conf)
    sim.dismiss()


def main():
    # do_inverse_kinematics_stuff()
    # quit()

    import rrt
    # rrt.test_self_collisions()
    rrt.test_rrt2()
    quit()

    scene_fn = 'scenes/sugarbox_free.yaml'
    scene, object_library, _ = burg.Scene.from_yaml(scene_fn)
    # grasps = sample_grasps(scene, visualize=False)
    # todo: temporary - to speed things up
    grasps = [get_fake_grasp()]

    # choose a random order to go through the grasps
    indices = np.arange(len(grasps))
    np.random.shuffle(indices)

    for idx in indices:
        grasp = grasps[int(idx)]
        simulate_grasp(scene, grasp, visualize=True)


if __name__ == '__main__':
    main()
