import os
import time

import numpy as np
import cv2
import burg_toolkit as burg

import simulation
from scenario import Scenario


class ImageRenderer:
    def __init__(self, bullet_client=None, camera_pose=None):
        # parameters that can be adjusted
        self.zfar = 10
        self.znear = 0.02

        # pybullet stuff
        self.bullet_client = bullet_client
        if camera_pose is None:
            camera_pose = burg.util.look_at(position=[2.7, -1.5, 1.5], target=[1, 0.3, 0.1], flip=True)
        self.camera_pose = camera_pose

        # compute projection matrix from kinect-like camera parameters
        camera = burg.render.Camera.create_kinect_like()
        self.w, self.h = camera.resolution
        cx, cy = camera.intrinsic_parameters['cx'], camera.intrinsic_parameters['cy']
        fx, fy = camera.intrinsic_parameters['fx'], camera.intrinsic_parameters['fy']
        self._projection_matrix = np.array([
            [2 * fx / self.w, 0, 0, 0],
            [0, 2 * fy / self.h, 0, 0],
            [-(2 * cx / self.w - 1), 2 * cy / self.h - 1, (self.znear + self.zfar) / (self.znear - self.zfar), -1],
            [0, 0, (2 * self.znear * self.zfar) / (self.znear - self.zfar), 0]
        ])
        self._ambient_light = [1., 1., 1.]

    def render_rgb(self, bullet_client=None, camera_pose=None):
        if bullet_client is None:
            bullet_client = self.bullet_client
            if bullet_client is None:
                raise ValueError('need to provide bullet client')
        if camera_pose is None:
            camera_pose = self.camera_pose
            if camera_pose is None:
                raise ValueError('need to provide camera pose')

        view_matrix = np.linalg.inv(camera_pose).T
        w, h, rgb, depth, seg_mask = bullet_client.getCameraImage(
            self.w, self.h,
            viewMatrix=view_matrix.flatten(),
            projectionMatrix=self._projection_matrix.flatten()
        )

        # returned images might be tuples
        rgb = np.reshape(rgb, (h, w, 4))
        rgb = rgb[:, :, :3]  # remove alpha
        rgb = rgb.astype(np.uint8)

        return rgb


class FourImageRenderer:
    # renders four different views and combines them in one image, rather than just one
    def __init__(self, scenario):
        cam_poses = []
        if scenario.id == 1:
            cam1 = burg.util.look_at(position=[2.7, -1.5, 1.5], target=[1, 0.3, 0.1], flip=True)
            cam2 = burg.util.look_at(position=[2.1, 1.5, 1.5], target=[1, 0.3, 0.1], flip=True)
            cam3 = burg.util.look_at(position=[1.5, 0.75, 0.2], target=[1, 0.7, 0.1], flip=True)
            cam4 = burg.util.look_at(position=[1.0, 1.3, 0.2], target=[1, 0.7, 0.1], flip=True)
        elif scenario.id == 2:
            cam1 = burg.util.look_at(position=[2.7, -1.5, 1.5], target=[1, 0.3, 0.1], flip=True)
            cam2 = burg.util.look_at(position=[-0.2, -1.1, 1.5], target=[1, 0.3, 0.1], flip=True)
            cam3 = burg.util.look_at(position=[1.15, 0.5, 0.8], target=[1, 0.7, 0.5], flip=True)
            cam4 = burg.util.look_at(position=[0.9, 1.4, 0.8], target=[1, 0.7, 0.5], flip=True)
        elif scenario.id == 3:
            cam1 = burg.util.look_at(position=[2.7, -1.5, 1.5], target=[1, 0.3, 0.1], flip=True)
            cam2 = burg.util.look_at(position=[-0.7, -1.5, 1.5], target=[1, 0.3, 0.1], flip=True)
            cam3 = burg.util.look_at(position=[1.7, 0.5, 0.3], target=[1, 0.8, 0.1], flip=True)
            cam4 = burg.util.look_at(position=[0.4, 1.1, 0.2], target=[1, 0.75, 0.1], flip=True)
        else:
            cam1 = burg.util.look_at(position=[2.7, -1.5, 1.5], target=[1, 0.3, 0.1], flip=True)
            cam2 = burg.util.look_at(position=[1.8, 2.0, 1.2], target=[1, 0.3, 0.1], flip=True)
            cam3 = burg.util.look_at(position=[0.2, 2.0, 0.8], target=[1, 0.7, 0.5], flip=True)
            cam4 = burg.util.look_at(position=[1.2, 1.8, 0.8], target=[1, 0.7, 0.5], flip=True)

        self.renderers = [ImageRenderer(camera_pose=c) for c in [cam1, cam2, cam3, cam4]]
        self.w = self.renderers[0].w * 2
        self.h = self.renderers[0].h * 2

    def render_rgb(self, bullet_client):
        imgs = []
        for r in self.renderers:
            imgs.append(r.render_rgb(bullet_client))

        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        img[:self.h//2, :self.w//2] = imgs[0]
        img[self.h//2:, :self.w//2] = imgs[1]
        img[:self.h//2, self.w//2:] = imgs[2]
        img[self.h//2:, self.w//2:] = imgs[3]
        return img


def indicate_workspace(sim, color=None):
    # just for visualization purposes in pybullet
    if color is None:
        color = [0.666, 0.666, 0.666, 1]
    if len(color) == 3:
        color = list(color) + [1]
    plane_id = sim.bullet_client.createVisualShape(
        sim.bullet_client.GEOM_BOX, halfExtents=[1, 1, 0.001], rgbaColor=color
    )

    pos = [1, 0, 0]
    body_id = sim.bullet_client.createMultiBody(
        baseMass=0, baseVisualShapeIndex=plane_id, basePosition=pos
    )


def visualize_waypoints(scenario, list_of_waypoints, target_pose=None, step_time=0.5,
                        repeat=True):
    """
    visualises a given list of waypoints for the franka robot with platform.
    you can either step through manually, or autoplay it with `step_time`, and even repeat the loop.

    :param scenario: a scenario object, which loads the scene file and gives robot and simulation
    :param list_of_waypoints: either nested list, or ndarray (n_waypoints, n_dof)
    :param target_pose: ndarray (4, 4) with target pose, or None
    :param step_time: float, seconds to show each waypoint; with None or 0 you need to step through manually
    :param repeat: if True, will loop the visualization
    :return:
    """
    robot, sim = scenario.get_robot_and_sim(with_gui=True)
    indicate_workspace(sim)

    if target_pose is None:
        target_pose = scenario.grasp_poses

    if len(target_pose.shape) == 2:
        # assume shape is (4, 4)
        sim.add_frame(tf=target_pose)
    elif len(target_pose.shape) == 3:
        # assume shape is (n, 4, 4)
        for tf in target_pose:
            sim.add_frame(tf=tf)
    else:
        print('WARNING: cannot interpret target pose... not visualizing it.')

    sim.add_frame([0, 0, 0])

    first_run = True
    fkin_frame_ids = []
    # create frames
    prevPos = None
    for i in range(len(list_of_waypoints)):
        pos, orn = robot.forward_kinematics(list_of_waypoints[i])
#        fkin_frame_ids.append(sim.add_frame(pos=pos, orn=orn))
        if i > 0:
            d = sum([ (prevPos[i] - pos[i])**2 for i in range(3) ])**(1/2) #distance between two consecutive waypoints
            steps = int(d / 0.01) + 1
            interpolation = [ prevPos*(1-i/steps) + (pos*i/steps) for i in range(steps) ]
            print(interpolation)
            for ipos in interpolation:
                fkin_frame_ids.append(sim.add_sphere(pos=ipos, orn=orn, scale=0.01))
            
        prevPos = pos


    while first_run or repeat:
        i = 0
        first_run = False

        while i < len(list_of_waypoints):
            robot.reset_arm_joints(list_of_waypoints[i])
            # pos, orn = robot.forward_kinematics(list_of_waypoints[i])
            # fkin_frame_ids.append(sim.add_frame(pos=pos, orn=orn))

            # user interface
            if step_time is None or step_time <= 0:
                # manual mode
                print('current waypoint:', list_of_waypoints[i])
                key = input(f'{i+1}/{len(list_of_waypoints)}: enter to proceed, p for previous, q to quit')
                if key == 'q':
                    repeat = False
                    break
                if key == 'p':
                    i = max(i-2, 0)
            else:
                # auto stepping; no controls
                time.sleep(step_time)
            i += 1

        # while fkin_frame_ids:
        #     sim.remove(fkin_frame_ids.pop())

    sim.dismiss()
    print('bye bye.')


def create_video(scenario, waypoints, video_fn, renderer, target_pose=None, fps=3):
    robot, sim = scenario.get_robot_and_sim(with_gui=False)
    sim.bullet_client.changeVisualShape(
        sim.look_up_body_id(scenario.scene.objects[0]), -1, rgbaColor=[0.75, 0.75, 0.25, 1.0])
    indicate_workspace(sim, color=[0.5, 0.8, 0.5])

    if target_pose is not None:
        sim.add_frame(tf=target_pose)

    video_writer = cv2.VideoWriter(f'{video_fn}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (renderer.w, renderer.h))

    for i in range(len(waypoints)):
        # robot.reset_arm_joints(list_of_waypoints[i])
        pos, orn = robot.forward_kinematics(waypoints[i])  # automatically resets arm
        fkin_frame_id = sim.add_frame(pos=pos, orn=orn)

        # make picture
        img_rgb = renderer.render_rgb(sim.bullet_client)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        video_writer.write(img)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)

        sim.remove(fkin_frame_id)

    sim.dismiss()


def show_scenario(scenario):
    robot, sim = scenario.get_robot_and_sim(with_gui=False)
    home_conf = robot.home_conf
    sim.dismiss()
    list_of_waypoints = [home_conf]
    visualize_waypoints(scenario, list_of_waypoints, step_time=0, repeat=False)


def test_vis(scenario, trajFile):
    waypoints = [
        [ 0.        ,  0.        , -0.01779206, -0.76012354,  0.01978261, -2.34205014,  0.02984053,  1.54119353,  0.75344866],
        [ 0.04      ,  0.04      ,  0.07543812, -0.72490447,  0.07638708, -2.37448407,  0.06908042,  1.54488921,  0.67831372],
        [ 0.08      ,  0.08      ,  0.1657339 , -0.6865332 ,  0.12951995, -2.40180711,  0.10798703,  1.55301725,  0.60104267],
        [ 0.03348037, -0.13874396,  0.2844179 , -0.6132292 ,  0.24820395, -2.33897511,  0.22667103,  1.63120925, -1.23003653],
        [ 0.03348037, -0.13874396,  0.4031019 , -0.5399252 ,  0.36688795, -2.27614311,  0.34535503,  1.70940125, -1.23003653],
        [ 0.03348037, -0.13874396,  0.51508008, -0.4666212 ,  0.48557195, -2.24875362,  0.46403903,  1.78759325, -1.23003653],
        [ 0.03348037, -0.13874396,  0.51508008, -0.3933172 ,  0.60425595, -2.24875362,  0.58272303,  1.86578525, -1.23003653],
        [ 0.03348037, -0.13874396,  0.51508008, -0.3200132 ,  0.72293995, -2.24875362,  0.70140703,  1.94397725, -1.23003653],
        [ 0.07348037, -0.22830728,  0.63376408, -0.2467092 ,  0.84162395, -2.18592162,  0.82009103,  2.02216925, -1.11135253],
        [ 0.11348037, -0.22830728,  0.74118352, -0.1734052 ,  0.96030795, -2.12308962,  0.93877503,  2.10036125, -0.99266853],
        [ 0.15348037, -0.22830728,  0.74118352, -0.1001012 ,  1.07899195, -2.06025762,  1.05745903,  2.17855325, -0.87398453],
        [ 0.19348037, -0.22830728,  0.74118352, -0.0267972 ,  1.19767595, -1.99742562,  1.17614303,  2.25674525, -0.75530053],
        [ 0.21450446, -0.18830728,  0.18224764, -1.16490337,  1.31635995, -1.93459362,  1.12311681,  2.33493725, -1.06778578],
        [ 0.21450446, -0.14830728,  0.18224764, -1.16490337,  1.43504395, -1.87176162,  1.12311681,  2.36596726, -1.06778578],
        [ 0.21450446, -0.10830728,  0.18224764, -1.16490337,  1.55372795, -1.80892962,  1.12311681,  2.36596726, -1.06778578],
        [ 0.21450446, -0.06830728,  0.18224764, -1.16490337,  1.67241195, -1.74609762,  1.12311681,  2.36596726, -1.06778578],
        [ 0.21450446, -0.02830728,  0.18224764, -1.16490337,  1.79109595, -1.68326562,  1.12311681,  2.36596726, -1.06778578],
        [ 0.25450446, -0.1576753 ,  0.30093164, -1.35041388,  1.50455529, -1.62043362,  0.50948168,  2.44415926, -0.94910178],
        [ 0.29450446, -0.1576753 ,  0.41961564, -1.35041388,  1.50455529, -1.55760162,  0.50948168,  2.52235126, -0.83041778],
        [ 0.33450446, -0.1576753 ,  0.53829964, -1.35041388,  1.50455529, -1.49476962,  0.50948168,  2.60054326, -0.71173378],
        [ 0.37450446, -0.1576753 ,  0.65698364, -1.35041388,  1.50455529, -1.43193762,  0.50948168,  2.67873526, -0.59304978],
        [ 0.41450446, -0.1576753 ,  0.77566764, -1.35041388,  1.50455529, -1.36910562,  0.50948168,  2.75692726, -0.47436578],
        [ 0.45450446, -0.1176753 ,  0.46978508, -1.27710988,  1.62323929, -1.30627362,  0.62816568,  2.56588813, -0.35568178],
        [ 0.49450446, -0.0776753 ,  0.46978508, -1.20380588,  1.74192329, -1.26018591,  0.74684968,  2.56588813, -0.23699778],
        [ 0.53450446, -0.31055444,  0.58846908, -1.13050188,  1.86060729, -1.19735391, -0.01269374,  1.9406962 , -0.27375675],
        [ 0.57450446, -0.3453008 , -1.4687618 , -1.05719788,  1.86690121, -1.37613417, -0.45294696,  2.0188882 , -0.18192514],
        [ 0.61450446, -0.3453008 , -1.4687618 , -0.98389388,  1.86690121, -1.37613417, -0.45294696,  2.0970802 , -0.18192514],
        [ 0.65450446, -0.3453008 , -1.4687618 , -0.91058988,  1.86690121, -1.37613417, -0.45294696,  2.1752722 , -0.18192514],
        [ 0.69450446, -0.3453008 , -1.4687618 , -0.83728588,  1.86690121, -1.37613417, -0.45294696,  2.2534642 , -0.18192514],
        [ 0.72563236, -0.3453008 , -1.4687618 , -0.76398188,  1.86690121, -1.37613417, -0.45294696,  2.3316562 , -0.18192514],
        [ 0.76563236, -0.3053008 , -1.3500778 , -1.24592425,  1.45504207, -1.92796248, -0.33426296,  2.4098482 , -0.06324114],
        [ 0.80563236, -0.2653008 , -1.2313938 , -1.24592425,  1.45504207, -1.92796248, -0.21557896,  2.4880402 ,  0.05544286],
        [ 0.84563236, -0.2253008 , -1.22741608, -1.24592425,  1.45504207, -1.92796248, -0.09689496,  2.5662322 ,  0.17412686],
        [ 0.88563236, -0.1853008 , -1.22741608, -1.24592425,  1.45504207, -1.92796248, -0.04868062,  2.6444242 ,  0.29281086],
        [ 0.92563236, -0.16862102, -1.22741608, -1.24592425,  1.45504207, -1.92796248, -0.04868062,  2.7226162 ,  0.41149486],
        [ 0.88362817, -0.12862102, -1.66833447, -1.17262025,  1.57372607, -2.16886715, -0.56370871,  2.8008082 ,  0.53017886],
        [ 0.88362817, -0.08862102, -1.66833447, -1.09931625,  1.60165799, -2.16886715, -0.56370871,  2.8790002 ,  0.64886286],
        [ 0.88362817, -0.04862102, -1.66833447, -1.02601225,  1.60165799, -2.16886715, -0.56370871,  2.9571922 ,  0.76754686],
        [ 0.88362817, -0.00862102, -1.66833447, -0.95270825,  1.60165799, -2.16886715, -0.56370871,  3.0353842 ,  0.88623086],
        [ 0.88362817,  0.03137898, -1.66833447, -0.87940425,  1.60165799, -2.16886715, -0.56370871,  3.1135762 ,  1.00491486],
        [ 0.88413797,  0.07137898, -1.63189238, -0.89563487,  1.72034199, -2.22824166, -0.62438836,  3.1346112 ,  0.95150129],
        [ 0.8817875 ,  0.11137898, -1.62654789, -0.88103191,  1.83902599, -2.30894172, -0.66511573,  3.15319158,  0.91617005]
    ]
    
    target = np.eye(4)
    target[:3, 3] = [1.0, 1.0, 0.2]

    s = Scenario(21)
    visualize_waypoints(s, waypoints, step_time=0)


def show_scenarios():
    for i in range(4):
        s = Scenario(i + 1)
        show_scenario(s)


def read_waypoints_from_file(filename):
    waypoints = np.genfromtxt(filename, delimiter=',')
    print(f'read waypoints from file. shape: {waypoints.shape}')
    return waypoints


def batch_create_videos(directory):
    vid_dir = os.path.join(directory, 'videos')
    burg.io.make_sure_directory_exists(vid_dir)

    for file in os.listdir(directory):
        if not file.endswith('.csv'):
            continue
        filename = os.path.join(directory, file)
        print(f'creating video for {filename}')
        print('scenario id:', file[len('scenario'):len('scenario')+2])
        scenario_id = int(file[len('scenario'):len('scenario')+2])
        scenario = Scenario(scenario_id)
        waypoints = read_waypoints_from_file(filename)
        video_fn = os.path.join(vid_dir, file.replace('.csv', ''))
        create_video(scenario, waypoints, video_fn, renderer=FourImageRenderer(scenario), fps=2)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: ", sys.argv[0], " <scenario> <trajectory.try> <outprefix> ")
        quit()
    scenario = sys.argv[1]
    trajFile = sys.argv[2]
    prefix = sys.argv[3]
    test_vis(scenario, trajFile)
    # show_scenarios()

    # batch_create_videos('../Burs/blind_trajectories/')
