import math
import numpy as np
import pybullet
import quaternion
import burg_toolkit as burg
import time


SCENARIO_DIR = 'scenarios'
SCENARIO_IDS = [11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35, 41, 42, 43, 44, 45]


class Timer:
    """
    this is a class to conveniently measure timings and additional count occurrences
    once instantiated, you can use timer.start('key') and timer.stop('key') to measure time, if you do it repeatedly
    it will sum up the elapsed time between all start and stop calls.
    with timer.count('key') you can count occurrences.
    finally, timer.print() will provide a summary of all stats.
    """
    def __init__(self):
        self.timers = {}
        self.counters = {}

    def start(self, key):
        if key not in self.timers.keys():
            self.timers[key] = -time.time()
        else:
            self.timers[key] -= time.time()

    def stop(self, key):
        if key not in self.timers.keys():
            raise ValueError('attempting to stop timer that has not been started')
        self.timers[key] += time.time()

    def count(self, key):
        if key not in self.counters.keys():
            self.counters[key] = 1
        else:
            self.counters[key] += 1

    def print(self):
        print('************ TIMINGS ************')
        for key, val in self.timers.items():
            print(f'\t{key}:\t{val:.2f}s')
        print('*********** COUNTERS ************')
        for key, val in self.counters.items():
            print(f'\t{key}:\t{val}x')
        print('*********************************')


class Color:
    GREEN = [0, 1, 0, 1]
    RED = [1, 0, 0, 1]
    BLUE = [0, 0, 1, 1]


def make_sphere(sim, pose, color):
    """
    adds a sphere to Simulator sim, for visualization purposes

    :param sim: simulation.Simulator
    :param pose: (4, 4) transformation matrix
    :param color: list of 4 values, [r, g, b, a], with a=1 for full opacity
    """
    radius = 0.02
    sphere_collision_id = sim._p.createVisualShape(
        sim._p.GEOM_SPHERE, radius=radius, rgbaColor=color
    )

    pos, orn = burg.util.position_and_quaternion_from_tf(pose, convention='pybullet')

    body_id = sim._p.createMultiBody(
        baseMass=0, baseVisualShapeIndex=sphere_collision_id,
        basePosition=pos, baseOrientation=orn
    )


def angle_between_quaternions(q1, q2, as_degree=False):
    """
    calculates the angle in radian between two quaternions that have pybullet (xyzw) convention.
    """
    diff_quat = pybullet.getDifferenceQuaternion(q1, q2)
    angle = 2 * np.arccos(np.clip(np.abs(diff_quat[3]), 0, 1))

    if as_degree:
        angle = np.rad2deg(angle)
    return angle


def quaternion_from_rotation_matrix(rot_mat):
    q = quaternion.from_rotation_matrix(rot_mat)
    return np.asarray([q.x, q.y, q.z, q.w])


def get_translation_and_angle(pos1, orn1, pos2, orn2) -> tuple[float, float]:
    """
    Calculates distance in task space, gives translation in [m] and rotation in degree
    :param pos1: [x, y, z] position1
    :param orn1: [x, y, z, w] quaternion1
    :param pos2: [x, y, z] position2
    :param orn2: [x, y, z, w] quaternion2
    :return: tuple(translation, angle in degree)
    """
    pos_dist = np.linalg.norm(pos2-pos1)
    angle = angle_between_quaternions(orn1, orn2, as_degree=True)
    return pos_dist, angle


def get_fake_grasp(pos=None):
    """produces a mock grasp based on the given pose, used for debugging"""
    if pos is None:
        pos = [1, 0.5, 0.2]
    pose = np.eye(4)
    pose[:3, 3] = pos
    grasp = burg.Grasp()
    grasp.pose = pose
    return grasp


