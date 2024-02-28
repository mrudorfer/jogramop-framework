# jogramop-framework
A framework for joint grasp and motion planning in confined spaces

### content
The framework contains models of the environments, robots, prepared graps, motion planners and example evaluation scripts

Scenarios
- are stored in scenarios/X, where X is the name of the scenario
- scene.yaml contains description of the scenarios, filenames pointing to various obstacles and objects
- grasps.npy - set of grasping points stored as numpy array
- export/grasps_IK_solutions.csv - contains joint configuration for each IK solution that was found (one configuration per line, each configuration is a list of robot joint angles in radians)
- export/grasps.csv - same as grasps.npy, but in csv format
- export/obstacles.obj - the whole environment stored in .obj format (you can visualize it e.g. using Blender or Meshlab)
- export/robot_start_conf.csv - joint configuration for the "home" position of the robot

Robots 
- robots are described by URDF format e.g. robots/frank_panda/mobile_panda_fingers.urdf
- URDF defines links of the robot and their joints and collision/visualization geometries
- visualization and collision meshes are in robots/frank_panda/meshes

Objects
- common 3D objects used across the scenarios. Individual scenarios refer to them in their yaml files


### motion planners

Motion planners are at a separate repository: https://github.com/Hartvi/Burs



### installation

```commandline
conda create -n jogramop python=3.10
conda activate jogramop
pip install git+https://github.com/mrudorfer/burg-toolkit@dev
```

I got a cdktree/numpy bug after `import burg_toolkit`, which was resolved by reinstalling numpy and numba.

There may be some issues when installing on linux.. will test to make platform-independent.

### robots & scenarios

We use the franka panda robot, and you can start it up `with_platform=True` to get it mounted on a mobile platform
that can move with two linear axes. This is to mimic a mobile manipulator. Without the platform, the base is just
fixed on the ground.

There are two scenarios:
- `sugarbox_free.yaml`: The sugarbox with no obstacles.
- `sugarbox_verticalgap_20cm.yaml`: Sugarbox in same pose, but vertical walls with a gap.

### main.py

The `main.py` represents the whole pipeline (but is work in progress). It should:
- load the scene including its objects
- sample a bunch of grasps using an antipodal grasp sampler
- for each grasp:
  - get joint pose with inverse kinematics (no collision checks atm)
  - move ptp from home pose to gripper pose (ignoring obstacles)
  - close gripper
  - move ptp back to home pose (hopefully with object grasped)

However, it got a bit messy as I started to implement my own inverse kinematics.

### rrt.py

The `rrt.py` just focuses on the motion planning part. It loads a scenario and tries to make the robot move to 
some arbitrary pose in task space (a fake grasp).
Core features:
- numpy-ndarray-based tree that uses a vectorised exhaustive nearest neighbour search
- extend-randomly as usual
- extend-to-goal uses pseudo-inverse Jacobian to move towards a target end-effector pose in task space
- for both methods, I limit the step size by defining a maximum step for each joint
  - this is the variable `step_size_q`, which is relative to normalised joint range (0-1).
  - ensures that steps taken by extend-randomly and extend-to-goal should be of similar magnitude
- the number of steps taken in each iteration can be adjusted (e.g. 5 steps towards `q_rand`)

When you run it, it starts planning. After, it shows the waypoints in pybullet and you can step through them by hitting
enter.

### ideas / plan

- Inverse Kinematics (IK)
  - pybullet's standard IK is limited, as obstacles are not taken into account, and it suffers when dealing with too many DoF (such as for mobile manipulator)
  - in theory, IK can take into account obstacles, esp. with mobile manipulator
  - might need to implement it ourselves, unless we find a workaround that does not require IK
- Motion Planning (MP)
  - assuming IK for grasp is solved and collision-free, we need MP to find a collision-free path
  - might have high number of DoF with mobile manipulator (9)
  - for the retreat, we can fix object to gripper and consider it in the collision func
- approach-based grasp sampling
  - instead of random selection of (pre-sampled) grasps, do something more intelligent, e.g.:
    - sample/filter grasps based on the current pose of the end-effector

### todo

- add 2 prismatic DoF for base of robot to simulate "mobile" manipulator
  - added in `mobile_panda.urdf`; however, the pybullet IK does not seem to be able to cope with the two axes
  - enabled switching between those two, using argument `with_platform=True`
  - there are some works on IK for mobile manipulators, e.g. M-FABRIK
  - might need to implement it on my own



- check grasp x-axis and swap in case of bad alignment
- pre-sample set of grasps
- integrate collision function
- pre-grasp poses
  - typically like 5cm on top of grasp pose, to approach object linearly
  - requires to plan linear trajectory from grasp pose to pre-grasp pose
  - they typically simplify planning, as MP would only need to get to pre-grasp pose (likely collision-free)
  - we may be able to lift this constraint
  - however, we cannot use synchronised ptp to get there, as the gripper may flip and then bump into object
