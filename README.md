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


### scenario.py
- script to show the scenarios, robots and graspin points
> python3.10 scenario.py




