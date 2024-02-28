# jogramop-framework
A framework for joint grasp and motion planning in confined spaces

### content
The framework contains models of the environments, robots, prepared graps, motion planners and example evaluation scripts

Scenarios
- are stored in scenarios/X, where X is the name of the scenario
- scene.yaml contains description of the scenarios, filenames pointing to various obstacles and objects
- grasps.npy - set of grasp poses stored as numpy array
- export directory contains all information related to this scenario and can be used by the planners implemented in C++ (see below) -- note that the base coordinate system is different: in the export, the robot is in the origin, hence grasps and obstacles in the export are transformed compared to the previous ones
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

This repository already contains all scenarios, grasps etc. pre-computed.
Furthermore, we provide the scripts that we used to create those scenarios, so that you can create further scenarios,
obstacles, grasp poses, robots, etc. 

### motion planners

Motion planners are at a separate repository: https://github.com/Hartvi/Burs

They work purely based on the exported files for each scenario and otherwise do not require this repository as dependency.

### installation

```commandline
conda create -n jogramop python=3.10
conda activate jogramop
pip install git+https://github.com/mrudorfer/burg-toolkit@dev
```

Potential issue:
If a cdktree/numpy bug comes up after `import burg_toolkit`, it can be resolved by reinstalling numpy and numba.


### recreation of scenarios

todo -- organise scripts and write instructions