# jogramop-framework
A framework for joint grasp and motion planning in confined spaces.

This is part of the publication:

M. Rudorfer, J. Hartvich, V. Vonasek: "A Framework for Joint Grasp and Motion Planning in Confined Spaces", 13th International Workshop on Robot Motion and Control (RoMoCo'24), 2024.

Check out our project website: https://mrudorfer.github.io/jogramop-framework/

## Content

The content of the publication is split into two repositories.
This repository contains the scenarios used for benchmarking, along with scripts for creation and pre-processing of all parts of the scenarios (in case you would like to re-create them, or use the pipeline to create your own scenarios). 
The sister repository at https://github.com/Hartvi/jogramop-planners contains the baseline planners.
The scenarios from this repository can be exported, and those files are used by the other repository.

### Scenarios 

Scenarios are stored in `./scenarios` directory.
Each scenario is contained in a subdirectory based on its ID `scenarios/<scenario-id>/*`.
The folder contains:

- `scene.yaml` contains description of the environment, references the objects and obstacles and defines their poses.
- `grasps.npy` is a set of 200 grasp candidates stored as end-effector poses in a numpy array.
- The `export` directory contains all information related to the  scenario and can be used by the planners implemented in C++ in the sister repository. Note that the base coordinate system is different: in the export, the robot is in the origin, hence grasps and obstacles in the export are transformed compared to the previous ones; furthermore, all obstacles are merged into one .obj file. The export folder contains:
  - `export/grasps.csv` - same as grasps.npy, but in csv format (and transformed)
  - `export/obstacles.obj` - the whole environment stored in .obj format (you can visualize it e.g. using Blender or Meshlab)
  - `export/robot_start_conf.csv` - joint configuration for the "home" position of the robot (currently same for all scenarios)
  - `export/grasps_IK_solutions.csv` - contains joint configuration for each inverse kinematics solution that was found (one configuration per line, each configuration is a list of robot joint angles in radians). We exclude IK solutions that are in collision with the environment, hence this list will have fewer than 200 solutions.
  
### Robots 
- Robots are described by URDF format e.g. `robots/frank_panda/mobile_panda_fingers.urdf`
- URDF defines links of the robot and their joints and collision/visualization geometries
- Visualization and collision meshes are in `robots/frank_panda/meshes`.
- We simplified the meshes for improved performance.

### Objects
- We use simple shapes to create obstacles (see `create_obstacle_mesh.py`).
- We use objects from the [YCB dataset](https://www.ycbbenchmarks.com/object-models/) as target objects for grasping.
- We store all objects in `./object_library/`, where additional files and data have been created (VHACD, URDF, stable resting poses) using the [BURG toolkit](https://mrudorfer.github.io/burg-toolkit/).

This repository already contains all scenarios, grasps etc. pre-computed.
Furthermore, we provide the scripts that we used to create those scenarios, so that you can create further scenarios,
obstacles, grasp poses, robots, etc.
See below for instructions on how to use them.

## Motion planners

The motion planners are in a separate repository: https://github.com/Hartvi/jogramop-planners

They work purely based on the exported files for each scenario and otherwise do not require this repository as dependency.

## Installation

```commandline
conda create -n jogramop python=3.10
conda activate jogramop
pip install git+https://github.com/mrudorfer/burg-toolkit@dev
```

Potential issue:
If a cdktree/numpy bug comes up after `import burg_toolkit`, we were able to resolve it by reinstalling numpy and numba.

Please get in touch (create an issue) if you encounter any issues during installation.

## Visualizing scenarios

You can visualize a scenario using the following code:

```python
from util import SCENARIO_IDS
from scenario import Scenario
from visualization import show_scenario

for i in SCENARIO_IDS:
    print(f'********** SCENARIO {i:03d} **********')
    s = Scenario(i)
    
    # select a random subset of grasps for better visualization
    s.select_n_grasps(10)
    
    show_scenario(s)
```

This repository comes with full robot simulation functionality based on pyBullet, see module `simulation.py`.
You can also use it directly to implement planners for these scenarios, see:
```python
from scenario import Scenario

s = Scenario(11)
robot, sim = s.get_robot_and_sim(with_gui=True)
```

Finally, once we have a planning result, we can visualize it as well.
A plan is essentially a list of joint configurations which represent the waypoints when moving from start to goal.
```python
from visualization import visualize_waypoints

visualize_waypoints(scenario, list_of_waypoints)
```

## Recreation of scenarios

As mentioned earlier, the scenario data provided in this repository is completed.
You do not need to recreate the scenarios unless you would like to modify them.

We created the data by running the scripts in the following order:
```cmd
# 1. create simple shapes used as obstacles
python create_obstacle_mesh.py

# 2. create VHACD, URDF, stable resting poses and organize in object library
# you may need to include the YCB objects manually
python create_object_library.py

# 3. create the scenarios, will create ./scenarios/<scenario-id>
python create_scenario.py

# 4. sample grasps for each of the scenarios
python create_graspset.py

# 5. create ik solutions and other export files for cpp planners
python create_cpp_export.py
```

Note that we manually used `simplify` from [this](https://github.com/hjwdzh/Manifold) repository (after step 2) to ensure the VHACD meshes are as small as possible without sacrificing geometric accuracy. 
This is not necessary but improves runtime of collision checking.
