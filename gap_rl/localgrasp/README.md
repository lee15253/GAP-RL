# LocalGrasp Model
This is a compiled LocalGrasp package for grasp detection. 

You can use it standalone to detect grasps given the **point cloud** (keep the frame especially the rotation as the `scene_example.npz`) and the **centers** sampled from the point cloud. Grasps can be detected around the centers.

# Installation
You can refer to requirements.txt file to build a conda environment with python=3.9.
For some packages like pytorch3d and pointnet2_ops_lib, and you may install them following the official guidelines. 

# Visualization Examples
```shell
python test.py
```
