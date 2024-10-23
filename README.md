# GAP-RL

Official code of paper "GAP-RL: Grasps As Points for RL Towards Dynamic Object Grasping".

# Installation
This code has been tested on Ubuntu20.04 with Cuda 11.7, Python3.9 and Pytorch 1.13.1.
Environment: create a conda environment with necessary packages specified by `requirements.txt`.
Run the following code to generate the seperate `gap_rl` package:
```shell
pip install -e .
```

# Examples
You can run the following code to visualize the trajectories 
({'random2d', 'line', 'circular', 'bezier2d'} as {'Rotation', 'Line', 'Circular', 'Random'} in the paper):
```shell :
cd gap_rl/examples
python test_env_trajs.py
```
