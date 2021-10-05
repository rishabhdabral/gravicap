# GraviCap
Official code repository for ICCV 2021 paper: Gravity-Aware Monocular 3D Human Object Reconstruction

We propose GraviCap - a new approach for joint 3D human-object reconstruction under gravity constraints. Given the 2D trajectory of an object and intrinsics, we recover the 3D trajectory of the object and the  human in absolute metric units along with the  camera tilt. The trajectories are recovered in absolute metric units while also estimating the camera tilt. We also release a new dataset with human and object annotations as a benchmark. This repository comes with a subset of dataset for demo purposes. Please download the full dataset from <a href="https://4dqv.mpi-inf.mpg.de/GraviCap/">here</a>.

## Installation
Install the requirements using:
```
pip install -p requirements.txt
```

## Quick Start
 Demo on S1
```
python exp_pose_optim.py --calib_path ./data/calibration_hps.pkl --pose_path ./data/VNect_Results/EDGAR_10_BALLS_17/ --cam 17 --eps 0 --gt_pose_path ./data/captury/EDGAR_10_BALLS/EDGAR_10_BALLS.mddd --annot_dir ./data/EDGAR_10_BALLS/annotations/
# Use the --cam and --eps flags to set the camera and episode ids.

```

Running on a custom video 
```
python exp_pose_optim.py --calib_path ./data/calibration_hps.pkl --pose_path ./data/VNect_Results/EDGAR_10_BALLS_17/ --cam 17 --eps 0 --gt_pose_path ./data/captury/EDGAR_10_BALLS/EDGAR_10_BALLS.mddd --annot_dir ./data/EDGAR_10_BALLS/annotations/
```
  
## References
```
@inproceedings{GraviCap2021, 
    author = {Dabral, Rishabh and Shimada, Soshi and Jain, Arjun and Theobalt, Christian and Golyanik, Vladislav}, 
    title = {Gravity-Aware Monocular 3D Human-Object Reconstruction}, 
    booktitle = {International Conference on Computer Vision (ICCV)}, 
    year = {2021} 
}    	
	
```
