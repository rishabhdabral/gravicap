# GraviCap
Official code repository for ICCV 2021 paper: <a href="https://openaccess.thecvf.com/content/ICCV2021/html/Dabral_Gravity-Aware_Monocular_3D_Human-Object_Reconstruction_ICCV_2021_paper.html">Gravity-Aware Monocular 3D Human Object Reconstruction</a>.

<p align="center">
  <img src="https://4dqv.mpi-inf.mpg.de/static/GraviCap_ICCV2021.gif" alt="animated" />
</p>

We propose GraviCap - a new approach for joint 3D human-object reconstruction under gravity constraints. Given the 2D trajectory of an object and intrinsics, we recover the 3D trajectory of the object and the  human in absolute metric units along with the  camera tilt. The trajectories are recovered in absolute metric units while also estimating the camera tilt. We also release a new dataset with human and object annotations as a benchmark. This repository comes with a subset of the dataset for demo purposes. Please download the full dataset from the <a href="https://4dqv.mpi-inf.mpg.de/GraviCap/">project page</a>.

## Installation
```
conda create --name gravicap python=3.6
git clone https://github.com/rishabhdabral/gravicap.git
pip install scipy
```
For running with VNect outputs, you may need to get access to VNect by registering <a href="https://vcai.mpi-inf.mpg.de/projects/VNect/">here</a>.

## Quick Start
 Demo on S6: We provide processed data of Sequence 6 from the Gravicap dataset. 
```
python main.py --calib_path ./data/calibration_hps.pkl --annot_dir ./data/ --cam 17 --eps 0 --gt_pose_path ./data/S6.mddd 
# Use the --cam and --eps flags to set the camera and episode ids.
```
If VNect outputs for are available:
```
python main.py --calib_path ./data/calibration_hps.pkl --annot_dir ./data/ --cam 17 --eps 0 --gt_pose_path ./data/S6.mddd --mode vnect --pose_path path/to/vnect/output/directory
# Use the --cam and --eps flags to set the camera and episode ids.
```
  
## License
Permission is hereby granted, free of charge, to any person or company obtaining a copy of this software and associated documentation files (the "Software") from the copyright holders to use the Software for any non-commercial purpose. Publication, redistribution and (re)selling of the software, of modifications, extensions, and derivates of it, and of other software containing portions of the licensed Software, are not permitted. The Copyright holder is permitted to publically disclose and advertise the use of the software by any licensee.

Packaging or distributing parts or whole of the provided software (including code, models and data) as is or as part of other software is prohibited. Commercial use of parts or whole of the provided software (including code, models and data) is strictly prohibited. Using the provided software for promotion of a commercial entity or product, or in any other manner which directly or indirectly results in commercial gains is strictly prohibited.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## References
```
@inproceedings{GraviCap2021, 
    author = {Dabral, Rishabh and Shimada, Soshi and Jain, Arjun and Theobalt, Christian and Golyanik, Vladislav}, 
    title = {Gravity-Aware Monocular 3D Human-Object Reconstruction}, 
    booktitle = {International Conference on Computer Vision (ICCV)}, 
    year = {2021} 
}    	
	
```
