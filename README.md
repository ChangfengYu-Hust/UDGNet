# UDGNet

### Unsupervised Image Deraining: Optimization Model Driven Deep CNN [Paper](https://owuchangyuo.github.io/files/UDGNet.pdf)
By Changfeng Yu*, Yi Chang* (https://scholar.google.com/citations?user=I1nZ67YAAAAJ&hl=en)(* indicates equal contribution)

##Demo
![Demo](/results/real_result.png)

## Dependencies and Installation

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.2](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install -r enviroment.txt`

#Dataset
- Our dataset RealRain(including the angle) and Rain_cityscape can be downloaded here. 
- Make your own data:
	1. Simulating by yourself, the angle information can be easily obtained during the simulation.
	2. Other sythetic datasets or real image dataset, you need to label the angle informations by ./lib/angle_label.py
- the dataset should have the following structure:
	```c++
	-train
 	-rain/data
 	-angle/data
 	-clean/data
	```  
	```c++
	-test
 	-rain/data
 	-angle/data
	```  

## How to Train
- **UDGNet**
	1. Run command:
	```c++
	python train_Decomposition_angle.py --rain_path ./dataset/test/rain --angle_path ./data/test/angle --clean_path ./data/test/rain --reset 1
	```

## How to Test
- **UDGNet**
	1. Run command:
	```c++
	python Test_Decomposition_angle.py --rain_path ./dataset/test/rain --angle_path ./data/test/angle --clean_path ./data/test/rain --weight_path ./output/real_model/generator_backup.pth
	```





# UDGNet
