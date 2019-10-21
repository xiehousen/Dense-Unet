# Dense-Unet
An idea of combining Densenet and Unet  
Network structure of Dense-Unet  

![image](https://github.com/xiehousen/Dense-Unet/blob/master/img/2.jpg)

Comparison of test results of Dense-Unet on medical data set(Isbi 2015 HeLa cells)  

|method|Vrand|Vinfo|
|:---|:---|:---|
|U-Net|0.5676|0.4883|
|FCN-8s|0.5122|0.2397|
|E-Net|0.5940|0.5120|
|Dense-Unet|0.6017|0.5735


# Setup
All code was developed and tested on Nvidia RTX2080Ti the following environment.

Python 2.7  
opencv3  
numpy  
tensorflow>=1.0  
keras>=1.4(Or newer)  
cuda>=8.0  
cudnn>=5.0  

The documentation is the data set used to test the code

# Quick Start
To train our model on the given dataset using:  

`python unet_xueguan_danboduan_densenet.py` 

