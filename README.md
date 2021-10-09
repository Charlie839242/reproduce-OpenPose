# reproduce-OpenPose
a record of using OpenPose.  
***该项目主要记录一下首次使用OpenPose的流程。***  
# 一. 直接运行OpenPose  
## 1. 下载源码  
&emsp;&emsp;从[OpenPose官网](https://github.com/CMU-Perceptual-Computing-Lab/openpose)下载源码。  
## 2. 编译源码  
&emsp;&emsp;该步骤需要在电脑上安装cmake，vs2015，cuda和cuDNN.  
&emsp;&emsp;详细步骤记录在了我之前训练Yolo-Fastest的流程中[YOLO-Fastest-on-a-no-gpu-windows-computer](https://github.com/Charlie839242/YOLO-Fastest-on-a-no-gpu-windows-computer).  
### 2.1 cmake编译  
&emsp;&emsp;在openpose-master目录下创建build文件夹用来保存编译结果。  
&emsp;&emsp;按照下图所示点击Configure和Generate。（该步编译会耗费一段时间，因为会自动下载模型文件，opencv等依赖）  
![image](https://github.com/Charlie839242/reproduce-OpenPose/img/cmake_0.jpg)  
![image](https://github.com/Charlie839242/reproduce-OpenPose/img/cmake_1.jpg)   
### 2.2 vs2015编译  
&emsp;&emsp;在openpose-master/build文件夹下，用vs2015打开ALL_BUILD.vcxproj。按照下图生成解决方案。  
![image](https://github.com/Charlie839242/reproduce-OpenPose/img/vs2015.jpg)  
### 2.3 移动相关文件夹  
&emsp;&emsp;将openpose-master/build/bin文件夹作为运行的位置。  
&emsp;&emsp;将openpose-master/build/x64/Release文件下的所有文件移动到openpose-master/build/bin下。  
&emsp;&emsp;将openpose-master/models文件夹移动到openpose-master/build/bin下。  
&emsp;&emsp;点击移动后的models文件夹中的getModels.bat。因为之前模型有可能没下完整。 
## 3. 运行  
&emsp;&emsp;运行指令  
```
OpenPoseDemo.exe --model_pose COCO --net_resolution 320x176  
后面两项是因为我的电脑配置不够，一运行就会出现out of memory的显卡显存报错，后两项可以减少使用的显存。
```
最后的工程在openpose文件夹下。 
# 二. 运行OpenPose-tf1.x版本  



