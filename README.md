# reproduce-OpenPose
a record of using OpenPose.  
***该项目主要记录一下首次使用OpenPose的流程。***  
# 运行OpenPose  
## 1. 下载源码  
&emsp;&emsp;从[OpenPose官网](https://github.com/CMU-Perceptual-Computing-Lab/openpose)下载源码。  
## 2. 编译源码  
&emsp;&emsp;该步骤需要在电脑上安装cmake，vs2015，cuda和cuDNN.  
&emsp;&emsp;详细步骤记录在了我之前训练Yolo-Fastest的流程中[YOLO-Fastest-on-a-no-gpu-windows-computer](https://github.com/Charlie839242/YOLO-Fastest-on-a-no-gpu-windows-computer).  
### 2.1 cmake编译  
&emsp;&emsp;在openpose-master目录下创建build文件夹用来保存编译结果。  
&emsp;&emsp;按照下图所示点击Configure和Generate。（该步编译会耗费一段时间，因为会自动下载模型文件，opencv等依赖）  
![image]()   
### 2.2 vs2015编译  
&emsp;&emsp;在openpose-master/build文件夹下，用vs2015打开ALL_BUILD.vcxproj。按照下图生成解决方案。  
![image]()  
### 2.3 移动相关文件夹  
&emsp;&emsp;将openpose-master/build/bin文件夹作为运行的位置。  
&emsp;&emsp;将openpose-master/build/x64/Release文件下的所有文件移动到openpose-master/build/bin下。  
&emsp;&emsp;将openpose-master/models文件夹移动到openpose-master/build/bin下。  
&emsp;&emsp;点击移动后的models文件夹中的getModels.bat。因为之前模型有可能没下完整。  

