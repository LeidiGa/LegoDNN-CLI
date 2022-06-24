## 简介

[English](README.md) | 简体中文

 目前使用比较广泛的主要有六种视觉类DNN应用，包括**图像分类、语义分割、目标检测、行为识别、异常检测和姿态估计**。在这六种视觉DNN应用中均包含了大量的卷积层。LegoDNN（[文章](https://dl.acm.org/doi/abs/10.1145/3447993.3483249)）是一个针对模型缩放问题的轻量级、块粒度、可伸缩的解决方案，根据卷积层从原始DNN模型中抽取块，生成稀疏派生块，然后对这些块进行再训练。通过组合这些块，扩大原始模型的伸缩选项。并且在运行时，通过算法对块的选择进行了优化。

 本项目为LegoDNN中的块提供了一个命令行工具，实现了通用的自动化块抽取算法，对于图像分类、目标检测、语义分割、姿态估计、行为识别、异常检测等类型的模型均可以通过算法，自动找出其中的块用于再训练。



## 安装

- **安装流程**

  1. 使用conda新建虚拟环境，并进入该虚拟环境

     ```
     conda create -n legodnn python=3.6
     conda active legodnn
     ```

  2. 根据[Pytorch官网](https://github.com/LINC-BIT/IoT-and-Edge-Intelligence)安装Pytorch和torchvision
     ![image](https://user-images.githubusercontent.com/73862727/146364503-5664de5b-24b1-4a85-b342-3d061cd7563f.png)
     根据官网选择要安装的Pytorch对应的参数，然后复制相应的命令在终端输入即可

     **注意请确定安装的是CPU版本Pytorch还是GPU版本，如果是CPU版本的Pytorch请将下面代码中的`device='cuda'`改为`device='cpu'`**

  3. 安装legodnn-cli命令行工具


  ```shell
  pip install legodnn-cli
  ```

  

## 开始使用

- 使用`--help`选项查看命令行工具的使用帮助。

- 自动进行块抽取需要提供5个参数。

  1. `-p/--path`参数指定保存原始模型文件的路径，这里的原始模型是使用`torch.save()`函数导出的。
  2. `-r/--ratio`参数指定块划分比例，也就是每个块中可压缩层数的上限。
  3. `-s/--shape`参数指定模型输入形状，图像为三维，视频为四维。数字应以`,`分隔，如`3,32,32`。
  4. `-o/--output`参数指定保存块文件的路径，默认与原始模型保存在相同路径下。
  5. `-d/--device`参数指定块文件所保存到的设备，默认为`cuda`。

- 使用本工具进行自动化块抽取的一个示例为

  ```
  legodnn extract -p path/to/the/original/model -r 0.125 -s 3,32,32 -o path/to/save/blocks -d cuda
  ```



