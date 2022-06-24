## 1 Introduction

 At present, [LegoDNN](https://dl.acm.org/doi/abs/10.1145/3447993.3483249) includes six kinds of widely used visual DNN applications, including image classification, semantic segmentation, object detection, action recognition, anomaly detection and pose estimation. The DNNs in all visual applications contain a large number of convolution layers and blocks.  LegoDNN is a lightweight, block-grained and scalable solution for running multi-DNN wrokloads in mobile vision systems. It extracts the blocks of original models via convolutional layers, generates sparse blocks, and retrains the sparse blocks. By composing these blocks, LegoDNN expands the scaling options of original models. At runtime, it optimizes the block selection process using optimization algorithms. 

This project provided a command line tool for blocks in LegoDNN and implemented a general block extraction algorithm, supporting the automatic block extraction of the models in image classification, target detection, semantic segmentation, attitude estimation, behavior recognition, and anomaly detection applications.



## 2 Installation

 **Prepare environment**

1. Create a conda virtual environment and activate it.

   ```
   conda create -n legodnn python=3.6
   conda active legodnn
   ```

2. Install PyTorch and torchvision according the [official site](https://github.com/LINC-BIT/IoT-and-Edge-Intelligence)
   ![image](https://user-images.githubusercontent.com/73862727/146364503-5664de5b-24b1-4a85-b342-3d061cd7563f.png)
   Get install params according to the selection in the official site,and copy them to the terminal.

   **Note: please determine whether the CPU version of pytorch or GPU version is installed. If the CPU version of pytorch is installed, please change the `device ='cuda'`in the following code to `device ='cpu'`**

3. Install legodnn-cli

   ```shell
   pip install legodnn-cli
   ```



## 3 Start

- Use the `--help` option to view help for using this cli.

- Five parameters are required for automatic block extraction.

  1. The `-p/--path ` parameter specifies the path to save the original model file. Here, the original model  exported by the `torch.save()` function.
  2. The `-r/--ratio` parameter specifies the block partition ratio, which is the upper limit of the number of compressible layers in each block.
  3. The `-s/--shape ` parameter specifies the input shape of the model. Numbers should be separated by `,` such as `3,32,32`.
  4. The `-o/--output ` parameter specifies the path to save the block file. By default, it is saved in the same path as the original model.
  5. The `-d/--device` parameter specifies the device to which the block file is saved. The default value is `cuda`.

- An example of automated block extraction using this tool is

  ```
  legodnn extract -p path/to/the/original/model -r 0.125 -s 3,32,32 -o path/to/save/blocks -d cuda
  ```



