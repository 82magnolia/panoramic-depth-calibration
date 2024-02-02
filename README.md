# Panoramic Depth Calibration
Official PyTorch implementation of **Calibrating Panoramic Depth Estimation for Practical Localization and Mapping (ICCV 2023)** [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Kim_Calibrating_Panoramic_Depth_Estimation_for_Practical_Localization_and_Mapping_ICCV_2023_paper.html) [[Video]](https://www.youtube.com/watch?v=KXz8IwrtJWg).

[<img src="calib_overview.png" width="600"/>](calib_overview.png)

Our method *calibrates* a pre-trained panoramic depth estimation network to new, unseen domains using test-time adaptation.
The resulting network can be used for downstream tasks such as visual navigation or map-free localization.
Below we show a qualitative sample, where our adaptation scheme leads to largely improved depth predictions amidst salt-and-pepper noise.

[<img src="adaptation_sample.png" width="600"/>](adaptation_sample.png)

In this repository, we provide the implementation and instructions for running our calibration method. If you have any questions regarding the implementation, please leave an issue or contact 82magnolia@snu.ac.kr.

## Installation
First setup a conda environment.
```
conda create -n pytorch3d python=3.9
conda activate pytorch3d
```
Then, follow the instructions from [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#installing-prebuilt-binaries-for-pytorch3d) to install the library. Here, please use the following to install PyTorch3D after all the other dependencies are installed.
```
conda install pytorch3d -c pytorch3d
```
Then, install other dependencies with `pip install -r requirements.txt`.

## Installation (CUDA >= 11.3)
For GPUs supporting CUDA versions greater than `11.3` (e.g., RTX3090), installation is much more straightforward. Run the following sequence of commands.
```
conda create -n pytorch3d python=3.9
conda activate pytorch3d
conda install -c pytorch pytorch=1.11.0 torchvision cudatoolkit=11.3
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
pip install -r requirements.txt
```
