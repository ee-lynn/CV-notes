# recognize objects in 3D spaces

 &nbsp;     　　　　　　　　　　   sqlu@zju.edu.cn

## 引言

本文主要聚焦三维物体识别的概念，方法，但考虑到其中一些方法的预备知识，还会补充少量三维物体分类，语义分割的基本方法和概念。在三维空间中，物体框将基于图片的物体识别问题扩展，物体框具有中心坐标tx,ty,tz,尺寸参数h,w,l,另外一般还含有鸟瞰图中物体框朝向角的参数\theta. (忽略了朝平行地面坐标轴的旋转角度)。在信息采集阶段可以采用雷达直接获取物体点云的方式，也可以采用多目相机获取物体深度图[双目立体视觉内容可参见img_proc.md]的方式。

根据三维物体信息表示方法的不同，识别方法大致可以分为三大类
（1）将立体物体三维网格化（voxel）
（2）将立体物体表示为多视角的图片
（3）以三维裸数据表示

## 将立体物体三维网格化的方法
 - Song, Shuran , and J. Xiao . "Deep Sliding Shapes for Amodal 3D Object Detection in RGB-D Images." CVPR 2016
- 
 - Yin Zhou, Oncel Tuzel. “VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection" CVPR 2018

## 将立体物体表示为多视角的图片

- Bo Li, Tianlei Zhang, Tian Xia， Vehicle Detection from 3D Lidar Using Fully Convolutional Network，IROS 2016

- Xiaozhi Chen, Huimin Ma, Ji Wan, Bo Li, Tian Xia “Multi-View 3D Object Detection Network for Autonomous Driving”，CVPR 2017

- Ku, Jason , et al. "Joint 3D Proposal Generation and Object Detection from View Aggregation." arXiv:1712.02294v4 .

- J Deng，K Czarnecki，“MLOD: A multi-view 3D object detection based on robust feature fusion method ”，arXiv 1909.04163v1
## 以三维裸数据表示

### 预备知识

- pointnet family
  
Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas. "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation”  CVPR 2017
Qi, Charles R., et al. "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space." NeurIPS 2017.
- gcn
- Kipf, Thomas N. , and M. Welling . "Semi-Supervised Classification with Graph Convolutional Networks." ICLR 2017.
Guohao Li, Matthias Müller, Ali Thabet, Bernard Ghanem.DeepGCNs: Can GCNs Go as Deep as CNNs? ICCV 2019
Y. Wang, Y. Sun, Z. Liu, S. E. Sarma, M. M. Bronstein, and J. M. Solomon. Dynamic graph cnn for learning on point
clouds. arXiv preprint arXiv:1801.07829, 2018
### Frustum PointNets
Qi, Charles R. , et al. "Frustum PointNets for 3D Object Detection from RGB-D Data." arXiv:1711.08488v2.

### PointRCNN
Shi, Shaoshuai , X. Wang , and H. Li . "PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud." CVPR 2019.

Yang, Zetong , et al. "IPOD: Intensive Point-based Object Detector for Point Cloud." arXiv:1812.05276.
