# recognize objects in 3D spaces

 &nbsp;     　　　　　　　　　　   sqlu@zju.edu.cn

## 引言

本文主要聚焦三维物体识别的概念，方法。在三维空间中，物体框将基于图片的物体识别问题扩展，物体框具有位置参数(如中心坐标$t_x,t_y,t_z$),尺寸参数(如长宽高$h,w,l$),另外一般还含有鸟瞰图中物体框朝向角的参数$\theta$(忽略了朝平行于地面坐标轴的旋转角度)。在信息获取阶段可以采用雷达直接获取物体点云的方式，也可以采用多目相机获取物体深度图间接获取点云的方式。

相比图片这种非常易于处理的网格化像素形式,点云具有无序性，不规则和不均匀的特点，需要将其表示为模型可以处理的输入形式。根据三维物体点云信息编码表示方法的不同，识别方法大致可以分为三大类

1) 将立体物体三维网格化(voxel)
2) 将立体物体表示为多视角的图片
3) 以三维裸数据表示

在物体框预测时，对于图片方案，anchor附着在feature map上每个像素，对于voxel方案，anchor附着在feature map的每个voxel,对于点云来说，anchor附着在每个点。

## 将立体物体三维网格化

### Deep sliding shapes

    Song, Shuran , and J. Xiao . "Deep Sliding Shapes for Amodal 3D Object Detection in RGB-D Images." CVPR 2016
  
- 该方法属于Faster RCNN在立体空间的自然推广，检测过程分成提案生成(proposal)和优化(refine)两个阶段。
- 点云编码：将整个场景分成三维网格，每个网格为网格中心到深度图表面最近点的距离向量（限幅到2倍网格宽度且带符号，称为Truncated Signed Distance Function,TSDF,实际使用时带上rgb颜色信息可以略微提高RPN性能）
- 第一阶段RPN：使用3D CNN作为RPN.为了解决多尺度问题，CNN出两支,浅层负责小anchor,深层负责大anchor，对点密度极小的anchor训练和推理阶段直接舍弃。后处理中3D NMS保留前2000个 proposal。
- 第二阶段ORN(Object Recognition Network): 将proposal稍向外pad（12.5%）然后以30x30x30分辨率网格化的空间编码,并且结合向图片的投影（VGG conv5_3后roi pooliing成7x7）两支输入预测物体框。直接将颜色编码进网格中效果不如基于图片的视角变换。
- 后处理：对尺寸，尺寸间比例指标落在训练集指标中心98%以外的预测框得分减半，可略微提升AP。
- Proposal 结构

      Voxel input
      Conv3d1/ReLU/Pooling
      Conv3d2/ReLU/Pooling
      Conv3d3/ReLU          Conv3d4/ReLU/Pooling
      Conv3d_cls+Conv3d_reg Conv3d_cls+Conv3d_reg
                             
- ORN结构
  
      3D proposal           image
      Conv3d1/ReLU/Pooling  VGG 
      Conv3d2/ReLU/Pooling  ROI Pooling(proposal projected)
      Conv3d3/ReLU/Pooling  FC(4096)
      FC(4096)
      Concat
      FC(1000)
      FC_cls+FC_reg

### VoxelNet

    Yin Zhou, Oncel Tuzel. “VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection" CVPR 2018

- 首先将点云按照三维网格划分，每个网格内用一个小pointnet进行特征提取并安排成网格状的特征量,然后用3D CNN进一步提取特征和融合，最后送入任务层
- 网格编码单元(Voxel feature encoding,VFE):本质上是一个用于分割的point,将网格内的点云按照坐标+特征+中心坐标表示，用共享的FC提取后maxpooling得到网格级别特征，然后concat点云各自特征得到考虑了网格内格点交互的点云特征，堆叠多层后maxpooling得到每个网格的特征编码。
- 实施细节：将所有点云安排成KxTx7的tensor,K为所有网格点,T为每个网格点中最大点数，7是坐标(3)+雷达回波(1)+中心化坐标(3).这样安排可以更好的组织计算.最后多层VFE输出再重组成三维网格的形式(空的网格特征为0)送入3D CNN。
- VFE单元结构：
  
      Point cloud feature
      FC
      Max pooling
      Concat with Point cloud feature
      ... repeat N times
- 中间层 将网络特征用3D CNN 提取.3D CNN在D,H,W维度上使用3x3x3卷积逐渐扩大感受野，在D（鸟瞰高度）维度上降采样，直至D维度为1（文中降采样至2,reshape到了C维度，得到了128xHxW的常规tensor）送至任务层
      
      Conv3d 3x3x3/2,1,1 64
      Conv3d 3x3x3 64
      Conv3d 3x3x3/2,1,1 64
      Reshape -> 128,H,W
- 任务层：常规CNN，多尺度融合后出一支回归，一支分类（基于训练集平均尺寸的anchor，0和90°两个朝向，在鸟瞰图中根据IOU匹配真值）. 物体框使用传统的中心点坐标和长宽高尺寸和朝向角参数化，中心点回归真值和长宽高回归真值均和Faster RCNN相同，角度直接回归偏差，任务层结构为
  
      Conv 3x3/2 128
      3 x Conv 3x3 128  ->deconv(a) 3x3/1 256
      Conv 3x3/2 128
      5 x Conv 3x3 128  ->deconv(b) 2x2/2 256
      Conv 3x3/2 256
      5 x Conv 3x3 256  ->deconv(c) 4x4/4 256
      Concat  deconv(a)(b)(c) 
      Conv_cls 1x1 2  Conv_reg 1x1 14 (两个anchor)

- 训练技巧
  - 目标界别数据增强：属于同一目标的点随机绕垂直地面轴旋转一个角度,移动一个随机向量，真值框也相应修改
  - 整个点云级别数据增强：点云坐标都乘以随机系数,吗，目标框对应缩放,点云和框整体绕垂直轴旋转随机角度

## 立体物体表示为多视角图片

### VeloFCN

    Bo Li, Tianlei Zhang, Tian Xia， Vehicle Detection from 3D Lidar Using Fully Convolutional Network，IROS 2016

- 数据编码:将点云映射到一个二维平面上，对于点云中某点(x,y,z)，映射到图片的(r,c)像素，其中$\triangle \theta  \triangle \phi$是两个方向的分辨率,对应于传感器圆柱形波面
$$\theta =atan(y,x),\phi=arcsin(\frac{z}{\sqrt{x^2+y^2+z^2}}), r=[\theta/\triangle\theta], c=[\phi/\triangle\phi] $$
在(r,c)上两通道数值为$\sqrt{x^2+y^2},z$,若多个点映射到相同坐标，取更靠近画面的点(后面的点被遮挡).没有点的像素全部填0.
- 特征提取与检测:采用全卷积网进行特征提取与预测，网络结构为

      Conv1 3x3/ 4,2
      Conv2 3x3/2
      Conv3 3x3/2
      Deconv4 2x2/2
      Concat Conv2
      Deconv5a 2x2/2   Deconv5b 2x2/2
      Concat Conv1     Concat Conv1
      Deconv6a 2x2/4,2 Deconv6b 2x2/4,2
  
  - Deconv6a输出2通道接softmax 输出objectness，使用cross entropy
  - Deconv6b 输出24通道，分别表示box的8个点的本地化坐标, 坐标系统定义为以点云为原点, x轴为雷达与点云连线,y轴与地面平行,相应可以得到z轴.这样定义坐标系统涵盖了旋转不变性.使用L2 loss训练.
  
- 训练技巧:在进行数据编码前，对整体点云进行旋转平移摄动进行数据增强;通过多任务loss的加权系数实现训练的不均衡.其中前景和背景的均衡采用离线计算训练集中正负点云样本比例,目标内点云不均匀的均衡采用离线计算属于同目标点云的平均数量和当前物体内点数量的比例
- 后处理:根据objectness大于0.5得到box后，以box内部点云的数量作为score做NMS,移除所有具有重合区域的预测框,并且删去框内小于5个点的框.
- 本方法是多视角方案的鼻祖,很多后续方法的创新都可以在这个算法中找到影子。该算法通过编码点云组织成可用卷积神经网络提取特征的形式，并且对每个点云进行了密集预测。对点的预测采用了以点为坐标原点的物体框坐标，这实际上是后续一系列anchor free检测方法的做法；以点为单位预测物体框的做法是后续基于点云裸数据预测的处理基础。后处理中以框内点数作为得分也使后续增强检测效果的常用手段

### MV3D
    Xiaozhi Chen, Huimin Ma, Ji Wan, Bo Li, Tian Xia “Multi-View 3D Object Detection Network for Autonomous Driving”，CVPR 2017

- 使用鸟瞰图得到3D框的proposal,再将proposal向别的视角投影后使用类似RCNN的方式处理得到检测框。整个模型融合了雷达点云和相机图片.点云编码了鸟瞰图和前视图
- 数据编码：
  - 鸟瞰图：以0.1m的空间分辨率对点云进行编码，共有M+2个通道:
    - 高度(height):将点云分成M片后将每片内对应位置最高的点云高度填入
    - 强度(intensity)：像素内鸟瞰图中最高的点对应的雷达回波强度
    - 密度(density)：像素内点云点的个数为N，则密度编码为$min(1.0,log_{64}(N+1))$.
  - 前视图：同VeloFCN中定义，加了一个雷达回波强度通道，共3通道
  - 彩色图
- 网络结构和预测:使用鸟瞰图得到proposal.在RPN阶段,backbone仅进行了3次pooling，最后接deconv使得整个feature map下采样了4倍. 
  
  对训练集的box进行尺寸聚类得到anchor(每种尺寸包含0和90°两个), RPN进行anchor的分类和参数回归(box中心坐标采用anchor尺寸归一化偏移,尺寸采用对数空间的偏移,与Faster RCNN相同),在RPN阶段不回归box朝向角度.
  
  在训练中，对不包含点云的anchor直接舍弃,anchor与真值匹配通过鸟瞰图的2D box IOU进行，小于0.5为负样本,大于0.7为正样本,中间忽略。RPN后在鸟瞰图上进行了2D NMS操作。
  
  将3D box的proposal 投射到鸟瞰图,前视图，彩色图视角后，并做ROI Pooling后，将三个特征进行融合(eltwise add)并输出检测框. 采用8点坐标(以proposal对角线归一化的角点偏移,文中称这样比中心坐标和尺寸参数更好)和置信度参数化检测框.
  
  特征提取网络均是VGG-16(channel都砍掉一半,删去了第4个pooling),在ROI pooling前4x/4x/2x上采样。在最终检测框前在VGG基础上加了一层fc

- 训练技巧:融合方式采用一种深度融合方式，结构如下。训练时加上了一支单独预测的辅助loss，用于表达真正的deep fusion，并且在训练时随机drop掉支路

      multi-modal input
      eltwise add
      Conv1p  Conv2p  conv3p
      ... repeat 3 times
  
- 总体思路与sliding shapes相同，是faster RCNN在三维空间的自然拓展.但利用了鸟瞰图的2D编码得到proposal节省了计算量。但对于点云编码成鸟瞰图为手工特征,难以说是最优的

### AVOD

    Ku, Jason , et al. "Joint 3D Proposal Generation and Object Detection from View Aggregation." arXiv:1712.02294v4 .

- 针对MV3D仅采用鸟瞰图生成proposal和小目标检出较低的问题,AVOD的RPN将点云鸟瞰图和彩色图融合(anchor投影的ROI crop resize)之后进行，并且backbone(一半通道的VGG16,删去conv4之后的部分)加上了FPN结构使feature map与输入具有相同分辨率
- RPN:因为特征融合需要对齐，将大量anchor进行了视角转换需要大量内存，因此在FPN backbone后面加了conv1x1将通道压缩成1,节省了roi feature map占用的内存;将鸟瞰图和彩色图中与anchor对应的ROI（已被压缩成单通道）crop resize后eltwise add融合，然后FC得到proposal(anchor和真值框匹配在鸟瞰图中根据IOU进行,proposal 没有朝向角)
- RCNN:编码box采用平面内四边行角点宽高坐标和上下沿高度，比8个角编码更符合长方体约束。另外朝向角用cos,sin向量编码，消除±π的二义性。除了将以anchor投影到特征换成proposal投影以外，其余与RPN相同

### MLOD

    J Deng，K Czarnecki，“MLOD: A multi-view 3D object detection based on robust feature fusion method ”，arXiv 1909.04163v1

- 指出了在多视角雷达和rgb图融合框架中的两个问题：
  - 在BEV中未匹配到真值的proposal在RGB图中因视角变换，可能对应真框(对应于深度不同但在相机连线上的目标),但却被当成负样本训练，原先的RGB支路在标注不完全正确的数据上训练，性能自然不是最优的
  - 三维框转到RGB区域后，区域内含有三维框外的颜色信息.

- 针对问题(1)提出了Foreground Mask操作，将框内的颜色像素挑选出来。针对点云的深度图,按照三维框选出符合深度的像素，为防止点云缺失导致颜色信息丢失，将深度图上缺失的区域也选出来，即

$$mask_{i,j}=
\begin{cases}
1 ,m_{i,j} \in [d_{min}-\epsilon,d_{max}+\epsilon]\mathop{\cup}[0,\epsilon] \\
0 ,otherwise
\end{cases}$$

- 针对问题(2)提出多视角检测头，即在每个视角下有各自的检测头
  
## 以三维裸数据表示

针对点云裸数据这种非网格化的数据使用传统的CNN难以直接处理.本节介绍了Pointnet，它在某种意义上就是一种图卷积网络(GCN),是很多点云模型的基础和backbone。

点云中点的特征主要由其空间坐标组成,但空间坐标分布具有很大的分散性,这类方法中对坐标的中心化变换是至关重要的,它保证了训练实例分布具有相似性。点到物体框中心的距离是不容易回归的，因此frustum pointnet和IPOD都额外使用了Tnet,PonitRCNN在点附近安排了网格状分布的anchor.

### 预备知识 pointnet family
  
    Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas. "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation”  CVPR 2017
    Qi, Charles R., et al. "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space." NeurIPS 2017.

#### pointnet

- 考虑到点云的性质
  - 无序。点云数据是一个集合，对排列是不敏感的。
  - 点与点之间的空间关系。一个物体通常由特定空间内的一定数量的点云构成，也就是说这些点之间存在空间关系。
  - 不变性。点云数据所代表的目标对某些空间转换应该具有不变性，如旋转和平移。
- 采用定义在集合上对顺序无感的函数f解决(1)(2)

$$f({\boldsymbol{x_1},\boldsymbol{x_2}...\boldsymbol{x_n}})\approx\gamma(max_{i=1...n}(h(\boldsymbol{x_i})))$$
$\gamma,h$均用mlp来参数化。f函数表示的模型称为pointnet

- 为解决(3)，pointnet中含有对数据和特征的变换矩阵T，T是类似于self-attention那样由特征预测并作用于特征(矩阵乘法)，T的预测网络也使用一个小的pointnet实现（称为$T_{net}$）。对特征做变换时，中间特征维度较高,为简化训练添加了正则化,使变换矩阵尽可能接近正交矩阵
  
$$||I-TT^T||_F$$

- 采用带有$T_{net}$的pointnet可获得点云的全局特征，对三维点云分类直接使用全局特征即可.但对于点云分割,需要输出点云级分类,则将各点属于自身的特征与全局特征concat后作为特征做点云级分类.

#### pointnet++

- Pointnet逐点提取特征,然后maxpooling一下子压缩了所有点的特征，这样太快了。pointnet++提出了set abstraction操作仿照CNN逐层提取特征。
- 模拟CNN encoding操作：Set abstraction (SA):由sampling,group,ponitnet组成。
  - Sampling: 使用最远点采样(farthest point sampling,FPS),递归地从点集中选取离已选点集最远的点)采样得到一系列中心点.它比随机采样更能覆盖整个点云。
  - Group: 将sampling阶段选出的中心点周围一定范围内的点找出来(ball quering),作为一个单元(点数可变，后续pointnet对点数无要求,也可以使用k近邻选取固定个数的点，但尺度不好控制)。
  - Pointnet: 对每个group阶段得到的单元抽取特征，其中单元内的点坐标采用相对中心的中心化坐标
- 为克服点云密度不均衡的特点，需要在模型中注入多尺度特性。
  - Multi-scale grouping (MSG).
  每个group操作使用不同尺度范围内的点,同时在训练时以不同概率随机drop掉单元内的点模拟不均匀的点云训练集.ponitnet对不同尺度的group输出特征concat后作为整个SA的输出
  - Multi-resolution grouping (MRG)
  每个SA输出均由上一层group经过pointnet和上一层所有点感受野范围内点云组成的单元经过poinitnet共同组成. MRG较MSG省去了第一层的多尺度融合，计算量稍小点. 但MRG比MSG难理解，且性能差一些，个人偏向MSG
- 模拟 CNN decoding操作 Feature propagation (FP): 对K近邻点的特征根据距离倒数加权平均得到上采样点的特征.在ponitnet++中，K取3，对应浅层中心点特征与上采样中心点特征concat后经过单元pointnet(即处理点集只有一个点,也可以理解为一连串FC)更新点特征(模仿1x1卷积)

### Frustum PointNets

    Qi, Charles R. , et al. "Frustum PointNets for 3D Object Detection from RGB-D Data." arXiv:1711.08488v2.

- 处理流程
  1. 三维区域proposal:使用2D目标检测器检测得到物体，将点云投影在图片框内部的点集(对应空间中一个透视锥形区域)转至中心射线垂直图片平面的区域作为三维proposal
  2. 在 proposal区域内对点云实例分割(这时变成了二分类语义分割，类似于maskrcnn)，预测锥形区域内属于目标物体的点，并将这些点坐标中心化
  3. 对属于目标的点进行物体框预测
      - Tnet预测中心坐标偏移（3）
      - 预测中心坐标偏移（3）+所有尺寸anchor得分（NS）+角度模板类别得分(NH)+所有尺寸anchor长宽高偏移(3NS)+角度相对模板的偏移(NH)

- 训练技巧
  - 网络中加入onehot向量带入目标检测器的类别增加语义信息
  - 在(2)中坐标中心化后，中心坐标距离真实目标框中心距离较远，因此额外训练一支Tnet预测中心坐标偏移，（3）中坐标采用以Tnet输出坐标为原点的坐标。最后绝对坐标是分割结果中心坐标+Tnet坐标偏移+物体框预测坐标偏移得到
  - 目标框作为一个整体，考虑了预测框和真值框8个角点的L1 loss。采用图片实例分割替代(1)和(2)效果不佳(如IPOD中所尝试)。在得到回归框的过程中，除了使用中心坐标外，还可以复用分割网络中点的特征(IPOD中采用)

### PointRCNN

    Shi, Shaoshuai , X. Wang , and H. Li . "PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud." CVPR 2019.
    Yang, Zetong , et al. "IPOD: Intensive Point-based Object Detector for Point Cloud." arXiv:1812.05276.

- 注意到在三维空间中，物体点云是不会重叠的.物体框的点实际上提供了实例分割的标签。本文利用这个性质将语义分割和proposal生成组成多任务训练
- Proposal生成使用bin based localization.这种方式在点附近划分网格，然后使用网格分类+坐标残差回归的方式优化模型输出（这实际上等价于在点附近均匀设置了相同尺寸的anchor，然后对匹配上的anchor进行分类，再回归坐标残差，角度也是按照这种方式）。因为高度一般较少，不设置分类，只有坐标残差回归（等价于只有高度与点相同的anchor）
- 物体框尺寸回归对数空间中的残差，基数采用训练集中每类的平均尺寸(等价于仅使用平均尺寸这一种尺寸的anchor)，中心坐标和角度解耦进行分类可以规避坐标+角度+尺寸代表的anchor和真值框匹配的过程。
- 每个物体框内有多少点，就会生成多少proposal，后处理中在鸟瞰视角下进行NMS，保留top100~300个proposal
- ROI pooling：将proposal略微扩大一些，然后把其中的点坐标送入下一阶段
- Bounding box refinement
  - 特征处理:首先将点坐标变换到本地坐标，包括坐标中心平移到proposal中心，轴与proposal朝向一致,本地坐标+颜色或雷达回波等特征+分割预测标签onehot,经过mlp变换维度到与分割特征维度相同后与其concat作为每个点特征
  - 根据3D IOU对proposal和真值匹配，匹配上的proposal与第一阶段相同，进行bin based localization.
- Backbone使用pointnet++: SA (np=4096,1024,256,64，MSG)+对称的FP. Refine使用在ROI中随机采样512个点后使用SA(np=128,32,1)
