# pose estimation
 &nbsp;     　　　　　　　　　　   sqlu@zju.edu.cn

人体姿态估计是计算机视觉中一个重要的任务, 也是计算机理解人类动作，行为必不可少的一步.在实际求解时对人体姿态的估计常常转化为对人体关键点的预测问题，即首先预测出人体各个关键点的位置坐标，然后根据先验知识确定关键点的空间位置关系,从而预测得到人体骨架.这里主要聚焦图片域的人体姿态估计,即每个关键点预测一个二维坐标(x,y).
对于2d姿态估计,主要有top-down和bottom-up两种思路:
- top-down 首先对图片进行目标检测,找出所有人,然后将人从图中crop出来,进行单人姿态估计.
- bottom-up 首先找出图片中所有关键点,然后对关键点分组,从而得到一个人.

# 数据集
- LSP(Leeds Sports Pose Dataset): 单人体关键点检测人数据集,关键点个数为14, 样本数2k, 在目前的研究中心作为第二数据集使用
- FLIC(Frames Labeled In Cinema)：单人体关键点检测数据集,关键点个数为9，样本数为2w，在目前的研究中心作为第二数据集使用
- MPII(MPII Human Pose DataSet): 单人/多人体关键点检测数据集,关键点数为16, 样本25k, 是单人体关键点检测的主要数据集
- MS coco: 多人体关键点检测数据集, 关键点数为17, 样本多于30w,多人体关键点主流数据集
- human 3.6M: 3D人体姿态估计的最大数据集,由360万个姿势和相应的视频帧组成，这些视频包含11位演员从4个摄像头视角执行15个日常活动的过程, 该数据集虽然较大，但人数较少,在受限的实验室场景录制，跟实际场景相差较大，容易被overfit. 不过3d骨架的数据集制作比较困难.
- PoseTrack: 多人体关键点跟踪数据集, 包含单帧关键点检测, 多帧关键点检测,多人关键点跟踪，多于500个视频序列, 帧数超过2w， 关键点数目为15.
- 评价指标
PCK@thred: Percentage of Correct Keypoints,计算检测的关键点与其对应的groundtruth间的归一化距离小于设定阈值(thred)的比例,FLIC 中是以躯干直径(torso size) 作为归一化参考. MPII 中是以头部长度(head length) 作为归一化参考，即 PCKh.

# 方法
## top-down系列
- CPM
  `Ramakrishna, V., Kanade, T., Sheikh, Y.. “Convolutional Pose Machines.” CVPR, 2016`

  从CPM开始,使用heat map(channel一般是人体关键点的个数或者关键点个数+1)隐式建模关键点的位置，而不是直接回归坐标. 这种heatmap方式广泛使用在人体骨架的问题里，跟人脸landmark有明显的的差异，一般人脸landmark会直接回归(FC)出landmark的坐标。人脸landmark对速度要求更高但相对比较简单,另外可以回归到subpixel,但是heatmap精度最多只能到pixel, 人体姿态的自由度较大，直接回归比较困难. heatmap的GT就是与关键点为中心的二维高斯分布(高斯核大小为超参).
  使用multistage结构,每个stage的输入是原始图片和上个stage输出的belief map, 当前stage根据这两个信息继续通过卷积提取信息，产生新的belief map,这样经过不断refinement,最终得到一个较为准确的结果. 每个中间belief map也计算loss,作为中间监督信号.

       input branch:
       input                                     
       Conv 9x9
       pool 2x
       Conv 9x9
       pool 2x
       Conv 9x9
       pool 2x
       Conv 5x5
    ---
       output  branch
       Conv 9x9
       Conv 1x1
       Conv 1x1  -> belief map  <- Loss(L2)
    ---
       fuse branch
       eltwise add                        
       Conv 11x11
       Conv 11x11
       Conv 11x11
       Conv 1x1
       Conv 1x1
    ---
       input branch -> [output branch  -> fuse branch ->] x N stage.
                           inuput branch  ^   
- hourglass

   `Newell, Alejandro , K. Yang , and J. Deng . "Stacked Hourglass Networks for Human Pose Estimation." ECCV 2016.`
  hourglass也是一种multistage结构，比CPM更简洁一些, 该网络由对个漏斗状网络堆叠起来,每个hourglass都包含bottom-up过程和top-down过程，前者通过卷积核pooling降采样,后者通过upsample上采样. hourglass也使用了intermediate supervision.

- CPN

   `Yilun Chen, Zhicheng Wang, Yuxiang Peng, Zhiqiang Zhang, Gang Yu, Jian Sun. Cascaded Pyramid Network for Multi-Person Pose Estimation. CVPR 2018`

   该网络是coco2017 keypoint benchmark冠军。网络由两部分组成,分别为GlobalNet和RefineNet.GlobalNet完成对关键点做初步检测,使用了ResNet作为backbone, 接上FPN后进行检测.对于没有检测到的关键点，使用RefineNet进一步挖掘,将FPN多个分辨率的feature map通过卷积和上采样统一到8x,concat后进行检测，在训练中RefineNet部分采用hard negative mining，只有loss较高的点才进行反传.

- MSPN
  
   `Wenbo Li, Zhicheng Wang, Binyi Yin, Qixiang Peng, Yuming Du, Tianzi Xiao, Gang Yu, Hongtao Lu, Yichen Wei, Jian Sun. Rethinking on Multi-Stage Networks for Human Pose Estimation`
   该网络是coco2018 keypoint benchmark冠军， 在CPN的基础上增加多stage的操作.
   (1) 仍采用ResNet和FPN结构,增加FPN的上采样通道(类似PANet),关键点的heatmap是在最后的高分辨率特征图上计算。
   (2) 多stage主要体现在neck中, 结构类似efficient det的neck, 相邻stage之间对应分辨率的两个feature map都连接起来.
   训练过程中，每个stage的中间监督信号为coarse-to-fine的方式，即heatmap的GT中高斯核大小从大逐渐变小。

- HRNet
  
  `Ke Sun1,Jingdong Wang, et.al. Deep High-Resolution Representation Learning for Human Pose Estimation. CVPR2019`
  该网络本身是pose estimation的各种网络结构观察中获得insight而提出,但作为一种backbone,不仅对姿态估计有效，也可以应用到计算机视觉的其他任务，诸如语义分割、人脸对齐、目标检测、图像分类等.
  大部分深度网络都有降采样过程,为了完成精细的定位后续又会上采样,即使浅层高分辨率特征和经下采样后上采样回来的特征之间有skip connetc,中间仍会丢失信息, HRNet的理念就是时钟保留每个分辨率的特征. 所有特征都并行存在和处理. 当一个stage结束需要降采样时便拉出一个分支，但前一个stage的分辨率仍保留, 并且在降采样这一点所有分辨率的特征进行一次融合,所有特征变换到其他特征的尺寸后相加,主要是使用strided Conv 3x3下采样和upsample上采样。
  关键点的heatmap是在最后的高分辨率特征图上计算的。

## bottom-up系列
- openpose
  
`Zhe Cao, Tomas Simon, Shih-En Wei, Yaser Sheikh. Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields,CVPR 2017`
该网络是coco2016 keypoint benchmark冠军. openpose首先根据图片生成(a)Part Confidence Maps 和(b)Part Affinity Field.
(a)和top-down的heatmap没有本质区别，只不过是所有人的关键点. (b) 引入的PAF是关键点之间建立的向量场,描述了limb的方向,推理时根据这个向量场, 生成各个关键点之间的连接权重，然后采用二分图匹配(匈牙利)算法进行关键点的分组.
PCM由N个channel组成, N为需识别的关键点个数,每一类关键点放在一个通道上.
PAF由2x(N-1)个通道组成, 描述了相连关键点之间的向量场.
网络采用同为Yaser Sheikh组出品CPM类似的结构, 采用多个stage不断refinement,前一个阶段的输出的PCM, PAF和初始feature map作为下一个阶段输入,并且采用intermediate supervision
PCM的真值生成较为容易, 只需按照定义, 按照关键点类别生成每个通道的真值，当同一位置有多个类别时，采用置信度较大的那个.
PAF定义为关键点之间的向量场, 当pixel位于两个关键点$x_{j1}, x_{j2}$ 之间时 (实际操作时，pixel到连线的距离小于阈值) $v = \frac{x_{j2}-x_{j1}}{||x_{j2}-x_{j1}||_2}$, 当不在连线上时为0. 当同一pixel位于多条连线时,向量取所有非零向量的平均值.
推理时结合PCM和PAF,计算每两个需相连关键点之间的权重: $\int_{u=0}^{u=1}PAF(p(u)) \frac{d_{j2}-d_{j1}}{d_{j2}-d_{j1}}; p(u) = (1-u)d_{j1}+ud_{j1}$，其意义就是在两个待相连的关键点之间与向量场做点积,若应当相连，则其积分应当更大. 操作时，在沿线上的pixel的与PAF之间作点积.得到两个待相连关键点所有连线权重后，采用匈牙利算法得到此二分图的匹配，得到每两个点最佳的匹配,组合为一个人.

- associative embedding
  
`Alejandro Newell, Zhiao Huang, Jia Deng. Associative Embedding: End-to-End Learning for Joint Detection and Grouping, 	arXiv:1611.05424`

在hourglass基础上扩展, hourglass原本只能处理单个人的关键点,现在嵌入其输出一个向量作为tag, 称为associative embedding, 拥有相似tag的关键点聚合为一组. 实践证明这个tag只需要一维就够用了(一个实数). 这样本来输出N channel tensor扩展为 2N, 每个heatmap维度上增加一维embedding
为了聚合同一个人的关键点并且区分不同人的tag，采用的loss为
$$
  \frac{1}{NK}\Sigma_{n}\Sigma_k{(\bar h_n-h_{k}(x_{nk}))^2}+\frac{1}{N^2}\Sigma{n}\Sigma{n'} \exp(- \frac{1}{2\sigma^2}(\bar h_n - \bar h_n')^2) 
$$
其中$\bar h_n = \frac{1}{K} \Sigma_{k}h_{k(x_{nk})}$, $h_k \in \mathbb{R}^{w \times h}$ 表示第k个关键点对应的embedding, N表示人的数量,K表示关键点个数, $x_{nk}$表示第n个人, 第k个关键点坐标.
前半部分使同一个人的所有embedding尽量接近,后半部分使不同人的embedding尽量远离.
本论文姊妹篇cornerNet将此思路运用在目标检测上(检测左上和右下两个关键点的组合)