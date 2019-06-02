# object detection 的新进展
 &nbsp;     　　　　　　　　　　   sqlu@zju.edu.cn
## 引言
本文承袭<RCNN和yolo的前世今生>,总结了目标检测体系中的创新.这些创新的思想本质有些已经在Faster-RCNN和yolo的高版本中得到了应用。

- SSD系列
  - Single Shot Detector(SSD)
      
      Liu W , Anguelov D , Erhan D , et al. "SSD: Single Shot MultiBox Detector", ECCV 2016.

    - 技术特点:全卷积,多尺度预测。
    - 属于较早期的成果,原文使用的是VGG-16的骨架.将stage IV与stage V之间的max pooling 2x2/2改为3x3/1，为保持感受野不变,将后续卷积Conv 3x3 dialation=2. 将FC全换成卷积层,在stage V之后再加了一系列卷积层:
        Conv 3x3 1024
        Conv 1x1 1024   -> head
        Conv 1x1 256
        Conv 3x3 /2 512 -> head
        Conv 1x1 128
        Conv 3x3 /2 256 -> head
        Conv 1x1 128
        Conv 3x3 256    ->head
        Conv 1x1 128
        Conv 3x3 256(V) -> head
    除此以外在Conv4_3上也接出一支head. head均是Conv 3x3，输出(4+C)K通道,C是类别(包含类别),k为anchor数,4是框的参数.构成每个anchor的类别概率和矩形框参数(定义同Faster-RCNN)
    - 每个尺度上,分配了s = (s_max-s_min)(k-1)/(m-1) k~[1,m]，即最大的feature map 尺度为s_min，最小的feature map尺度为s_max，之间均为线性分布的尺度anchor.取s_min = 0.2原图大小,s_max = 0.9原图大小.再次为每个尺度设置不同宽高比{1,2,3,1/2,1/3}，再加1:1尺度为sqrt(s_k*s_{k+1})共6个anchor.
    - 训练中因背景众多,对背景样本loss排序,取前面loss大的，保证正负样本比例为1:3.(online hard negative mining, OHNM)
    - anchor分配: 为每个GT分配IOU最大的anchor,其余anchor与GT IOU大于0.5也关联上,使得每个GT至少有一个anchor关联上.
  
  - Deconvolutional SSD (DSSD)

      Fu, Cheng Yang , et al. "DSSD : Deconvolutional Single Shot Detector." 	arXiv:1701.06659.

    - 更新了主干网架为ResNet-101,且使用基于deconvolution的encoder-decoder结构,增强大分辨率feature map语义信息.改善小目标检测性能.
    - head 加入了一层residual block提高性能,不使用这种输出结构，ResNet-101骨架效果还不如VGG-16
        Conv 1x1 256
        Conv 1x1 256
        Conv 1x1 1024
        eltwise sum
    - Deconvolution的方式: 上采样后与前面等大分辨率feature map融合:
        feature map(smaller)                  feature map(larger)
        convolutiontranspose 2x2 512          Conv 3x3  512
        conv 3x3      512                     Conv 3x3  512
        Eltwise sum/prob         
    - 与SSD类似,在ResNet-101 stageV中stride=2改为1,3x3卷积dialation均改为2.扩大分辨率同时保持感受野不变.
    - 训练流程：
      (1) 拿ImageNet pretrained的ResNet-101加上head和extra layer训练SSD;
      (2) 固定共有参数,训练deconvolution和相应head得到DSSD;
      (3) fiunetune全网络.(某些实验这步效果反而更差了,可省略)
      (4) 为加快推理，卷积层吸收了BN层.

- R-FCN

      Dai J , Li Y , He K , et al. "R-FCN: Object Detection via Region-based Fully Convolutional Networks". NIPS 2016.

检测器有分类器作为骨架,但分类器一个重要的特性是对象位置的平移不变性，但位置对检测器至关重要。对于ResNet-101骨架,将RPN和检测器head都放在stage V,效果很差(Faster RCNN将这些都放在stage IV)，导致stage V 10层网络都是ROI wise的，推理时消耗较多时间.为将Faster-RCNN改造成全卷积结构,提出了position sensitive score map和position sensitive roi pooling
  - position sensitive score map:就是一般的feature map,仅仅是channel数规定为kxkx(C+1),k为每个roi被pooling至kxk，C为检测器类别
  - position sensitive roi pooling:(论文没有写的很明白,通过看源码终于懂了什么意思)将一个ROI拆成kxk网格,每个网格内做pooling,但kxk中的每个网格是在position sensitive score map上对应channel取的，即kxk中每个网格在不同的channel上pooling,最后合并起来，得到Cxkxk的feature map，与Faster RCNN roi pooling之后结构相同.
  - 在训练中,每个网格的信息会汇集到不同channel上,做到位置敏感.(个人觉得不太靠谱，但实验做出来有效)
  - 为了增大最后feature map分辨率,在stage V中移除了stride=2，改用dialation=2.其余跟Faster-RCNN相同.

- 多任务Mask-RCNN

- 逐步完善
- Zhang S , Wen L , Bian X , et al. "Single-Shot Refinement Neural Network for Object Detection". CVPR 2018
- Cai Z , Vasconcelos N . "Cascade R-CNN: Delving into High Quality Object Detection".	arXiv:1712.00726

- 改善多尺度目标检测性能
 
  - 多层特征融合改善小目标检测 feature pyramid network

      Lin T Y , Dollar P , Girshick R , et al. Feature Pyramid Networks for Object Detection. CVPR 2017.

  CV中pyramid有几种用法:(1)构造不同尺度的图片,在图像金字塔中计算特征(每个尺度相互独立,这消耗较多内存);(2)在不同尺度特征上做预测,类似SSD;(3)浅层与深层特征融合后做预测，但仅局限在最深一层. feature pyramid network首次提出将浅层和深层特征融合后在不同尺度上做预测.后续DSSD，yolov3均采用了这种框架.
    - bottom-up路径即网络骨架,每个stage 最后一个block输出 lateral connection
    - top-down路径即逐渐上采样的过程
    - lateral connection: 两支的融合:bottom up的feature conv 1x1和top-down的上采样(最近邻插值)后feature map eltwise sum.然后 Conv 3x3消除锯齿效应. bottom-up的最后一层Conv 1x1后作为top-down第一层.
    - head的classifier和regressor均共享参数,因此输出通道均为256(即在骨架上增加的卷积层)
    - 为每个scale分配不同尺度的anchor,每个尺度宽高比为1,2,1/2，，在每个尺度上加Conv 3x3和两支Conv 1x1做为RPN. anchor与GT有最大的IOU或者IOU大于0.7的正样本,IOU小于0.7的为负样本.
    - RPN出来的结果尺寸到相应层去做ROI pooling [k0+log2(sqrt(wh)/224)],原本ResNet在stage IV上做为k0=4，现在加了2,3,5三个stage. ROI pooling之后加了两层FC(1024)+{regressor(4xC)+classifier(C+1)}，训练得到的两阶段检测器称为feature pyramid network.
  

  - 多尺度网架
    - 分支结构: Li Y , Chen Y , Wang N , et al. "Scale-Aware Trident Networks for Object Detection". arXiv.1901.01892
    - 对block修改: Liu S , Huang D , Wang Y . "Receptive Field Block Net for Accurate and Fast Object Detection". ECCV 2018




- 针对loss的改进
  - 正负样本不均衡 focal loss

       Lin, T. Y. , Goyal, P. , Girshick, R. , He, K. , & Dollár, Piotr. Focal loss for dense object detection. IEEE Transactions on Pattern Analysis & Machine Intelligence, 2017.

  two stage detector解决正负样本不均衡问题采用的机制:1张图中仅有0.3~1k个proposal,proposal中采样,正负样本保持比例为1:3组成minibatch.而one stage detector中直接对dense sampling的anchor进行预测(相当于proposal为sliding window),会有大量简单负样本.
    - focal loss：
$$
-\alpha_{t} (1-p_t)^\gamma log(p_t)
$$
\alpha_{t}用于调节该种类权重,(1-p_t)^\gamma用于区分难易样本. p_t为该类的预测概率
    - 在FPN上每个feature level上连接两支subnet,分别是分类和回归。均是4层Conv 3x3再加 Conv 3x3输出，分类输出CK通道，回归输出4K通道,K为anchor个数,每个feature level安排3个anchor,3个宽高比共9个anchor.参数形式与Faster RCNN相同.回归系数是每个anchor共享(与别的检测器不同).C个类别用sigmoid表示,因此无背景类且每个类别独立,都是2分类问题.
    - 为了稳定训练,分类最后一层bias初始化为 -log((1-p)/p)),p取0.01,这样初始输出前景的置信度便为0.01,新增的卷积层均用高斯初始化.训练得到的单阶段检测器称为RetinaNet
  
  - GH
      Li, Buyu , Y. Liu , and X. Wang . "Gradient Harmonized Single-stage Detector". arXiv:1811.05181

  
- 改造置信度,使其意义更加明确
  - IOUNet
  - ConRetinaNet
  - 


- 改进NMS

      He Y , Zhu C , Wang J , et al. "Bounding Box Regression with Uncertainty for Accurate Object Detection". arXiv:2018.08545
      Bodla N , Singh B , Chellappa R , et al. "Improving Object Detection With One Line of Code". arXiv:1704.04503v2 

- 改造anchor(anchor free detector)


  Law H , Deng J . "CornerNet: Detecting Objects as Paired Keypoints". ECCV 2018
	Kaiwen Duan, Song Bai, Lingxi Xie, Honggang Qi, Qingming Huang, Qi Tian. "CenterNet: Keypoint Triplets for Object Detection". arXiv:1904.08189
  Xingyi Zhou, Dequan Wang, Philipp Krähenbühl. "Objects as Points". arXiv:1904.07850
  Tao Kong, Fuchun Sun, Huaping Liu, Yuning Jiang, Jianbo Shi. "FoveaBox: BeyondAnchor-basedObjectDetector". arXiv:1904.03797
 