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
  - 传统的检测器确定正负样本时以IOU>0.5为判断依据，如要提高检测精度，需要提高该阈值，但该阈值提高时,正样本迅速减少，使得训练难以进行.
  - 一种提高精度的方法是在后处理中不断迭代定位(iterative box):即得到检测框后,迭代将其作为proposal,使用回归和分类参数得到检测框。这种做法没有考虑到训练时分类和回归参数是根据IOU阈值进行的,以低阈值训练得到参数在高IOU时性能并不好.
  - 一般而言,proposal与真值的IOU将在检测层后提高,因此在训练时就采用多stage,且不断提高IOU阈值,那么推断时就天然具有逐步完善结构,且因目标框质量不断改善,正样本数量不会减少,逐渐变高的IOU阈值也使得不同阶段的分类和回归参数得到针对性的训练.解决了上述的问题.
  - cascading R-CNN:faster R-CNN的一种扩展,要点是多个stage中IOU阈值不断提高,检测头不共享权值,训练时逐stage进行,推理时结构与训练相同。

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
  
  - TridentNet

        Li Y , Chen Y , Wang N , et al. "Scale-Aware Trident Networks for Object Detection". arXiv.1901.01892

    - 使用图像金字塔计算量太大,而像FPN的特征金字塔不同尺度的目标特征因深度不同,语义表达能力不同。本文使得不同尺度特征表达能力相同.通过多只相似的扩张卷积,调整dialation rates来实现感受野的变化.为了不增加参数,使参数训练更加充分,同时减少模型过拟合的风险，模型在不同尺度的分支上参数共享,仅仅是dialation rates不同.
    - scale aware training: 不同尺度分支的预测上,设置sqrt(wh)区间[l,h],当anchor位于区间内时,分配至这一支。推理时,首先过滤掉位于区间外的结果，然后多只一起做NMS.(在coco上根据s,m,l来设置,并有一定重合)
  
  - RBFNet

        Liu S , Huang D , Wang Y . "Receptive Field Block Net for Accurate and Fast Object Detection". ECCV 2018




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

  
- 改造置信度,使其意义更加明确，优化了直接使用置信度来后处理的NMS
  - IOUNet


  - ConRetinaNet
     Tao Kong, Fuchun Sun, Huaping Liu, Yuning Jiang, Jianbo Shi."Consistent Optimization for Single-Shot Object Detection". arXiv:1901.06563v2
    - 单阶段检测器训练和推理时存在不协调,使得置信度难以很好表示检测的结果:置信度按照anchor分类得到,目标框却依据回归参数对anchor进行了偏移.这种不协调会带来一些列问题,比如,物体比较紧密时,anchor覆盖的区域的类别和回归后的检测框类别不一致;或者背景anchor回归后定位较为精确,但因为置信度较低在NMS中被抑制.
    - 解决方法非常直观: 将分类loss改成回归后目标框的该有的标签. 为了训练的稳定,实际使用的分类loss使用anchor的标签和回归后目标框的标签计算的loss和
    - 还可以将单阶段检测器输出结果进一步refine,即检测头输出更多支,是在前面预测结果框的基础上进一步refine.该文中添加了一个回归分支,对前一阶段的目标框进一步计算,实验表明这种配置效果最优.[此时框的两次回归与两次分类实际上又是不协调的,分类结果对应于第一次回归框,但再添加一次分类,性能略微变差了] 总体的损失函数就是多个阶段的分类和回归损失的和，分类共享权值,各阶段输出同一个标签. 当然单阶段通过框的进一步修正存在的困难在于不像两阶段那样，可以通过ROI pooling调整特征的位置,单阶段的特征是与像素对齐的. 

  - soft nms
  
       Bodla N , Singh B , Chellappa R , et al. "Improving Object Detection With One Line of Code". arXiv:1704.04503v2 

  - softer nms
      He Y , Zhu C , Wang J , et al. "Bounding Box Regression with Uncertainty for Accurate Object Detection". arXiv:2018.08545
    - 一方面,在现有数据集中发现某些目标的目标框标注具有不确定性(遮挡,歧义,标注不正确等造成),因此自然而然想到在目标框回归中引入不确定性，用于表征目标框的确定性.
    - 在传统NMS中使用分类置信度进行抑制，使有些定位更好的框被抑制掉了，因此使用目标框的确定性来抑制更合理,并且可以利用重合在一起的多个框的信息.
    - 目标框参数化为四个角点(定义方式类似于传统的中心点),回归输出角点建模为方差为\sigma，均值为输出的高斯分布，方差跟坐标一样平行作为一支输出。真值的分布为dirac函数，因此只需要优化他们的交叉熵即可,交叉熵为(x_g-x_e)^2/(2\sigma^2)+log(\sigma^2)/2 考虑到\sigma始终为正,采用\alpha = log(\sigma^2)表示，且将L2替换成smoothL1,更容易优化. 于是回归任务损失函数为 exp(-\alpha)(|x_g-x_e|-0.5)+\alpha/2 刚开始初始化时,alpha支路输出至非常小，跟传统的smoothL1回归相同.    
    - 后处理时,NMS中抑制不再使用置信度,考虑到IOU越大,预测方差sigma_i越小定位应当越准确,因此在NMS中具有与最大置信度的框IOU大于0的坐标权重为 exp(-(IOU(b_i,b)-1)^2/t)/\sigma_i与最大的置信度框的坐标进行加权平均. 其中t是个超参数(可取0.005~0.05，衰减需要快一些)

- 改造anchor(anchor free detector)

  Law H , Deng J . "CornerNet: Detecting Objects as Paired Keypoints". ECCV 2018
  - 将目标检测问题建模成为预测目标左上/右下两个点的问题.这样绕开了anchor的使用,从而避免了anchor数巨大带来的计算量和一些列超参数设置.
  - 模型将为每一类 预测左上/右下两个关键点,为每个关键点预测一个embeding,超像素偏移。根据embeding之间距离最短将左上/右下关键点组成目标框,偏移用于恢复下采样feature map像素位置的小数部分
  - 预测关键点时,左上和右下分成两部分进行,每个真值都采用高斯软化后的标签.因在关键点真值一定范围内,都可以恢复质量较好的目标框,根据IOU至少为0.3,可以计算出一个半径,在关键点真值的该半径内，都可以生成目标框，从而将方差设置为1/3的该半径,真值mask被设置为e^(-\beta(x^2+y^2)/2\sigma^2). 分割关键点时loss设置为focal loss形式，使用预测概率动态降低负样本的权重.
  - 为左上和右下关键点预测超像素偏移时,类间共享,采用smoothL1 loss
  - 为关键点预测embedings时,使用一维标量。两个关键点的emdebing之差的绝对值采用pull-push loss优化：关键点应当组合时,emdeding使用L2 loss(pull),不应组合是，使用hinge loss(push) (均除以加和项数归一化)
  - 考虑到角点邻域内没有信息难以定位，在预测环节中加入鲜先验信息: 左上角需要朝右和下看，右下角需要朝左和上看，提出了corner pooling。对于左上关键点预测的两个feature，分别求向右和向下的max pooling然后相加，对于右下关键点预测的两个feature，分别求向左和向下的max pooling然后相加.  
  - 预测环节：传统residual block中第一个conv 3x3 改成了平行两支卷积后作pooling再相加.再经过两个conv 3x3后,三支Conv 1x1分别输出heatmap(C),offsets(1)和embeding(C)
  - backbone: 因为预测关键点，采用分割常用的hourglass结构：先采用Conv 7x7/2 128, Conv 3x3/2 256，然后是两个hourglass，hourglass下采样5次,中间是4个 residual block，上采样时，采用最近邻upsample+conv，还具有skip connection。 训练时不单独添加中间信息对第一个hourglass进行监督(实验发现有损性能),而是添加了跨过整第一个hourglass的skip connection(输入图片和第一个hourglass输出都经过conv 1x1后相加送入第二个hourglass) backbone上加两个预测环节头，分别为左上和右上的预测。
  - 后处理：首先对两个heatmap使用3x3 max pooling，然后挑选出Top 100的关键点,使用对应offset修正，然后使用embeding之间距离(L1)进行过滤，当距离大于0.5或者关键点不属同一类别被过滤掉. 置信度只用两个关键点置信度的平均值，使用原图和horizontal flip图前向后的预测结果框进行soft nms得到最终结果。



	Kaiwen Duan, Song Bai, Lingxi Xie, Honggang Qi, Qingming Huang, Qi Tian. "CenterNet: Keypoint Triplets for Object Detection". arXiv:1904.08189
  Xingyi Zhou, Dequan Wang, Philipp Krähenbühl. "Objects as Points". arXiv:1904.07850
  Tao Kong, Fuchun Sun, Huaping Liu, Yuning Jiang, Jianbo Shi. "FoveaBox: BeyondAnchor-basedObjectDetector". arXiv:1904.03797
 