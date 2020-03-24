# object detection 的新进展
 &nbsp;     　　　　　　　　　　   sqlu@zju.edu.cn
## 引言
  目标检测作为计算机视觉一个非常基础的问题，当DSSD，yolov3，RetinaNet殊途同归的时候，I thought detection is dead.没想到还是发展如此迅猛。最近进展主要集中在
  （1）利用额外的分割标签. 例如mask-RCNN,cascade R-CNN-v2
  （2）修改训练的细节提高模型性能，例如GHM,Libra-RCNN
  （3）多尺度预测：虽有feature pyramid，仍有一些work采用dialation conv死磕,例如TridentNet,RFBNet；
  （4）重解释置信度：以往目标检测的置信度理论上讲其实是proposal（two stage）或anchor（single shot）的分类置信度，与定位情况并不完全一致,例如IOUNet,ConRetinaNet,softer nms
  （5）摒弃faster RCNN以来anchor这个宝贝，用关键点建模目标框,例如Guided Anchoring,CornerNet,FSAF,FoveaBox,CornerNet,FCOS
  总体而言，从cvpr 2019来看，目标检测这只鸡，可能需要分割这把牛刀来杀。
  本文除了总结了以上目标检测的新成果以外,还罗列了几个baseline[SSD/DSSD,R-FCN,FPN/RetinaNet]用以参考.(R-CNN系列和yolo系列在《RCNN和yolo的前世今生中已经总结,这里不再列出》)

- SSD系列
  - Single Shot Detector(SSD)
      
      Liu W , Anguelov D , Erhan D , et al. "SSD: Single Shot MultiBox Detector", ECCV 2016.

    - 技术特点:全卷积,多尺度预测。
    - 属于较早期的成果,原文使用的是VGG-16的骨架.将stage IV与stage V之间的max pooling 2x2/2改为3x3/1，为保持感受野不变,将后续卷积改成Conv 3x3 dialation=2. 将FC全换成卷积层,在stage V之后再加了一系列卷积层:
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
    除此以外在Conv4_3上也接出一支head. head均是Conv 3x3，输出(4+C)K通道,C是类别(包含背景),k为anchor数,4是框的参数.构成每个anchor的类别概率和矩形框参数(定义同Faster-RCNN)
    - 每个尺度上,分配了s = (s_max-s_min)(k-1)/(m-1) k~[1,m]，即最大的feature map 尺度为s_min，最小的feature map尺度为s_max，之间均为线性分布的尺度anchor.取s_min = 0.2原图大小,s_max = 0.9原图大小.再次为每个尺度设置不同宽高比{1,2,3,1/2,1/3}，再加1:1尺度为sqrt(s_k*s_{k+1})共6个anchor.
    - 训练中因背景众多,对背景样本loss排序,取前面loss大的，保证正负样本比例为1:3(online hard negative mining, OHNM)
    - anchor分配: 为每个GT分配IOU最大的anchor,其余anchor与GT IOU大于0.5也关联上,使得每个GT至少有一个anchor关联上.
  
  - Deconvolutional SSD (DSSD)

      Fu, Cheng Yang , et al. "DSSD : Deconvolutional Single Shot Detector." 	arXiv:1701.06659.

    - 更新了主干网架为ResNet-101,且使用基于deconvolution的encoder-decoder结构,增强大分辨率feature map语义信息,改善小目标检测性能.
    - head 加入了一层residual block提高性能,不使用这种输出结构，ResNet-101骨架效果还不如VGG-16
        Conv 1x1 256
        Conv 1x1 256
        Conv 1x1 1024
        eltwise sum
    - Deconvolution的方式: 上采样后与前面等大分辨率feature map融合:
        feature map(smaller)         feature map(larger)
        convtrans 2x2/2 512          Conv 3x3  512
        conv 3x3      512            Conv 3x3  512
        Eltwise sum/prob         
    - 与SSD类似,在ResNet-101 stageV中stride=2改为1,3x3卷积dialation均改为2.扩大分辨率同时保持感受野不变.
    - 训练流程：
      (1) 拿ImageNet pretrained的ResNet-101加上head和extra layer训练SSD;
      (2) 固定共有参数,训练deconvolution和相应head得到DSSD;
      (3) finetune全网络.(某些实验这步效果反而更差了,可省略)
      (4) 为加快推理，卷积层吸收了BN层.

- R-FCN

    Dai J , Li Y , He K , et al. "R-FCN: Object Detection via Region-based Fully Convolutional Networks". NIPS 2016.

  检测器由分类器作为骨架,但分类器一个重要的特性是对象位置的平移不变性，但位置对检测器至关重要。对于ResNet-101骨架,将RPN和检测器head都放在stage V,效果很差(Faster RCNN将这些都放在stage IV)，导致stage V 10层网络都是ROI wise的，推理时消耗较多时间.为将Faster-RCNN改造成全卷积结构,提出了position sensitive score map和position sensitive roi pooling
  - position sensitive score map:就是一般的feature map,仅仅是channel数规定为kxkx(C+1),k为每个roi被pooling至kxk，C为检测器类别
  - position sensitive roi pooling:将一个ROI拆成kxk网格,每个网格内做pooling,但kxk中的每个网格是在position sensitive score map上对应channel取的，即kxk中每个网格在不同的channel上pooling,最后合并起来，得到Cxkxk的feature map,与Faster RCNN roi pooling之后结构相同.但得到Cxkxk的feature map后,直接在空间上ave pooling后softmax得到类别,对于目标框回归系数,position sensitive score map channel数是4xkxk， postion senitive pooling后得到4xkxk的feature map,然后在空间上ave pooling得到4个回归系数.检测头只剩下PSROI Pooling和AVE Pooling,完全避免了Faster RCNN在ROI Pooling之后的计算.   另一方面kxkx(C+1)厚的feature map实在是太厚了加速不明显,后续的light head RCNN 和Thunder Net将这部分用更少的特征数代替(类别不可知),pooling后保留一个FC,也可以得到很好的效果.
  - 在训练中,每个网格的信息会汇集到不同channel上,做到位置敏感.为了增大最后feature map分辨率,在stage V中移除了stride=2，改用dialation=2.

- 使用分割信息做多任务
    Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick "Mask r-cnn". ICCV 2017
  
  - 除classifier和regressor以外,再加一支全卷积(4xconv+deconv),用于预测mask。像素级预测对对齐要求较高，为了获得较好的mask，需要对原先的ROI Pooling进行升级。原先的ROI Pooling有两次量化操作(1)proposal的边界量化到feature map的像素上,(2)pooling到7x7时,也进行了取整操作。ROI Align取消了这两个量化操作,在pooling至7x7时,直接根据分数坐标在每个bin中插值计算1个或4个点(4个点再pooling至1个点)
  - 因为Faster RCNN中已经有分类分支,因此在mask中再进行类似语义分割那样做像素级分类采用softmax将会耦合两支,Mask RCNN采用sigmoid进行优化,每次进行时,对相应类别那个通道计算loss,推理时根据分类支路取出对应通道mask.
  - 训练和推理方式:训练时将proposal与GT重合的部分像素作为真值(将分割真值整点化到proposal 到全卷积输出的尺寸)。类似地,推理时将全卷积输出对应通道取出后resize到输出框尺寸，然后用阈值过滤。为了减少计算量,mask支路相对目标检测支路针对更少量的proposal进行计算(置信度阈值更高).
  - 实验的notes:(1)mask支路输出单通道(类别不可知)性能下降非常少,但采用softmax性能下降很大,分类还是应放在整体分类支路完成;(2)backbone一直采用到ResNet stage V后,性能比stage IV更好了,跟以往目标检测与平移不变性认识有些矛盾,主要是RIOAlign解决了特征对齐，在大stride时仍没有偏移. (3)采用ROIAlign后检测器性能提升较多,再加mask多任务训练后,AP也提升一些(~1个点)

- 逐步完善
    Zhang S , Wen L , Bian X , et al. "Single-Shot Refinement Neural Network for Object Detection". CVPR 2018
    - 将单阶段检测器改造成1.5阶段.即虽然跟两阶段检测器相同有proposal，但仍跟单阶段检测器那样将ROI与feature map直接绑定
    - backbone与DSSD相似,在SSD支路上(bottom-up)在各个尺度上提取proposal,即加上RPN head做anchor的regressor和fg/bg的二分类，滤掉大量简单proposal.然后在转置卷积分支上(top-down)各个尺度上进行对前序proposal的refine[与普通检测器头相同,做坐标回归和多分类].整个网络直接端到端进行proposal和refine的优化[使用feature map进行ROI的回归目标框不再对应于网格,feature难以对齐,恐怕需要使用deformable convolution效果会更好]

    Cai Z , Vasconcelos N . "Cascade R-CNN: Delving into High Quality Object Detection".	arXiv:1712.00726
    - 传统的检测器确定正负样本时以IOU>0.5为判断依据，如要提高检测精度，需要提高该阈值，但该阈值提高时,正样本迅速减少，使得训练难以进行.
    - 提高精度的方法是在后处理中不断迭代定位(iterative box):即得到检测框后,迭代将其作为proposal,使用回归和分类参数得到检测框。这种做法没有考虑到训练时分类和回归参数是根据IOU阈值进行的,以低阈值训练得到参数在高IOU时性能并不好，且输入数据分布发生了变化,共享相同的参数效果也不好.
    - 一般而言,proposal与真值的IOU将在检测层后提高,因此在训练时就采用多stage,且不断提高IOU阈值,那么推断时就天然具有逐步完善结构,且因目标框质量不断改善,正样本数量不会减少,逐渐变高的IOU阈值也使得不同阶段的分类和回归参数得到针对性的训练.解决了上述的问题.
    - cascading R-CNN:faster R-CNN的一种扩展,要点是多个stage中IOU阈值不断提高,检测头不共享权值,训练时逐stage进行,推理时结构与训练相同。

- 改善多尺度目标检测性能
  - 多层特征融合改善小目标检测

      Lin T Y , Dollar P , Girshick R , et al. Feature Pyramid Networks for Object Detection. CVPR 2017.

    CV中pyramid有几种用法:(1)构造不同尺度的图片,在图像金字塔中计算特征(每个尺度相互独立,这消耗较多内存);(2)在不同尺度特征上做预测,类似SSD;(3)浅层与深层特征融合后做预测，但仅局限在最深一层. feature pyramid network首次提出将浅层和深层特征融合后在不同尺度上做预测.后续DSSD，yolov3均采用了这种框架.
    - bottom-up路径即网络骨架,每个stage 最后一个block输出 lateral connection
    - top-down路径即逐渐上采样的过程
    - lateral connection: 两支的融合:bottom up的feature conv 1x1和top-down的上采样(最近邻插值)后feature map eltwise sum.然后 Conv 3x3消除锯齿效应. bottom-up的最后一层Conv 1x1后作为top-down第一层.
    - head的classifier和regressor(2隐层mlp)均共享参数,因此head卷积层输出通道均为256(即在骨架上增加的top-down卷积层)
    - 为每个scale分配不同尺度的anchor,每个尺度宽高比为1,2,1/2，，在每个尺度上加Conv 3x3和两支Conv 1x1做为RPN. anchor与GT有最大的IOU或者IOU大于0.7的正样本,IOU小于0.7的为负样本.
    - RPN出来的结果尺寸到相应层去做ROI pooling [k0+log2(sqrt(wh)/224)],原本ResNet在stage IV上做为k0=4，现在加了2,3,5三个stage. ROI pooling之后加了两层FC(1024)+{regressor(4xC)+classifier(C+1)}，训练得到的两阶段检测器称为feature pyramid network.
  
  - TridentNet

      Li Y , Chen Y , Wang N , et al. "Scale-Aware Trident Networks for Object Detection". arXiv.1901.01892

    - 使用图像金字塔计算量太大,而像FPN的特征金字塔不同尺度的目标特征因深度不同,语义表达能力不同。本文使得不同尺度特征表达能力相同.通过多支相似的扩张卷积,调整dialation rates来实现感受野的变化.为了不增加参数,使参数训练更加充分,同时减少模型过拟合的风险，模型在不同尺度的分支上参数共享,仅仅是dialation rates不同,这里的多分支在最后一个stage上使用.
    - scale aware training: 不同尺度分支的预测上,设置sqrt(wh)区间[l,h],当anchor和proposal位于区间内时,分配至这一支。推理时,首先过滤掉位于区间外的结果，然后多支一起做NMS.(区间在coco上根据s,m,l来设置,并有一定重合)
  
  - RFBNet

      Liu S , Huang D , Wang Y . "Receptive Field Block Net for Accurate and Fast Object Detection". ECCV 2018
    - 使用人类的视觉系统的研究成果来指导CNN的结构设计,在计算量较少的情况下性能尽可能好.与之接近的工作是Inception,它使用多分支不同大小的卷积核捕捉不同尺度的特征;ASPP(Atrous Spatial Pooling)使用不同的dialation rate捕捉不同尺度的特征; 这里提出的receptive field block(RFB)就是将Inception和ASPP融合起来,使用Inception-ResNet的基本多分支结构,在Conv 3x3后面加Conv 3x3 dialation rate = 3,在两层Conv 3x3后面加Conv 3x3 dialation rate = 5.
    - RFBNet将RFB取代VGG based SSD中在backbone后面添加的feature map大于5的两个卷积(再后面feature map小于5难以做Conv5x5仍用普通卷积)，在Conv4_3后面接一个感受野较小的RFB(RFB-s：两层Conv 3x3分支改成Conv 3x3,单层Conv 3x3改成两支Conv 1x3 和Conv 3x1,其余不变)。RFBNet在SSD baseline上有明显的提升

- 针对训练过程的改进
  - focal loss及RetinaNet

       Lin, T. Y. , Goyal, P. , Girshick, R. , He, K. , & Dollár, Piotr. Focal loss for dense object detection. IEEE Transactions on Pattern Analysis & Machine Intelligence, 2017.

    two stage detector解决正负样本不均衡问题采用的机制:1张图中仅有0.3~1k个proposal,proposal中采样,正负样本保持比例为1:3组成minibatch.而one stage detector中直接对dense sampling的anchor进行预测(相当于proposal为sliding window),会有大量简单负样本.
    - focal loss：
    $$
    -\alpha_{t} (1-p_t)^\gamma log(p_t)
    $$
    \alpha_{t}用于调节该种类权重,(1-p_t)^\gamma用于区分难易样本. p_t为该类的预测概率
    - 在FPN上每个feature level上连接两支subnet,分别是分类和回归。均是4层Conv 3x3再加 Conv 3x3输出，分类输出CK通道，回归输出4K通道,K为anchor个数,每个feature level安排3个anchor,3个宽高比共9个anchor.参数形式与Faster RCNN相同.回归系数是每个anchor共享(与别的检测器不同).C个类别用sigmoid表示,因此无背景类且每个类别独立,都是2分类问题.
    - 为了稳定训练,分类最后一层bias初始化为 -log((1-p)/p)),p取0.01,这样初始输出前景的置信度便为0.01,新增的卷积层均用高斯初始化.训练得到的单阶段检测器称为RetinaNet
  
  - Gradient harmonized mechanism
  
      Li, Buyu , Y. Liu , and X. Wang . "Gradient Harmonized Single-stage Detector". arXiv:1811.05181
  - 与focal loss相同，都是为了解决训练one shot detector中dense sampling anchor时正负样本比例,难易样本比例相差悬殊.GMH是用梯度批量归一化方式来解决这个问题,使各种norm的梯度贡献均衡,这样比例小样本的就不会被淹没
  - GHM-C(classifier)
    - 对于cross-entropy loss的norm是有界的,将[0,1]分成M(30)个bin,分别统计每个bin中样本个数,属于每个bin中的梯度均用该频率去归一化。比例较大的梯度权重便自动降低,比例小的梯度权重自动增加.在每个mini batch中对梯度模直方图使用EMA来平滑和维护,抑制可能出现的极端值.
  - GHM-R(regressor)
    - smoothL1 loss的梯度在x_hat,x_gt达到一定距离时就是1,不是一个连续变化的值,难以使用梯度模的直方图.因此将regression的loss改造成 sqrt((x_hat-x_gt)^2+\miu^2)-\miu  \miu取0.02. 这样梯度便是[0,1]内连续变化的值，可以计算直方图,使用方式与GHM-C相同
  
- 改造置信度,使其意义更加明确，优化了直接使用置信度来后处理的NMS
  - IOUNet
      Jiang B , Luo R , Mao J , et al. Acquisition of Localization Confidence for Accurate Object Detection. ECCV 2018.
    - 置信度并没有很好的表示最后目标框的质量,因此需要检测器输出一个指标更直接反应目标框的质量,IOUNet直接使检测器输出预测框与真值框的IOU,作为预测框的评分.IOUNet分支作为与分类和回归的平行分支,就是在ROI Pooling后加2层mlp,可以是类别相关的IOU。
    - 该IOU预测值可以直接指导NMS.在原始采用分类置信度NMS的方式中,低质量框却有高分类置信度的框将高质量但低分类置信度的框抑制掉了,采用IOU预测值替代原先分类置信度,可以保留质量最好的框.分类置信度在抑制过程中取大保留.
    - 因需要预测IOU,特征对齐比较重要,因此IOUNet是一种两阶段检测器,在ROI-Pooling中,将以前对坐标不可导的操作改成积分平均值(原先的ROI MAX Pooling或ROI Align对坐标均不可导,借鉴了grid sample，将feature看成是坐标的双线性插值函数,可计算ROI内的积分，然后除以面积作为AVE Pooling).该pooling对坐标可导,因此可在预测完目标框后可进一步refine目标框,采用梯度上升方式,提升至IOU评分收敛.(迭代次间插值绝对值小于一定值). 相比而言,直接使用检测器的回归head多次,会损坏检测器的性能(因此cascade R-CNN 多阶段head不共享权重,采用提高IOU阈值重采样训练)。这一步作为最后一步后处理(在IOU guided NMS后)
    - 训练IOUNet分支时,采用真值框随机变换出许多训练框,不同IOU(归一化至[-1,1])均匀采样,使用smoothL1训练IOUNet分支的head.

  - ConRetinaNet
      Tao Kong, Fuchun Sun, Huaping Liu, Yuning Jiang, Jianbo Shi."Consistent Optimization for Single-Shot Object Detection". arXiv:1901.06563v2
    - 单阶段检测器训练和推理时存在不协调,使得置信度难以很好表示检测的结果:置信度按照anchor分类得到,目标框却依据回归参数对anchor进行了偏移.这种不协调会带来一些列问题,比如,物体比较紧密时,anchor覆盖的区域的类别和回归后的检测框类别不一致;或者背景anchor回归后定位较为精确,但因为置信度较低在NMS中被抑制.
    - 解决方法非常直观: 将分类loss改成回归后目标框该有的标签. 为了训练的稳定,实际使用的分类loss使用anchor的标签和回归后目标框的标签(需要重分配标签)计算的loss之平均
    - 还可以将单阶段检测器输出结果进一步refine,即检测头输出更多支,使在前面预测结果框的基础上进一步refine.该文中添加了一个回归分支,对前一阶段的目标框进一步计算,实验表明这种配置效果最优.[此时框的两次回归与两次分类实际上又是不协调的,分类结果对应于第一次回归框,但再添加一次分类,性能略微变差了] 总体的损失函数就是多个阶段的分类和回归损失的和，分类共享权值,各阶段输出同一个标签. 当然单阶段通过框的进一步修正存在的困难在于不像两阶段那样，可以通过ROI pooling调整特征的位置,单阶段的特征是与像素对齐的,只能修改变化后的匹配标签. 

  - soft nms
  
       Bodla N , Singh B , Chellappa R , et al. "Improving Object Detection With One Line of Code". arXiv:1704.04503v2 
    - greedy NMS 循环地将与最大置信度的box IOU大于一定阈值的box置信度直接置零.一方面置信度越高的box不一定位置就越精确，另一方面这种做法在同一种类目标重叠较多时会抑制正确的目标框,简单提高NMS的IOU阈值造成的误检会更多，难以解决这个问题。 还有一种特殊情况是某些误检框会同时框多个在一起的目标,这种框较大，与单个目标框的IOU都不是很大,使用NMS难以抑制(每个目标抑制一次,每次都达不到抑制的IOU阈值).
    - soft NMS 改造了粗暴的greedy NMS做法,根据框之间的IOU对置信度进行一定惩罚,直至小于输出阈值被抑制掉. 抑制的方式是IOU的连续函数,可以用线性(conf = conf*(1-IOU))或者高斯函数(conf = conf*(1-exp(-IOU/\sigma)),\sigma与greedy NMS阈值差不多即可)表示
    - 采用soft NMS后密集的目标都能出来,只是临近的置信度会低一些,也可以抑制掉多个目标的大框,因为它被单个目标多次抑制到了阈值下.

  - softer nms
      He Y , Zhu C , Wang J , et al. "Bounding Box Regression with Uncertainty for Accurate Object Detection". CVPR 2019
    - 一方面,在现有数据集中发现某些目标的目标框标注具有不确定性(遮挡,歧义,标注不正确等造成),因此自然而然想到在目标框回归中引入不确定性，用于表征目标框的确定性.
    - 在传统NMS中使用分类置信度进行抑制，使有些定位更好的框被抑制掉了，因此使用目标框的确定性来抑制更合理,并且可以利用重合在一起的多个框的信息.
    - 目标框参数化为四个角点(定义方式类似于传统参数化方式的中心点),回归输出角点建模为方差为\sigma，均值为输出的高斯分布，方差跟坐标一样平行作为一支输出。真值的分布为dirac函数，因此只需要优化他们的交叉熵即可,交叉熵为(x_g-x_e)^2/(2\sigma^2)+log(\sigma^2)/2 考虑到\sigma始终为正,采用\alpha = log(\sigma^2)表示，且将L2替换成smoothL1,更容易优化. 于是回归任务损失函数为 exp(-\alpha)(|x_g-x_e|-0.5)+\alpha/2 刚开始初始化时,alpha支路输出至非常小，跟传统的smoothL1回归相同.    
    - 后处理时,NMS中抑制不再使用置信度,考虑到IOU越大,预测方差sigma_i越小定位应当越准确,因此在NMS中具有与最大置信度的框IOU大于阈值的坐标权重为 exp(-(IOU(b_i,b)-1)^2/t)/\sigma_i与最大的置信度框的坐标进行加权平均. 其中t是个超参数(可取0.005~0.05，衰减需要快一些)，此外还应用了soft NMS

- 改造anchor
  
  - Guided Anchoring
  
        Jiaqi Wang, Kai Chen, Shuo Yang, Chen Change Loy, Dahua Lin. "Region Proposal by Guided Anchoring". CVPR 2019
  
    - 以往的方法中anchor均是由sliding window产生,这样产生的anchor就会非常多,大部分覆盖不到目标,且依赖手工设计的尺寸和宽高比.这里提出来一种anchor生成方式,可以极大提高anchor生成质量,减少anchor数量,也使anchor从手工设计中解放出来. 将anchor建模成两个分解问题: 
    - 出现的位置:使用Conv 1x1 + sigmoid产生的feature map表示anchor出现的概率,并使用阈值过滤
    - 依赖于位置的形状:使用Conv 1x1生成两通道feature map,表示对数宽高dw,dh(等价于对一个大小适中的anchor的宽高(对应于原图中w_0,h_0)进行regression，实际宽高为w = w_0exp(dw),h = h_0exp(dh)
    - 因预测的anchor形状在不同位置形状都会变化,为使特征感受野与anchor形状匹配,还需要对特征进行adaptation:用预测的形状来计算deformable convolution的offsets(在形状预测feature map后面加Conv1x1得到offset],用deformable convolution来计算矫正后的特征.
    - 训练方式:在普通的检测器loss额外再加两个anchor位置和形状loss
    - anchor位置loss: 在真值框宽高比例范围[0,\sigma_1)为正样本,[\sigma_1,\sigma_2)为忽略,[\sigma_2,1)及真值框外部均维负样本. 当FPN下多个feature map上生成anchor时,首先在合适的feature map上分配真值生成正样本,正样本相邻尺度的对应区域为忽略区域。当真值框有重叠时,采用的策略是优先级顺序,正样本>忽略区域>负样本.采用二分类的focal loss
    - anchor形状loss: 首先计算在该位置上能产生最大IOU的宽高.因连续解析计算困难，实际上采用的是采样几个点(用的是retinaNet的anchor形状参数)，代进去计算argmax.然后使用iou bounded loss: L1(1-min(w/w_g,w_g/w))+L1(1-min(h/h_gm,h_g/h))
    - 由于使用guided anchor数量较RPN少很多且质量更高,训练two stage detector时第二阶段应使用更高的正样本IOU阈值和更少的proposal训练. 该文还发现可以使用guided anchor来finetune，预测时就直接使用guided anchor可以提升性能. 

  - CornerNet
    
        Law H , Deng J . "CornerNet: Detecting Objects as Paired Keypoints". ECCV 2018
    
    - 作为无anchor检测器的开山鼻祖，启发了后续一系列工作.将目标检测问题建模成为预测目标左上/右下两个点的问题.这样绕开了anchor的使用,从而避免了anchor数较大带来的计算量和一些列超参数设置.
    - 模型将为每一类 预测左上/右下两个关键点,为每个关键点预测一个embeding,超像素偏移。根据embeding之间距离最短将左上/右下关键点组成目标框,偏移用于恢复下采样feature map像素位置的小数部分
    - 预测关键点时,左上和右下分成两部分进行,每个真值都采用高斯软化后的标签.因在关键点真值一定范围内,都可以恢复质量较好的目标框,根据IOU至少为0.3,可以计算出一个半径,在关键点真值的该半径内，都可以生成目标框，从而将方差设置为1/3的该半径,真值mask被设置为e^(-\beta(x^2+y^2)/2\sigma^2). 分割关键点时loss设置为focal loss形式，使用预测概率动态降低负样本的权重[在原始focal loss基础上在乘以(1-真值)^\beta用来降低过度区域权重].
    - 为左上和右下关键点预测超像素偏移时(stride整数化后的小数部分),类间共享,采用smoothL1 loss
    - 为关键点预测embedings时,使用一维标量。两个关键点的emdebing之差的绝对值采用pull-push loss优化：关键点应当组合时,emdeding使用L2 loss(pull),不应组合时使用hinge loss(push) (均除以加和项数归一化)
    - 考虑到角点邻域内没有信息难以定位，在预测环节中加入先验信息: 左上角需要朝右和下看，右下角需要朝左和上看，提出了corner pooling。对于左上关键点预测的两个feature，分别求向右和向下的max pooling然后相加，对于右下关键点预测的两个feature，分别求向左和向下的max pooling然后相加.  
    - 预测环节：传统residual block中第一个conv 3x3 改成了平行两支卷积后作pooling再相加.再经过两个conv 3x3后,三支Conv 1x1分别输出heatmap(C),offsets(2)和embeding(C)
    - backbone: 因为预测关键点，采用分割常用的hourglass结构：先采用Conv 7x7/2 128, Conv 3x3/2 256，然后是两个hourglass，hourglass下采样5次,中间是4个 residual block，上采样时，采用最近邻upsample+conv，还具有skip connection。 训练时不单独添加中间信息对第一个hourglass进行监督(实验发现有损性能),而是添加了跨过整第一个hourglass的skip connection(输入图片和第一个hourglass输出都经过conv 1x1后相加送入第二个hourglass) backbone上加两个预测环节头，分别为左上和右上的预测。
    - 后处理：首先对两个heatmap使用3x3 max pooling，然后挑选出Top 100的关键点,使用对应offset修正，然后使用embeding之间距离(L1)进行过滤，当距离大于0.5或者关键点不属同一类别被过滤掉. 置信度只用两个关键点置信度的平均值，使用原图和horizontal flip图前向后的预测结果框进行soft nms得到最终结果。

    Kaiwen Duan, Song Bai, Lingxi Xie, Honggang Qi, Qingming Huang, Qi Tian. "CenterNet: Keypoint Triplets for Object Detection". arXiv:1904.08189
  
    - 在CornerNet的基础上增加预测目标框中心点(也是通过feature map的网格分类和超像素offsets来实现定位)缓解CornerNet在左上/右下关键点匹配时仅有embeding之间的距离判据而造成的误检,即需要判断在目标框中央区域有同一类别的中心点(同样在heat map中选出top k个点)才保留预测框. 中心区域定义为目标较大时(大于150像素)采用5x5网格中心网格区域,小目标(小于150像素)采用3x3网格中心网格区域。 因为采用相同比例时小目标中心区域太小，中心点概率较低易被拒绝,而大目标反之,因此中心区域大目标比例较小而小目标比例较大.
    - 中心点预测时增加了center pooling，即结果为该行,该列feature map最大值得和.
    - 另外还提出了cascade corner pooling，即在原始corner pooling基础上增加了往区域内部看的机制:在一次pooling后再往取到最大值的地方往区域内部正交方向做pooling,两者求和。以左上 向右的pooling为例,增加了右边取到最大值得点再向下pooling再相加.虽然从概念上看cascade corner pooling增加了区域内的信息，但一方面较繁琐,另一方面从ablation看用处也不是很大.

  - 关键点+边界框距离
       
      Chenchen Zhu, Yihui He, Marios Savvides. "Feature Selective Anchor-Free Module for Single-Shot Object Detection". CVPR 2019
   
    - 将目标检测问题建模成预测像素是目标出现的概率和该像素距离目标边界距离.
    - 目标出现的概率:与检测头类似,在特征上加一层卷积层(C通道)+sigmoid,直接预测该像素有该类别目标的概率.真值框投射到该feature map后,在在真值框宽高比例范围[0,\sigma_1)为正样本,[\sigma_1,\sigma_2)为忽略,[\sigma_2,1)及真值框外部均为负样本.采用focal loss训练. (文中提到小目标真值实例优先级更高,没明白必要性:若采用sigmoid类别之间没有竞争性,可以同时兼顾，但也可以每个像素仅分配一个类别,文中还提到忽略区域在多尺度特时相邻特征也忽略,这在真值投射时候就已经保证了)
    - 目标框形状: 在特征上加一层卷积层(4通道)+ReLU.使用四个参数,但意义是像素距边界框的距离，然后可计算与真值框的IOU,直接对正样本区域IOU进行优化.最终形成目标框时,左上坐标为(i-x1,j-x2)，右下坐标为(i+x3,j+x4)，置信度就是(i,j)的目标出现概率.
    - 在多尺度预测训练时,采用一种online feature selection的方式,即在各个feature level上前向,反传loss最小的那个level. 推理时不需要这个方式,可直接在NMS层面将多余的结果抑制掉
    - FSAF module可与anchor based检测器采用multitask的形式一起训练,可以获得更好的性能
 
	    Tao Kong, Fuchun Sun, Huaping Liu, Yuning Jiang, Jianbo Shi. "FoveaBox: Beyond Anchor-based Object Detector". arXiv:1904.03797
    - 作为comtemporary的work,和FSAF思路几乎相同,仅在细节上有所差别.它们之间差别为：
    - 仍根据目标尺寸分配至各feature level,在[S_l/a^2,S_l*a^2]范围内的目标均分配在P_l 上. 在retinaNet中P3的尺寸S3取32，a取2.区间有所重叠，有些目标会被非配置不同level上
    - 定位的参数化方式不同. FoveaBox采用除以 sqrt(S_l)的归一化尺寸,且映射到对数空间,以SmoothL1作为损失.

      Zhi Tian, Chunhua Shen, Hao Chen, Tong He. Fcos: fully convolutional one-stage object detection. FCOS: Fully Convolutional One-Stage Object Detection ICCV 2019
    - 与上面两个anchor free检测器方法相同.不同之处在该文着重论述了当box重叠时预测目标的二义性的解决:不同尺度的目标放在FPN不同尺度上,可以降低feature map上一个像素点属于同一个目标,当这种情况真的发生时，该像素属于框回归更小的目标(更小的目标)
    - 因为FCOS在框内都作为正样本,因此会有一些false positive。因此它在主干后面加了一只centerness的评分.定义为sqrt(min(l,r)/max(l,r)*min(b,t)/max(b,t)),越接近1越靠近中心，边缘处接近0.最后目标置信度为centerness与分类置信度的成绩,这样会在NMS阶段抑制false positive。因此它在主干后面加了一只centerness的评分
    - 检测head共享权重，但回归距离归一化时各个尺度学习独有的归一化参数，即FoveaBox中sqrt(S_l)，FCOS以exp(a_i*x)表示回归距离，a是可学习参数


      Xingyi Zhou, Dequan Wang, Philipp Krähenbühl. "Objects as Points". arXiv:1904.07850
  
    - 思路更加简洁,每个框使用中心关键点加上回归尺寸表示.在骨架上添加head得到C通道预测stride整点关键点,2通道预测超像素的offset,2通道预测该中心点对应的宽和高.head结构均为Conv 3x3 + Conv 1x1. 回归的loss选用L1(实验中L1比smoothL1好很多)
    - 预测中心关键点时,真值和loss处理方式与cornerNet相同,回归的宽高尺寸变化范围较大,loss的权值需要偏小(<=0.1).
    - 后处理时与cornerNet相同,在heatmap首先 max pooling 3x3寻找峰值,后在所有类别上寻找Top 100个框(可以是同一个位置). max pooling充当了NMS环节,后期出框后不需要NMS.置信度用中心关键点置信度表示.
