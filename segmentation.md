# segmentation

 &nbsp;     　　　　　　　　　　   sqlu@zju.edu.cn

## introduction

从细粒度由粗到细来看,视觉感知的任务有分类(图像级别),检测(实例框),分割(像素),其中作为像素感知的分割任务进一步有语义分割和实例分割两类,语义分割只需要给出每个像素属于何种类别,实例分割需要进一步区分每一个像素属于哪一个实例.作为从图像分类发展并壮大而来的卷积神经网络,在解分割任务时,需要解决分类任务天生具有的平移不变性和分割像素位置敏感之间的矛盾.

语义分割主要围绕使特征分辨率更精细和使特征语义信息更丰富之间的对立统一展开,使用全卷积网络将语义分割建模成像素级分类已是统治级的方法。为得到精细化的特征,deeplab系列使用空洞卷积(dialated convolution)在保持感受野和模型深度不变的情况下增大特征空间尺寸,FCN等方法将特征上采样,融合浅层特征的encoder-decoder结构(e.g. U-Net)是主要表现形式;为增强语义信息,可添加功能模块或增大特征感受野,如Global CNN,parseNet,PSPNet,BiSeNet等等。

实例分割具有bottom-up和top-down两种范式.bottom-up范式(segment then cluster)通过预测每个像素的embedding，然后聚类后处理得到结果,因其效果一般比不上top-down的方法不展开. top-down方法(detect then segment)需要依赖检测器,典型的依赖二阶段检测器的方法有FCIS,Mask-RCNN,依赖单阶段检测器方法有YOLACT,Blendmask.当然也有不依赖检测器直接得到实例分割结果的Polarmask和SOLO.依赖二阶段检测器的实例分割方法主要通过检测器物体框进行ROI pooling(Align)后定位实例位置解决平移不变性问题,而当没有检测框定位时,CNN需要学实例在空间上的绝对位置.解决的方法首先由FCIS引入,通过将位置信息编码到不同通道上,后续YOLOACT更是证明了带有zero padding的CNN可以直接学到实例绝对位置信息(无纹理的图可以在不同位置进行不同表达).SOLO中将空间坐标作为输入送给分割头,可将AP提高了2+个点,Blendmask则是借鉴FCIS将YOLACT中基mask的系数改成position sensitive,可在基本不增加耗时的同时提高性能.

## semantic segmentaion

### FCN (Fully Convolutional Networks)

  `Jonathan Long, Evan Shelhamer, Trevor Darrell. Fully Convolutional Networks for Semantic Segmentation CVPR 2015.`

- 率先将语义分割问题建模成像素级分类且将图像分类的模型改造成全卷积网络作为预训练模型:在CNN的最后一层feature map空间上每一个像素对应一片感受野,对其分类就是对感受野分类,与切块patch by patch 训练是等价的。FCN中感受野重合较多,前向比patch by patch 的方式更加高效.对于像AlexNet,VGG这样的网络,最后有全连接层,可直接将FC转化为卷积层,即卷积核尺寸等于最后的feature map空间尺寸,将 $(C_{in} \times h \times w) \times C_{out}$的FC参数转为$C_{out} \times C_{in} \times h \times w$的卷积核。这样输出图片变大后,输出将在空间上扩展成一个feature map.
- 为了提高预测结果的空间精确度, 将最粗糙的feature map上采样(因convtranspose只能整数倍上采样,因此需要crop到与对应feature map相同尺寸)concat后Conv 1x1来分割.实现时为避免concat的内存开销,直接两个分别Conv 1x1后相加(相当于分两组的卷积)。融合pool4(VGG中)得到16 stride的结果称为 FCN-16s,再融合pooling3得到8 stride的结果称为 FCN-8s
- 训练时将网格输出插值上采样到与原图相同:网络中上采样用双线性变换核初始化的convtranspose,最后8x上采样到原图是固定的双线性变换。

### deeplab series

使用空洞卷积使分割结果更加精细,并经历了系列改进逐渐推高性能.
#### deeplabV1
  
  `L.-C. Chen, G. Papandreou,I. Kokkinos ,K. Murphy and A.L. Yuille. Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs. ICLR, 2015.`

- 堆叠conv+pooling的方式使得feature map分辨率太小,结果太粗糙:使用dialated convolution解决. 移除pooling(或stride)并以dialated convolution替换.这种情况下输入的feature map不再下采样，代之以卷积核上采样,在某种意义上等价，原先模式下的计算都保留了下来且感受野是不变的，得到的feature map更大.
- 为解决CNN平移不变性与分割的定位之间矛盾,采用CRF(条件随机场)后处理解决.因这部分在后续被抛弃，因此不展开.
- pipline：借鉴FCN,将分类的CNN最后的FC变成卷积层,然后去掉最后两个stage的pooling(或stride),加大这两个stage中卷积层的dialation rate,最终可以得到8x降采样的预测，再双线性上采样至原图大小，最后使用CRF refine结果。
- 实现上的优化:在改造VGG的FCN改造中第一个FC本来对应的是Conv7x7,dialation rate=4，感受野相同时,替换为conv3x3且dialation rate=12. 将这些FC的输出从4096压缩到了1024. 称为deeplab-largeFOV.
  
#### deeplabV2
  
  `L.-C. Chen, G. Papandreou,I. Kokkinos ,K. Murphy and A.L. Yuille. DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. PAMI, 2016.`
  
- 借鉴了PSPNet:(`Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia. Pyramid Scene Parsing Network. CVPR,2017`)，强调整图语境对像素预测的作用,增大感受野对提高性能至关重要,且提高对多尺度目标的性能,增加了ASPP(atrous spatial pyramid pooling)模块:将最后大感受野的卷积层(Conv 3x3，r=12)改成多分支并行dialatied convolution, r=6,12,18,24,然后融合(eltwise add)送入后续层.(PSPNet中则通过pooling至不同尺寸后Conv1x1再上采样resize到pooling前尺寸再concat后预测)
- 实现dialation convolution的方式变迁: v1在caffe中实现,因此按照定义改变im2col中feature map的采样方式,比较直接. v2改在tensorflow中实现，因此考虑将输入feature map 先变成rxr个feature map(在N维上concat)，然后再通过普通卷积. rxr个feature map通过`concat(f[:,:,i::r,j::r]) for i in range(r) for j in range(r)`得到，即dialation采样的rxr种方式.
- learning rate policy改进成poly $(1-\frac{iter}{max_iter})^{power}, power = 0.9$
- 推理时使用图像金字塔(0.5,0.75,1),将结果resize到原图并且max融合后得到最终结果,backbone改用coco预训练的Resnet101.

#### deeplabV3

  `L.-C. Chen, G. Papandreou, F.Schroff, H. Adam. Rethinking Atrous Convolution for Semantic Image Segmentation. CVPR 2017`

- 进一步改进了ASPP模块: 当dialation rate过大时,feature map中有效元素与卷积核作用很少，卷积核大部分落在padding区域,卷积核退化成了Conv 1x1.因此ASPP中r=24的分支直接用Conv1x1替代，并增加了global ave pooling分支(借鉴ParseNet). 整个平行4分支concat,conv1x1 256后送分类 (conv 1x1 +softmax)
- 采用8x预测结果上采样到原图后计算loss,比真值下采样8x再计算loss更佳,这一步可以看成一个最简单的decoder.

#### deeplabV3+

  `L.-C. Chen, Y.Zhu, G. Papandreou, F.Schroff and H. Adam. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. ECCV 2018`

- 为了得到更精细的分割结果,可通过膨胀卷积的方式使输出分辨率进一步加大,但计算量和显存吃不消，所有中间stage的feature map都增大了.因此v3+增加了decoder结构,预测到4x尺寸. 以v3的结果双线性上采样到4x,与对应浅层特征concat(已经过conv 1x1 压缩到32或48)，再经过conv 3x3(2个),最后再双线性上采样到原图尺寸得到结果.
- 将backbone替换成Xception-65.(较原始版本的Xception做了略微改动:将其中max pooling都改成Conv 3x3/2,middle flow中resblock重复次数由8遍增大到16遍, exit flow 倒数第二层增加重复一遍). 在ASPP和decoder涉及到的卷积也都改成了depthwise seperatable convolution,且在depthwise conv3x3和 pointwise conv1x1之间有BN和ReLU.
- 训练时显存吃紧且保证足够大的batchsize以更新BN统计信息比较重要,可通过多保留一个pooling或stride,减少dialated rate来减少预测的feature map输出(16x).推理时因dialated Conv权重共用,可以直接增大输出尺寸(8x),效果可能比8x训练更好 (显存有限导致BN有问题).

### others

- U-Net
  
    `Ronneberger, Olaf , P. Fischer , and T. Brox . "U-Net: Convolutional Networks for Biomedical Image Segmentation." arXiv:1505.04597.`

使用encoder-decoder结构,encoder中将空间分辨率逐渐下采样,decoder中将空间分辨率逐渐上采样，并且与encoder中相同尺寸的特征进行融合,构成了U字型的网络结构.
- Global Convolution Network
  
    `Peng, Chao , et al. "Large Kernel Matters —— Improve Semantic Segmentation by Global Convolutional Network." CVPR 2017.`

使用大卷积核增大特征的有效感受野，从而增强特征的语义信息.实现大卷积核的方式是用Conv 1xk+Conv kx1并联Conv kx1+Conv 1xk.将含大卷积核的模块嵌入在FCN中
- BiSeNet
  
    `Yu, Changqian , et al. "BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation." ECCV 2018.`

显式使用两分支网络来分辨编码空间语义(spatial context)和精细度(spatial detail)来解决语义分割两大矛盾.空间语义分支使用较少channel,较深的网络快速降采样(包括global pooling)来获得大感受和语义;特征精细度分支使用较多channel,较浅的网络保留丰富空间信息(实际上可只使用几个卷积层). 融合两支时使用concat+ conv(+SE)的方式得到结果.
- EfficientFCN
  
  `Jianbo Liu, Junjun He, Jiawei Zhang, et.al. EfficientFCN: Holistically-guided Decoding for Semantic Segmentation, ECCV 2020`

使用深层特征来引导编码浅层高分辨率的特征,绕开了上采样和空洞卷积,缓解空间精细和高层语义之间的矛盾.
1. 对8x,16x,32x三个stage的特征进行融合:统一bilinear resize到8x和32x后concat;
2. 对32x的深层次语义特征进行spatial attention得到codebook: 使用conv1x1分别得到$K \in \mathbb{R}^{n \times h/32 \times w/32}$,  $V \in \mathbb{R}^{1024 \times h/32 \times w/32}$ Q就直接使用单位张量, 进行attention操作$softmax_{w,h}(KQ)^TV$ 得到 $coodbook \in \mathbb{R}^{n \times 1024}$
3. 使用codebook引导高分辨率特征: 使用8x的高分辨率特征conv1x1得到 $n \times h/8 \times w/8$ 与 codebook进行转置相乘得到精细化特征(形状为$1024 \times h/8 \times w/8$, codebookh和channel一一对应scale)，再与8x高分辨率特征concat
   
在3中得到的featuremap进一步conv1x1、上采样后得到语义分割.

- 等等一众在性能和效率上引入小设计的方法和模型，个人感觉更多是architecture engineering,没有很多insight

## instance segmentaion

### FCIS
  
  `Yi Li, Haozhi Qi, Jifeng Dai, Xiangyang Jin Yichen Wei. Fully Convolutional Instance-aware Semantic Segmentation. CVPR 2017`

- 跟R-FCN是姊妹篇,思路相同,均利用图中同一位置在不同实体中相对位置不同,设计了position sensitive map将每个ROI分成kxk网格安排在feature map不同channel上,每个网格关注实例不同的相对位置.实例分割中区分了两类像素:(1)在物体框内但在物体mask外和(2)物体mask内.分别用(C+1)xKxK个通道的feature表示,分割头共2(C+1)xKxK个通道.
- 使用ImageNet pretrain的模型,利用dialated conv将最后一个stage feature map放大一倍后得到16降采样的feature，然后使用Conv 1x1压缩一下通道,然后Conv 1x1得到上述2(C+1)xKxK个通道的position sensitive map.另外与R-FCN相同,还有目标框回归坐标结果4xKxK个通道的特征.
- 在RPN的proposal上用position sensitive pooling(直接复制黏贴归属KxK网格中那一个网格的feature,不需要做pooling)后,表示同一类别实例内外的两个特征softmax得到分割结果,两个特征之间eltwise max后ave pooling,类别间softmax得到实例类别.物体框回归系数与R-FCN处理相同.
- 推理时物体框回归的结果和RPN输出一起用作ROI(各300个),进行NMS后Position Sensitive pooling得到类别和分割结果,position sensitive map后没有额外的计算,是全卷积的.

### Mask RCNN

`Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick "Mask r-cnn". ICCV 2017`
非常经典且成功的detect then segment框架,其要点已经在本公众号另一篇`object detection 的新进展`中`使用分割信息做多任务`详述.作为两阶段分割器,Mask RCNN在性能上是很强的benchmark.针对该框架后续有一些改进如下

`Shu Liu, Lu Qi, Haifang Qin, Jianping Shi, Jiaya Jia. Path Aggregation Networks. CVPR 2018`
- 在Mask RCNN的FPN后再增加与之对等的bottom-up neck(N2->N3->N4->N5),缩短了浅层特征到高层level之前的距离.例如本来stage III 8x 降采样的特征需要经过主干网络的stage IV,V才能到32x的检测头,在FPN后加入bottom-up后,因为有lateral connection,从stage III 8x 到现在32x的检测头有一条捷径,也可作为一种深度监督的方式存在. 增加的neck与FPN结构相似:每个level经过Conv3x3/2后与FPN下一个level相加,再经过conv 3x3得到下一个level,且最开始的level就是FPN最小的level.
- 针对原FPN根据目标尺寸分配在不同level上这种人为规定的方式可能不尽合理,改成了多尺度融合,让模型根据实际特征决定采用来自哪一个level的特征.每个level都做ROIAlign后,融合得到检测结果和分割结果.融合均放在原检测头第一层之后,即在检测头上,ROIAlign后首先经过每个level各自的FC,eltwise max后在FC出类别和目标框,在分割头上，经过1个每个level各自的conv后eltwise max,再经过3个Conv 3x3和deConv.
- 分割头原先仅由一支FCN组成.FCN注重局部感受,现增加一支并行的支路预测类别不可知的二分类mask,含有且仅含1个FC,一方面增强全局感受，另一方面尽量不破坏空间特性.特别地,在分割头的第三个Conv后分出一支Conv3x3 Conv3x3 FC(784) reshape(28x28)后,然后使用eltwise add对应类别的mask得到最终instance mask.

`Huang, Z. , Huang, L. , Gong, Y. , Huang, C. , and Wang, X.. Mask scoring r-cnn. CVPR 2019`
- Mask RCNN中mask的置信度由分类分支给出,一方面分类置信度本身是检测器框置信度,置信度高的框质量不一定好(这在检测器的设计中已经被广泛讨论和改进),另一方面即使框的质量好,mask质量也不一定好,mask的质量需要更精细地描述。因此在ROIAlign后增加一支MaskIOU分支,最终mask置信度由分类置信度和MaskIOU相乘得到.
- ROIAlign和预测的mask(经max pooling 2x2/2) concat后经4个conv 3x3(最后一个stride=2，然后ave pooling)后进过3个FC(1024,1024,C)回归MaskIOU. 优化时采用L2 loss且仅优化对应类别.

### YOLACT

`Bolya, D. , Zhou, C. , Xiao, F. , & Lee, Y. J.. Yolact: real-time instance segmentation. ICCV 2019`
`Bolya, D. , Zhou, C. , Xiao, F. , & Lee, Y. J. YOLACT++: Better Real-time Instance Segmentation,arxiv.1912.06218`

- 当遵循两阶段检测器改造成单阶段检测器的思路,只把第二阶段ROI pooling之后部分拿掉,在第一阶段直接出结果,直接在单阶段检测器头上使用FC(Conv 1x1)出mask，破坏了mask的空间特性,效果不好.因此yolact仍使用卷积出mask,检测头出mask的系数.每个mask可作为这些系数的基,最终的实例mask是系数对这些基的线性叠加.
- mask生成:挑选语义信息最丰富的层(如FPN 的P3)上加FCN,文中称protonet:
  
      Conv 3x3 256
      Conv 3x3 256
      Conv 3x3 256
      upsample
      Conv 3x3 256
      Conv 1x1 K
- 系数生成:检测头上多一支k个系数的预测,使用tanh激活.(检测模型检测头原有C+1类别支路和回归系数支路)
- 最终在实例mask生成后使用物体框crop出来.这部分主要是针对小目标所必须的.
- 一些效率上的考虑:
  - RetinaNet使用平行的4x Conv 3x3+Conv 1x1,yolact将这部分改成共享一个Conv 3x3，然后用三支Conv 3x3出类别,回归系数和mask系数.
  - 加速版本NMS. 使用IOU矩阵的上三角部分,根据最高置信度的框抑制掉该列所有其他框.实际上存在某些应被更高置信度框抑制掉了的框,去抑制了一些本该保留下来的框,但几乎不影响性能.这部分本来只能按列循环,跳过被抑制了列，现在直接并行做掉了,加速了数十毫秒.
- 一些性能上的考虑:
  - 在protonet输入层并行地增加一层卷积层,用于语义分割的多任务loss.因在将实例分割转为语义分割标签时一个像素可分属不同类别,使用sigmoid loss. 
  - 加mask scoring分支.在mask后再加一小FCN+global pooling得到每类的IOU.然后使用预测IOU修正置信度 (yolact++)
  -  适当将一些卷积替换成deformable conv (yolact++)
  - 增加anchor scale的个数 (yolact++)

### PolarMask

  `Xie, Enze, et al. "PolarMask: Single Shot Instance Segmentation with Polar Representation." CVPR 2020`

- 将mask在极坐标上表示,即中心点+n个轮廓上等角度分布的点与中心点的距离. 是anchor free检测器的推广,anchor free检测器仅回归中心点到上下左右四个点的距离.
  
- 框架基于FCOS,中心点是实例质心坐标附近9~16个像素,比物体框中心点更佳,因为物体框中心点更容易在物体外部.使用focal loss优化. 分类分支也输出centerness预测分支,定义为$\sqrt{\frac{min_{i=1..n}(d_i)}{max_{i=1..n}(d_i)}}$使中心区域具有更大概率. n个距离采用IOU loss优化. 把坐标原点放在GT质心,则IOU=$\frac{\int {min(d_{GT},d_{pred})^2d\theta }}{\int {max(d_{GT},d_{pred})^2d\theta}}$,离散化后为IOU=$ \frac{\Sigma_i{min(d_{GT},d_{pred})^2}}{\Sigma_i{max(d_{GT},d_{pred})^2}}$, 看成分类问题使用BCE优化.因IOU真值应为1，因此loss为$-log{IOU}$. 使用时把平方扔掉几乎不影响效果. IOU loss将所有距离统一考虑,且中心点分类和回归自然是平衡的,比L1之类的回归loss效果好很多.
- 真值生成:使用cv2.findContour找到物体轮廓,分别计算每一点距质心点的距离和角度,当质心点在物体外面导致某些角度射线无法与物体轮廓相交时这些角度的距离强制为eps，实际上强制将质心点放入了预测mask内部。这种做法上限AP仍非常高，不影响表达.

### Blendmask
    
  `Hao Chen, et al. BlendMask: Top-Down Meets Bottom-Up for Instance Segmentation, CVPR 2020`

- 是YOLACT的扩展.YOLACT在检测头上预测系数,并把实例分割结果看成是系数与基mask的线性叠加,忽略了每个基空间上对不同实例的不同贡献，会造成实例靠近时难以区分,退化成语义分割.借鉴了FCIS中position sensitive的思想,将系数建模成具有空间分布的feature map,然后与基mask eltwise prod后相加,可区分基mask在空间上不同位置的贡献.
- 基mask由称为bottom module的分支产生,可以是FPN中P3和P5按照deeplabv3+的decoder结构,也可以是YOLACT protonet，生成k个通道的基mask.检测结果在基mask上做ROIAlign后得到kxRxR的feature
- 检测头增加预测kMM通道的系数,reshape成kxMxM后沿k轴softmax,最近邻resize到kxRxR后,与ROIAlign后的基mask eltwise prod并叠加，得到实例分割logit.
- 由于区分了基mask的不同空间上的贡献,基的个数可以大幅减少,直至到1,一般取4. 为保证分割结果精细,R需要大一些可以取到56(mask RCNN该值为28), M对应position sensitive的网格数不需要很大,取到14足够.
- 当M=1时,BlendMask退化成YOLACT,当kMM通道的系数为onehot时,BlendMask退化成FCIS,即此时基mask就是position sensitive map,且$k=M^2$.

### SOLO

  `Wang, X., Kong, T. , Shen, C. , Jiang, Y. , and Li, L. Solo: segmenting objects by locations. arXiv:1912.04488v2`

- 语义分割能区分像素类别但无法区分不同的实例.若将语义分割结果扩展成多通道,不同通道预测不同位置的分割,那么实例就可通过位置得以区分.与anchor free类似地,为避免相同目标中心点落在同一位置区域，在FPN多尺度特征上分配不同尺寸的目标.
- 分割头为两支，一支预测类别(CxSxS)，一支预测类别不可知mask(S^2xHxW).类别分支类似于anchor free检测器的中心点预测,网格没必要很精细,在FPN输入时空间上先adaptive pooling或bilinear interpolation到SxS,FPN上预测小目标的分支网格密即S大一些,大目标的分支网格粗即S小一些.预测mask的分支保留空间尺寸,最后upsample到原图大小,为使mask分支学到绝对位置,在FPN特征上再追加x,y两通道归一化坐标[-1,1]一起输入,这个做法可直接涨点2+ AP.
- 在真值分配时,分类分支与anchor free检测器中心点做法类似,在中心坐标$[c_x±\sigma w,c_y±\sigma h]$中间为正样本,采用focalloss优化.mask分支采用Dice loss优化有正样本网格对应的通道(比focal loss效果好很多).Dice loss = $1-\frac{2\Sigma_{x,y}{p_{x,y}\times q_{x,y}}}{\Sigma_{x,y}{p_{x,y}^2}+\Sigma_{x,y}{q_{x,y}^2}}$,p,q为空间上的像素.
- 后处理仅包含mask NMS.
- 分割分支共有$S^2$个通道比较厚,可以分解宽高两个维度,转而预测2S个通道,分别负责行和列,最后i行j列的位置由负责i行和负责j列的mask eltwise prod得到,这种方式称为decouple SOLO,效果基本不变.
  