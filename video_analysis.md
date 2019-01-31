# Video Recognition 
&nbsp;   　　　　　　　　　　　　　sqlu@zju.edu.cn
 
在CIFAR-10,CIFAR-100,ImageNet等数据集上训练的静止图像分类深度神经网络模型进步飞速，且已具有划时代的成果(《Nerual Network archetecture》已详述),但在视频识别和分类上却进展缓慢。除去将2D特征扩展成3D特征的手工设计传统方法,目前深度学习在该领域主要的技术框架可以分成3类,分别是(1)采用3D卷积神经网络;(2)融合手工设计编码运动特征描述符的双流2d网络模型;(3)扩展至整个视频尺度的编码和建模方案.当然这三类并非绝对互斥,类别之内也融入了其余类别的方案.例如3D CNN仍会使用光流输入进行ensemble,或者使用None-local组件扩展3D卷积在时域上的感受野;在长视频建模的框架下仍会采用双流基础模型;双流模型的变种也会融入3d卷积更好的融合两支特征,等等。

## 大型视频分类数据集: 
(1)sports-1M:487类体育运动,1M的数据量,视频较长,由于是若监督方式标注的,其中含有错误的标签，也有虽然主题符合，但内容不合适的内容,比如记分牌,比赛解说等)
(2)UCF-101:13320数据量,含有101类,根据内容主题分为5大类
(3)HMDB-51:6800数据量,含有51类
(4)Kinetics:数据量240000,400类，根据运动的主体分为3大类别
(5)someting-something:数据量108499,174类，具有强时间反向相关性,例如从左向右扔和从右向左扔,假装打开盒子等.
## 雏形
    Karpathy A , Toderici G , Shetty S , et al. Large-Scale Video Classification with Convolutional Neural Networks. CVPR. 2014.
- 尝试CNN来对大规模视频数据集进行分类的尝试.考虑到视频相比静态图片多了时间维度,需要比较在时间维度上不同计算方式的方案.
- 因拓展了时间维度计算量增加,做了两支前向:降采样一半和center crop一半在FC前concat
- 网络骨架： AlexNet-like
    
        Conv 11x11 /3 96
        LRN ReLU
        Max Pool 2x2 /2
        Conv 5x5 256
        LRN ReLU
        Max Pool 2x2 /2
        Conv 3x3 384 ReLU
        Conv 3x3 384 ReLU
        Conv 3x3 256 ReLU
        Max Pool 2x2 /2
        FC 4096 ReLU
        FC 4096 ReLU
        softmax
        
- 时间维度上不同的组合方式:
    - Early Fusion: 第一层卷积层后接Tx1x1,将T帧frame直接加权合并
    - Late Fusion:相隔T帧frame分别前向后在FC前融合
    - Slow Fusion: 第一层卷积层后接4(/2V)x1x1;第二层卷积后接2(/2V)x1x1;第三层卷积后接2(/2v)x1x1，此时时间维度上感受野为10.
- 推理方式:video中采样多个clip，crop或者flip预处理后再全部平均后得到video预测结果.结果显示Slow Fusion效果最佳.但比单帧静止图像分类效果提升非常有限.

## Two-Stream models及其变种
### 初探
    Simonyan K., Zisserman A.. Two-Stream Convolutional Networks for Action Recognition in Videos. NIPS. 2014

- 第一次提出使用光流作为附加模型输入，并将结果与基于RGB的模型推理结果融合.在本篇末尾附有光流的基本概念和采用深度学习来计算光流的方法.
    - spatial stream inputs random samples still video RGB frame(pretrained by ImageNet)
    - temporal stream inputs dense optical flow (2L channels)
    - 融合方式: averaged softmax score 或 Linear SVM based on L2 nomalized softmax scores.
- 两支结构均相同,也均为AlexNet-like(省略ReLU):
        
        Conv 7x7 /2 96 LRN
        Max Pool 3x3 /2
        Conv 5x5 /2 256 LRN
        Conv 3x3 512
        Conv 5x5 512
        Conv 5x5 512
        Max Pool 3x3 /2
        FC 4096 dropout
        FC 4096 dropout
        softmax

- 光流支路输入2L channel,分别是L个图相对计算的光流(减去均值去掉相机全局运动,离线计算并缩放至[0,255]用jpeg压缩),考虑了两种形成方式:
    - 直接将光流图concat 
    - 将光流图按照归集修正.即下一帧制定像素光流的值用光流预测该像素下一帧位置上的值,表示了同一位置上像素的运动轨迹上光流值。
    结果表明第一种效果略好.光流可以是t时刻往后L帧,也可以将t时刻放在中间,前后各L/2帧.光流图经过CNN特征提取后可以看成是基于光流手工设计特征的扩展.因光流数据少,在UCF-101和HMDB-51上多任务训练.

### ST-ResNet    
    Feichtenhofer C., Pinz A, Richard P .Spatiotemporal Residual Networks for Video Action Recognition. NIPS 2016
- 针对双流模型两支在浅层完全分离,设计了两支具有交互的结构,企图学习什么(apearance)做什么运动(motion).并且将双流模型在时间维度上扩展.
- 在ResNet框架下,通过实验发现信息在block之间传递,或者双向传递效果都不理想.本文采用的方法是将temporal stream中间信息加到spatial stream对应的residual支路中去,相加发生于每个stage(除去第一个stage)的第二个residual block.并且将bottleneck的1x1卷积膨胀为3x1x1卷积,膨胀方式与I3D相同.最后在空间的global ave pooling后在时间上采用5x1x1的max pooling(比ave pool效果好)
- 训练中为增加正则化效果，BN的方差均值计算采用的batchsize仅取为4.
- 整个ST-ResNet训练分成3个阶段:
    - (I) 按照传统方式分别训练spatial stream和temporal stream network
    - (II)输入5帧,temporal stride在[5,15]随机生成
    - (III) 将输入改成11帧,temporal stride在[1,15]随机生成，最后在时间维度上max pool后再ave pool
- 推理时采用全卷积的方式:空间采用全尺寸,并且输入25帧。最后在所有的时空维度上做平均.

### slowfast Net
    Feichtenhofer C., Fan  H., Malik  J., He  K.. SlowFast Networks for Video Recognition. ArXiv.1812.03982
- 在ST-ResNet基础上将temporal stream替换成temporal resolution很高的fastNet。为捕捉运动信息,采样率是spatial stream(这里是slowNet)的a倍(a可取8)，此支不需要编码空间信息,因此宽度可以很小(是slowNet的b倍,b可取1/8)
- slowNet可以是随意一个CNN(这里用ResNet实例化,在stage(IV)(V)处第一个bottleneck pointwise conv改成 3x1x1 conv,浅层不用temporal convolution的原因与2D/3D混合结构heavy top性能更好相同)
- fastNet的输入帧数是slowNet的a倍,宽度是b倍的几乎网络,差别是每个stage第一个pointwise都做temporal conv,且自始至终没有temporal stride.
- fastNet和slowNet采用lateral connection融合.fastNet的feature map向slowNet融合时因形状不兼容,可采取的措施是:
    - bC,aL,H,W [reshape->] abC,L,H,W ,采用concat融合,当ab=1时,还可以eltwise相加
    - bC,aL,H,W [temporal采样->] bC,L,H,W 采用concat融合
    - bC,aL,H,W [strided temporal conv,5x1x1->] bC,L,H,W 采用concat融合.  效果最优,但所有方式差不多
- 该模型难以使用ImageNet预训练模型,但文中实验说明在kinetics上from scratch和用ImageNet预训练最终效果几乎相同.

## long range temporal prediction
要对整个video建模,需要摆脱以往用dense sampling的图像序列(clips or snippets),而应当扩展至整个视频来考虑.

### CNN+pooling/CNN+LSTM
    Yue-Hei Ng, Joe, et al. Beyond Short Snippets: Deep Networks for Video Classification. CVPR 2015
    
- 视频都是采样clip后在video level平均,更需要一个直接在video level得到预测结果的方法.但video 是变长的,有两种方式来应对:
    - 单帧经过CNN后，pooling多帧(pooling不需要参数,CNN和FC均共享参数)
    - 单帧经过CNN后经过循环神经网络进行分类。
- pooling策略的要点(按照实验结果从好到坏排序,骨架都是AlexNet-like)
    使用Max pooling,梯度更稀疏,比Ave pooling 更好
    - Conv pooling: 在最后一层卷积后pooling,连接FC
    - local pooling: 仅是slow pooling的第一阶段,这样softmax更大.
    - slow pooling:pooling分两次在两个FC之间进行,第一阶段时间维上 10/5V
   - late pooling: 在FC后再pooling
   - Time-Domain Conv: 最后卷积层后面增加256x10(/5V)x3x3卷积再pooling
    训练时因参数共享,逐渐由短向长扩展
- LSTM策略的要点
    - CNN+5层隐变量512维LSTM,每帧最后LSTM都接softmax,按照时间线性加权计算loss,推理时也线性加权得分.
- 使用1fps的视频帧可长达120frame.再加上15fps采样的光流,与双流相同(此时L=15).
- 结论:实验证明CNN+LSTM比CNN+Pooling略好(基本相当,~1%)

### TSN
    Wang L , Xiong Y , Wang Z , et al. Temporal Segment Networks: Towards Good Practices for Deep Action Recognition. ECCV 2016.
- TSN将整个视频分成k(文中取3)部分,每一部分采样一个clip用2d CNN(沿用two stream model的方法)推理后经过一致性函数(consensus function)融合这k个结果(可选为max pooling, ave pooling,weighted sum，其中ave pooling最优,但也没明显优势),最后给出每个类别的概率(softmax)
- 在TSN框架下训练CNN,CNN在k个部分中共享权值.
- two stream的基础上探索了增加帧差(噪声较大，最后弃用),warped光流(首先使用透视变换消除相机运动影响)模态输入后一起ensemble
- 在训练中提出在ImageNet预训练模型因更新第一层BN层均值方差,在temporal network这一支将预训练模型中第一层卷积参数在RGB通道上在输入总通道上平均.
- 推理时,在时间上在每个部分采样多次,在空间上多次crop,flip,使用多模态输入最后在logits上全部平均后得到类别概率.

### TLE
    A. Diba, V Sharma, L. V. Gool. Deep Temporal Linear Encoding Networks. CVPR 2017 
- 针对TSN框架下融合每个部分特征的方式并形成描述符而提出temporal linear encoding方法.被融合的特征由2D/3D CNN提取,均是最后一层卷积层后ReLU的输出.
- 每个部分i特征为S_i,聚合得到特征X,再经过encoding得到描述符.
    - 聚合的方式有
        - eltwise sum
        - eltwise max
        - eltwise multiplication
    - encoding的方式有
        - FC
        - bilinear model: `X.view(h*w,c),y = (X.T*X).view(-1), y=sign(y)*sqrt(y),y=normalize(type="L2")`特征做矩阵外积,每2个channel做内积后作为编码特征 (文中说使用了tensor sketch的方法给最终的特征降维,不需要直接计算维数很高的外积)
- 训练分成两阶段:
   - (I) finetune softmax(固定CNN和encoding参数)
   - (II) finetune全部
  
### TRN  
    Zhou B , Andonian A , Torralba A . Temporal Relational Reasoning in Videos. ECCV 2018.
- 按照时间顺序在尺度推理:
$$
T_{d} = h_{\phi}^d(\Sigma_{i_{1}<i_{2}...<i_{d}}g_{\theta}^d(f_{i1},f_{i2}...f_{id}))
$$
其中
$$f_{i}$$
为video(V)按照时间顺序采样的frame根据2d CNN提取的特征
- 多尺度推理:
$$
MT_{N} = \Sigma_{i=2}^NT_{i}(V)
$$
$$h_{\phi}^d,g_{\theta}^d$$
均利用MLP参数化
- 训练和推理时为提高效率,每个frame只计算CNN前向一次,将提取的特征保存在队列中以供采样和配对.
- 实验中每个尺度选了k(=3)组,
  $$g_{\phi}$$
  为2层256个单元的MLP,
  $$h_{\theta}$$
  为单层MLP,输出个数为类别个数. 多尺度中选了N=8.

### ECO
    Zolfaghari M , Singh K , Brox T . ECO: Efficient Convolutional Network for Online Video Understanding. ECCV 2018.
- 为解决TSN只在temporal维度做late fusion，难以有效发觉时域相关信息,提出在TSN框架下将其consensus function改成3D CNN.其中frame用Inception-V1提取语义信息(直至stage III)，后接stage III后的3D-ResNet-18(空间尺度为18x18),称为ECO-Lite
- 考虑到有些内容只看静态图片就能判断,与3D-ResNet-18并行的加入Inception-V1剩余部分(对每个segment产生的feature map做ave pooling,即对输入做temporal global ave pooling),最后将两支输出concat后连接softmax,称为ECO-Full.
- 此文更关注模型效率,对于online应用,因视频不是一开始就全部可见,提出了一种采用sliding window的online预测方式:维护队列Q，每输入N帧,在Q中采样50%个输入c采样50%组成新的队列送入ECO。预测结果做时域指数滑动平均(相当于采样和预测都是EMA形式，靠近此刻越近权重越大)
- 
### TSM
    Lin J , Gan C , Han S . Temporal Shift Module for Efficient Video Understanding. arXiv:1811.08383v1.
- 使用2D CNN达到了3D CNN的效果,其做法与将group convolution与channel shuffle联用沟通channel之间信息有异曲同工之妙.只不过此时是将channel在temporal维度移动,使2D卷积在channel维计算到了别的时间信息.
- 为达到temporal kernel=3的卷积效果(单层感受野为3),使feature map中一定比例channel前移一步时间,一定比例后移一步时间
- 最优成分:移动的比例为1/4,temporal pad为zero pad，TSM加在residual支路中(而不是block之间).
- TSM计算量与2d CNN完全相同,只需要将视频分成N份(8/16)，每分钟采样frame,就可以比相应的TSN提高两位数的准确率.
  
## 3D卷积及其变种
为捕捉在时间维度上的特征,该类方案将传统图像分类的网络中2d卷积核扩张成3d，其中一维是时间维度。按照惯例,张量都表示为序列长度(L)x空间高(H)x空间宽(W). 最后推理视频类别时，采用多次采样最后融合的方式计算score.
### C3D

    Tran D , Bourdev L , Fergus R , et al. Learning Spatiotemporal Features with 3D Convolutional Networks. ICCV 2015.
    
- 将时空维度同等对待，采用统一的3D卷积核处理,企图得到VGG那样的视频描述符.
- 空间上沿用3x3尺寸,时间维度上尺寸为1,3,5,7和可变长(逐渐变大和变小)，实验证明均匀的3x3x3卷积核尺寸最优.
- 因输入分辨率直接影响庞大的FC参数个数,在性能和模型大小之间trade off为输入112x112
- C3D 结构(VGG-like,均为same padding,pooling stride与尺寸相同)

        Conv 3x3x3 64
        Max pooling 1x2x2
        Conv 3x3x3 128
        Max pooling 2x2x2
        Conv 3x3x3 256
        Conv 3x3x3 256
        Max pooling 2x2x2
        Conv 3x3x3 512
        Conv 3x3x3 512
        Max pooling 2x2x2
        Conv 3x3x3 512
        Conv 3x3x3 512
        Max pooling 2x2x2
        FC
        FC
        softmax

### I3D

    Carreira J., Zisserman. A.. Quo Vadis,Action Recognition? A New Model and the Kinetics Dataset . CVPR 2017

- 为解决C3D训练困难的问题,Inflated 3D模型将2D模型转换成3D CNN模型:若将图像看成静止的序列,那么2D CNN中卷积核在每一帧提取的特征相同，再在时间维度尺寸上平均,得到的结果与2DCNN输出完全相同.按照这种思想，可以充分利用预训练的2dCNN模型. 即原卷积核参数为blob(假设blob.shape = 512x3x3,需要在时间维度扩张为3),  blob.unsqueenze(1).repeat(1,3,1,1)/3即为inflated 的3D卷积核.
- 时间维度上的尺寸一般考虑为与空间相同,卷积核为cubic.需要详细确定的是时间维度的stride,一般而言，空间感受野较小时,时间stride不宜太多否则会导致不同的物体特征之间冲突.网络早期temporal pooling stride均为1.
- 在Inception-V1上改造,前两个Max pooling temporal维度尺寸和stride均为1
- I3D也可直接分光流图，融合RGB和光流图可以进一步提高性能.
    
### P3D/(2+1)D/separatable 3D
**P3D**:

    Qiu Z , Yao T , Mei T . Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks. ICCV 2017. 
**R(2+1)D**:

    Xie S , Sun C , Huang J , et al. Rethinking Spatiotemporal Feature Learning: Speed-Accuracy Trade-offs in Video Classification. ECCV 2018
**S3D**:
    
    Tran D , Wang H , Torresani L , et al. A Closer Look at Spatiotemporal Convolutions for Action Recognition. CVPR. 2018.
 
- 针对C3D模型参数众多且难以训练，P3D提出了一种Pseudo-3D block.将3x3x3卷积核分解为1x3x3和3x1x1，且1x3x3卷积核可以采用2D模型对应的卷积核参数初始化，可以利用静止图像预训练的模型.
- 将Pseudo-3D block替代进ResNet时,具体的形式可能有3种:
    - 1x3x3和3x1x1串联;
    - 1x3x3和3x1x1并联;
    - 1x3x3和3x1x1串联,但1x3x3同时skip 3x1x1输出.
- 仍保留bottleneck的ponitwise convolution，此时增加时间维为1x1x1.将以上三种变种依次顺序代入ResNet 3x3卷积中,结果将最优(最直观的串联结构效果已经不错，是三种变种中最好的)

- 考虑各种CNN变种,处理图片序列的CNN可以有6种形式(最后都时空维度上global ave pooling):
    - 第一层用3D卷积将时间维度全部融合掉
    - 每个frame单独,最后在FC前全部融合掉
    - 先3D 后 2D
    - 先2D 后 3D
    - 纯3D
    - 将纯3D结构中3D结构分解成空间和时间,即kxkxk分解为1xkxk和kx1x1
    R(2+1)D结论为性能(1)<(2)<(5)~(4)<(3)<(6)，2D/3D混合方案中占比影响不大.
    S3D结论为(2)<(3)<(4)(5)<(6),且2D/3D混合方案中,3D卷积比例越大(参数越多),性能越好.
    个人认为S3D结论更靠谱一点.在空间感受野较小时捕捉运动特征会导致不同的物体特征矛盾或者有孔径问题.
- R(2+1)D方案即P3D串联变种,空间和时间中间通道数设计得使分解的卷积和3D卷参数相同,这就使得R(2+1)D比P3D空间卷积输出更厚宽.
- 时间维度的输入不是越长越好,有一个峰值(R(2+1)D实验得出是32帧)
- 网络均可以在光流和RGB图上分别训练后融合,进一步提升性能. S3D还利用了一种简化版本的SE模块(SE模块MLP不含隐层)进一步提升性能.

### 提升组件: None local module
   
    X. Wang, R. Girshick, A. Gupta, K. He. Non-local Neural Networks for Video Classification CVPR 2018
- 以往卷积操作均是局部操作符,通过逐层堆叠增加感受野的才能捕获全局信息.这里直接对全局信息建模.将一个特征看成一个时空点(具有C维),它是全部特征点的加权平均，即某个是空间xi经过none-local模块变换成yi:
  $$ y_i = 1/C(x)\Sigma_jf(x_i,x_j)g(x_j)) $$
  其中C(x)是归一化系数,g(x_j)是pointwise convolution,f(x_i,x_j)可以是:
    - $$ \exp(x_i^Tx_j) [gaussion]$$
    - $$ \exp(x_i^T W_\theta^T W_\phi x_j) [embeded gaussion] $$
    - $$ x_i^T W_\theta^T W_\phi x_j [dot product]$$
    - $$ ReLU(W_f^Tconcat(W_\thetax_i, W_\phix_j)) [concat] $$
  以上结果都差不多,第二种类似于softmax,即NLP中的self attention.
- 为减少计算量, W_\theta,W_\phi,g(x_j)都使通道数减半,且在空间上降采样.为帮助训练和利用预训练模型，None-local模块采用residual形式,即最终输出为:
  $$z_i=W_zy_i+x_i $$
  其中W_z恢复通道数,在W_z增加BN层,scale初始化为0，保证起始阶段None-local输出与预训练模型输出完全一致.
- 实验表明:None-local module对加在网络中的位置不敏感,是视频和静止图像任务均有稳定提升作用.

    