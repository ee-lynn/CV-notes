# Video Recognition 
&nbsp;   　　　　　　　　　　　　　sqlu@zju.edu.cn
 
在CIFAR-10,CIFAR-100,ImageNet等数据集上训练的静止图像分类深度神经网络模型进步飞速，且已具有划时代的成果(《Nerual Network archetecture》已详述),但在视频识别和分类上却进展缓慢。除去将2D特征扩展成3D特征的手工设计传统方法,目前深度学习在该领域主要的技术框架可以分成３类,分别是(1)采用3D卷积神经网络;(2)融合手工设计特征的双流网络模型;(3)采用卷积神经网络提取视频帧语义信息,采用循环神经网络编码图像序列的模型.
大型视频分类数据集有: 
(1)sports-1M:487类体育运动,1M的数据量,视频较长,由于是若监督方式标注的,其中含有错误的标签，也有虽然主题符合，但内容不合适的内容,比如记分牌,比赛解说等)
(2)UCF-101:13320数据量,含有101类,根据内容主题分为5大类
(3)HMDB-51:6800数据量,含有51类
(4)Kinetics:数据量240000,400类，根据运动的主体分为3大类别
本篇以方法在视频分类和识别领域深度学习的进展.
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

## 双流模型
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

### 端到端学习的光流
    Y Zhu, Z. Lan, S. Newsam, A. Hauptmann. Hidden Two-Stream Convolutional Networks for Action Recognition. ACCV 2018

### 双流两支具有不同结构且有交互    
    Feichtenhofer C., Pinz A, Richard P .Spatiotemporal Residual Networks for Video Action Recognition. NIPS 2016
    Feichtenhofer C., Fan  H., Malik  J., He  K.. SlowFast Networks for Video Recognition. ArXiv.1812.03982


## CNN+LSTM

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
   - Time-Dmain Conv: 最后卷积层后面增加256x10(/5V)x3x3卷积再pooling
    训练时因参数共享,逐渐由短向长扩展
- LSTM策略的要点
    - CNN+5层隐变量512维LSTM,每帧最后LSTM都接softmax,按照时间线性加权计算loss,推理时也线性加权得分.
- 使用1fps的视频帧可长达120frame.再加上15fps采样的光流,与双流相同(此时L=15).
- 结论:实验证明CNN+LSTM比CNN+Pooling略好(基本相当,~1%)

    A. Diba, V Sharma, L. V. Gool. Deep Temporal Linear Encoding Networks . CVPR 2017 
    

## 3D卷积及其变种
为捕捉在时间维度上的特征,该类方案将传统图像分类的网络中2d卷积核扩张成3d，其中一维是时间维度。按照惯例,张量都表示为序列长度(L)x空间高(H)x空间宽(W)
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
    -  第一层用3D卷积将时间维度全部融合掉
    - 每个frame单独,最后在FC前全部融合掉
    -  先3D 后 2D
    - 先2D 后 3D
    - 纯3D
    - 将纯3D结构中3D结构分解成空间和时间,即kxkxk分解为1xkxk和kx1x1
    R(2+1)D结论为性能(1)<(2)<(5)~(4)<(3)<(6)，2D/3D混合方案中占比影响不大.
    S3D结论为(2)<(3)<(4)(5)<(6),且2D/3D混合方案中,3D卷积比例越大(参数越多),性能越好.
    个人认为S3D结论更靠谱一点.在空间感受野较小时捕捉运动特征会导致不同的物体特征矛盾或者有孔径问题.
- R(2+1)D方案即P3D串联变种,空间和时间中间通道数设计得使分解的卷积和3D卷参数相同,这就使得R(2+1)D比P3D空间卷积输出更厚宽.
- 时间维度的输入不是越长越好,有一个峰值(R(2+1)D实验得出是32帧)
- 网络均可以在光流和RGB图上分别训练后融合,进一步提升性能. S3D还利用了一种简化版本的SE模块(SE模块MLP不含隐层)进一步提升性能.

## attention
   
    X. Wang, R. Girshick, A. Gupta, K. He. Non-local Neural Networks for Video Classification CVPR 2018

## 附录:光流
###稀疏光流

###稠密光溜

###采用深度学习计算光流
    Fischer P., Dosovitskiyz A. , Ilgz E., et al, FlowNet: Learning Optical Flow with Convolutional Networks. ICCV 2015
    Ilg  E., Mayer N., Saikia T., et.al. FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks, CVPR 2017
    