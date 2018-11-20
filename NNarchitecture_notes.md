# deep convolutional nerual network architecture
 &nbsp;　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　       by sqlu@zju.edu.cn
## overview

CNN在计算机视觉任务中具有越来越举足轻重的作用，不光光在图片分类,目标检测,语义分割等基础任务上一个性能优良的CNN主干网架能极大提升效果，各种主干网架中基础结构和操作还广泛存在于图片风格转换,行为分析,超分辨率,以图搜图等高级任务中.网架结构的设计也逐渐由人工设计发展至算法设计，但即使算法搜索出来的结构，其搜索空间也是人为设计的基本组件构成.
本文综述自2012年CNN在ILSVRC夺冠以来在结构上的发展,包括

- 越来越准确的模型
    - AlexNet
    - VGG
    - Inception家族(V1-V4,以及autoML产物NasNet)
    - ResNet及其各种变种(Wide ResNet,ResNeXT,DenseNet,DualpathNet);
- 适合移动端部署的高效小模型
    - SqueezeNet
    - MobileNet家族(V1/V2)以及autoML产物MnasNet
    - shuffleNet 家族(V1-V2);
- CNN中的提升组件:self-attention机制
    - spatial attention: spatial transform
    - channel attention:SE
    - serial decoupled spatial-channel attention:CBAM
    - coupled spatial-channel attention:Residual attention.

## powerful models

随着深度学习主干网络结构的变迁,作为视觉两大基础任务的分类和目标检测在大型数据集上(ImageNet, coco)的指标被逐渐推高.这里以一些明星网络结构为例,罗列其结构的要点。

### AlexNet

    Krizhevsky, Alex, I. Sutskever, and G. E. Hinton. "ImageNet classification with deep convolutional neural networks." International Conference on Neural Information Processing Systems Curran Associates Inc. 2012:1097-1105.

AlexNet是Hinton和他的学生Alex Krizhevsky设计的，吹响了CNN大规模进军计算机视觉的号角，其成功的因素可以归纳为:

- 激活函数均使用了ReLU(Rectfier Linear Unit)
- data augmentaion: 
horizontal flipping, random crop[resize 256->crop 224], PCA jittering
- GPU training
- dropout
- weight decay
- 网络结构(全部为same padding)

        Conv      11x11/4 96          ReLU
        Conv      5x5     256 group=2 ReLU
        Max pool  3x3/2      
        Conv      3x3     384 group=2 ReLU
        Max pool  5x5/2       
        Conv      3x3     384 group=2 ReLU
        Conv      3x3     256 group=2 ReLU
        Max pool  3x3/2    
        FC                4096        ReLU 
        dropout(0.5)
        FC                4096        ReLU
        dropout(0.5)  
        FC 
        softmax

### VGG
    Simonyan K, Zisserman A. "Very Deep Convolutional Networks for Large-Scale Image Recognition". arXiv:1409.1556, 2014.
    
 VGG网络结构规则整齐.其成功的因素归纳为
 
 - 3x3的卷积核代替大的卷积核
 事实上，两层3x3卷积跟一层5x5卷积具有等价感受野,但参数更少,且其中可以插入激活函数添加非线性.
 - multi-scale training  resize 到不同大小的图片后再crop
 - 网络结构，以最常见的VGG-16为例
 
        ----------------------------stage I x2
        Conv 3x3 64    ReLU
        -----------------------------
        maxpool 2x2/2
        -----------------------------stage II x2
        Conv 3x3 128  ReLU
        -----------------------------
        max pool 2x2/2
        -----------------------------stage III x3
        Conv 3x3 256  ReLU
        -----------------------------
        max pool 2x2/2
        -----------------------------stage III x3
        Conv 3x3 512  ReLU
        -----------------------------
        max pool  2x2/2
         -----------------------------stage IV x3
        Conv 3x3 512  ReLU    
        ---------------------------
        max pool 2x2/2
        FC  4096 dropout(0.5)
        FC  4096 dropout(0.5)
        FC  
        softmax
  
### GoogLeNet and Inception family

Google提出GoogLeNet后,又相继提出了Inception V2,V3,V4.他们的共同点都是使用了Inception module. 这种module是一种多路拼接结构。加入了Batch Normalization使训练更加稳定，成为训练深度网络的必备元素。将多路拼接以group convolution的观点重新阐释并推向depthwise seperatable convolution，即可引出Xception.Inception module也是后续神经网络架构自动搜索技术的雏形结构,使用autoML后，可引出NasNet.

#### GoogLeNet & Inception V1
    
    Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich. "Going Deeper with Convolutions" arXiv:1409.4842v1
    Sergey Ioffe, Christian Szegedy. "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift". arXiv:1502.03167v3

- intuition: 来自神经科学的hebbian theory.两个神经元或者神经元系统，如果总是同时兴奋，就会形成一种‘组合’，其中一个神经元的兴奋会促进另一个的兴奋。比如狗看到肉会流口水，反复刺激后，脑中识别肉的神经元会和掌管唾液分泌的神经元会相互促进，“缠绕”在一起，以后再看到肉就会更快流出口水 
- 为了减少计算量，Inception module在3x3,5x5前使用１x1卷积来降低输入通道数. max pooling后的1x1卷积目的在于组合不同通道, 结构如下(卷积层后自带ReLU)
        
        Conv 1x1 n1 Conv 1x1 n31 Conv 1x1 n51 Max pool 3x3
                    Conv 3x3 n3  Conv 5x5 n5  Conv 1x1 np
        Concat
        
- 为了使梯度更有效地传播，训练时在网络更浅处插入了两个辅助输出分支, 分别以0.3的权重比例辅助传递梯度. 该举措在后续被证明在训练初期没什么特别大的改善作用.仅在收敛后期略有作用.
- 预处理和数据增强.　图片只做了减均值预处理. 随机宽高比3/4~4/3, crop 8%~100%，随机的resize插值方法,等。测试时使用144 crop: resize短边到256,288,320,352; 使用左中右/上中下的正方形; 四个224尺寸的角和直接resize到224;镜像.　这样可以得到144个结果，最后做平均.
- 网络结构(卷积层后自带ReLU,全部都same padding)
    - GoogLeNet
    
            Conv 7x7 /2 64
            max pool 3x3 /2
            Conv 1x1 64
            Conv 3x3 192
            max pool 3x3 /2
            inception(3a) n1=64 n31=96 n51=16 np=32 n3=128 n5=32
            inception(3b) n1=128 n31=128 n51=32 np=64 n3=192 n5=96
            max pool 3x3 /2
            inception(4a) n1=192 n31=96 n51=16 np=64 n3=208 n5=48
            inception(4b) n1=160 n31=112 n51=24 np=64 n3=224 n5=64
            inception(4c) n1=128 n31=128 n51=24 np=64 n3=256 n5=64
            inception(4d) n1=112 n31=144 n51=32 np=64 n3=288 n5=64
            inception(4e) n1=256 n31=160 n51=32 np=128 n3=320 n5=128
            max pool 3x3 /2
            inception(5a) n1=256 n31=160 n51=32 np=128 n3=320 n5=128
            inception(5b) n1=384 n31=192 n51=48 np=128 n3=384 n5=128
            gloabl AVE pool
            dropout(0.4)
            FC 1000
            softmax
        
     - 辅助分支接在inception(4a)(4d)处
       
             Ave pool 5x5/3V
             Conv 1x1 128
             FC 1024
             dropout(0.7)
             FC 1000 
             softmax
             
      - Inception V1
      在ReLU前均插入了BatchNorm层，此后Conv-BatchNorm-ReLU成为CNN标配.将Inception module中5x5卷积改成两个3x3卷积, max pool 改成ave pool.
      在inception之间的max pool 用inception表示,其中3x3 conv stride=2, 内部的pooling仍为max pool。如增加的(3c)和修改的(4e).　inception 内部参数具有调整，现为
        
            inception(3a) n1=64 n31=64 n51=64 np=32 n3=64 n5=64
            inception(3b) n1=64 n31=64 n51=64 np=64 n3=96 n5=64
            inception(3c) n1=0 n31=128 n51=64 np=0 n3=160 n5=64 
            inception(4a) n1=224 n31=64 n51=96 np=128 n3=96 n5=96
            inception(4b) n1=192 n31=96 n51=96 np=128 n3=128 n5=96
            inception(4c) n1=160 n31=128 n51=128 np=128 n3=160 n5=128
            inception(4d) n1=96 n31=128 n51=160 np=128 n3=192 n5=160
            inception(4e) n1=0 n31=128 n51=192 np=0 n3=192 n5=192 
            inception(5a) n1=352 n31=192 n51=160 np=128 n3=320 n51=160
            inception(5b) n1=352 n31=192 n51=192 np=128 n3=320 n5=192

#### Inception V2 & V3

    Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna. "Rethinking the Inception Architecture for Computer Vision". arXiv:1512.00567v3
    
- inception(3)把大于3x3的卷积核全部用3x3卷积堆叠实现等效的感受野,减少计算量
       
        Conv 1x1  Conv 1x1  Conv 1x1  Max pool 3x3
                  Conv 3x3  Conv 3x3  Conv 1x1
                            Conv 3x3
        Concat
        
- 在网络的中间阶段将inception(4) nxn卷积分解成1xn,nx1卷积的堆叠,进一步减少计算量(这里n使用了7)

        Conv 1x1 Conv 1x1 Conv 1x1 Max pool 3x3
                 Conv 1x7 Conv 1x7 Conv 1x1
                 Conv 7x1 Conv 7x1
                 Conv 1x7
                 Conv 7x1
        Concat
        
- 用strided convolution+pool的concat代替单纯的pool或者conv(　feature　map减少时,单独用pool做不到通道数增加,扩大的strided　conv计算量又太大)
- 将Inception(5)改的更宽了，将1x3,3x1堆叠改成并联结构
       
        Conv 1x1         Conv 1x1         Conv 1x1       Max pool 3x3
                      Conv 1x3   Conv 3x1  Conv 3x3      Conv 1x1
                                       Conv 1x3   Conv 3x1
        Concat
        
- 输入分辨率改成299(初始主干部分和convolution　stride时候都不pad,主干部分Conv 1x1, Con 3x3, Max pool,改成两个Conv 3x3, Conv 3x3 /2, Conv 3x3)
以上结构称为**Inception V2**

- 将最开始的7x7卷积也改成3层3x3卷积堆叠(conv 3x3 /2; conv 3x3; conv  3x3 padded),使用RMSProp训练，添加先验为均匀分布的交叉熵损失(权重0.1,作为label　smoothing)　称为**Inception V3**

#### Inception V4 & Inception-ResNet V1 & V2

    Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi. "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning".  arXiv:1602.07261v2 
    
- 发布了３个模型，分别为Inception V4, Inception-ResNet V1 & V2, 其中Inception-ResNet V1 与Inception V3计算量相当,　inception-ResNet V2 与Inception V4计算量相当
- 结果显示,Residual learning 可以让模型更快收敛,但是最终性能还是由模型容量决定.
- 在Residul 支路加一个0.1的恒定常数scale，可以稳定训练过程。
*notes* 文中提到因内存的限制,Inception-ResNet在residual之后eltwise add之后没加BatchNorm,其实这跟惯例相同.
- Inception-ResNet V1　网络结构

        stem                    见(1)
        5 x inception-resnet-A  见(7) k=48,l=64,m=384
        Redution-A              见(5) k=192,l=192,m=256,n=256
        10 x inception-resnet-B 见(8) k=m=192,l=160,n=1154
        reduction-B             见(9)
        5 x inception-resnet-C  见(10) k=224,l=256,m=2048
        Ave pool
        dropout(0.2)
        FC
        softmax

- Inception V4　网络结构

        stem              见(1)
        4 x Inception-A   见(2)
        Reduction-A       见(5) k=192,l=224,m=256,n=384
        7 x Inception-B   见(3)
        Reduction-B       见(6)
        3 x Inception-C   见(4)
        Ave pool
        dropout(0.2)
        FC
        softmax

- Inception-ResNet V2　网络结构

        stem                   见(11)
        5 x inception-resnet-A 见(7) k=32,l=32,m=256
        Redution-A             见(5) k=256,l=256,m=384,n=384
        10 x inception-resnet-B见(8) k=l=m=128,n=896
        reduction-B            见(9)k=l=m=256
        5 x inception-resnet-C 见(10)k=l=192,m=1792
        Ave pool                       
        dropout(0.2)  
        FC
        softmax

- 模块单元结构 *notes: V represents  valid padding*
   
    - (1)
                    
            Conv 3x3 /2 32 V
            Conv 3x3 32 V
            Conv 3x3 64
            Max pool 3x3 /2   Conv 3x3 /2 96
            Concat
            Conv 1x1 64       Conv 1x1 64
            Conv 3x3 96 V     Conv 7x1 64
                              Conv 1x7 64
                              Conv 3x3 96 V
            Concat
            Conv 3x3 192 V  Max pool 3x3 /2 V
            Concat
            
    - (2)
             
            Ave pool    Conv 1x1 96  Conv 1x1 64 Conv 1x1 64
            Conv 1x1 96              Conv 3x3 96 Conv 3x3 96
                                                 Conv 3x3 96
            Concat
    
    - (3)
    
            Ave pool     Conv 1x1 384  Conv 1x1 192   Conv 1x1 192
            Conv 1x1 128               Conv 1x7 224   Conv 1x7 192
                                       Conv 7x1 256   Conv 7x1 192
                                                      Conv 1x7 224
                                                      Conv 7x1 256
            Concat

    - (4)
    
            Ave pool     Conv 1x1 384     Conv 1x1 384        Conv 1x1 384
            Conv 1x1 128            Conv 1x3 256 Conv 3x1 256 Conv 1x3 448
                                                              Conv 3x1 512
                                                       Conv 3x1 256 Conv 3x1 256
            Concat

    - (5)
    
            Maxpool 3x3/2V Conv 3x3 <n>/2 Conv 1x1 <k>
                                        Conv 3x3 <l>
                                        Conv 3x3 <m>/2 V

    - (6)
    
            Max pool  Conv 1x1 192      Conv 1x1 256
                      Conv 3x3 192/2V   Conv 1x7 256
                                        Conv 7x1 320
                                        Conv 3x3 /2 320 V
                                        
    - (7) residual
    
            Conv 1x1 32  Conv 1x1 32 Conv 1x1 32
            　　　　　　　　Conv 3x3 32 Conv 3x3 <k>
            　　　　　　　　　　　　　　　Conv 3x3 <l>
            Concat
            Conv 1x1 <m>
    
    - (8)residual
   
            Conv 1x1 <k>  Conv 1x1 128
                          Conv 1x7 <l>
                          Conv 7x1 <m>
            Concat
            Conv 1x1 <n>
                
     - (9)
     
            Maxpool 3x3/2V  Conv 1x1 256    Conv 1x1 256 Conv 1x1 256
                            Conv 3x3 384/2V Conv 3x3/2 <k>V Conv 1x1 <m>
                                                         Conv 3x3/2 <l>V
            Concat
              
     - (10)residual
     
            Conv 1x1 192 Conv 1x1 192
                         Conv 1x3 <k>
                         Conv 3x1 <l>
             Concat
             Conv 1x1 <m>
                                        
     - (11)
       
             input
             Conv 3x3 32/2V
             Conv 3x3 32 V
             Conv 3x3 64
             Max pool /2 V
             Conv 1x1 80
             Conv 3x3 192 V
             Conv 3x3 256/2V
     
#### Xception
    François Chollet. Xception: Deep Learning with Depthwise Separable Convolutions.  arXiv:1610.02357v3
    
 - intuition :　在Inception　module中，输入分支路经1x1卷积后再经过其他卷积处理.　这可以等价与一种操作:输入统一进行1x1卷积，之后卷积选择固定一些通道上进行(若后续都只接相同维数的3x3则就是一种group　convolution).极端情况下,Inception　module分支与输入通道一样多,每支都进行各自的卷积.那么此时Inception module退化成一种depthwise separatble convolution.
 - 按照表示空间的角度来理解,将一个2d卷积核分解成先depthwise在pointwise就是在空间和通道的解耦,而之前Inception　V2中将nxn卷积分解成nx1,1xn卷积则是将空间中宽与高解耦
 - 因为堆叠的模式,因此先ponitwise,depthwise　convolution顺序就不重要; 但实验表明在两者之间插入ReLU，会使得性能变差.
 - 网络结构
 
        input 299x299
        Conv 3x3 /2 32
        Conv 3x3 64
        sep Conv 3x3 128        Conv 1x1 /2  128
        sep  Conv 3x3 128
        Max pool 3x3 /2
        Eltwise add
        sep Conv 3x3 256      Conv 1x1/2 256
        sep Conv 3x3 256
        Max pool 3x3 /2
        Eltwise add
        sep  Conv 3x3 728        Conv 1x1 /2 728                
        sep Conv 3x3 728
        Max pool 3x3 /2
        Eltwise add
        -------------------------------------------x 8
        sep Conv 3x3 728        Indentity
        sep Conv 3x3 728
        sep Conv 3x3 728
        Eltwise add
        --------------------------------------------
        sep Conv 3x3 728        Conv 1x1 /2 1024
        sep Conv 3x3 1024
        Max pool 3x3/2
        Eltwise add
        sep Conv 3x3 1536
        sep Conv 3x3 2048
        Ave pool
        dropout(0.5)
        FC
        softmax
    
#### NasNet

    Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le. Learning Transferable Architectures for Scalable Image Recognition.  arXiv:1707.07012v4
    
 - 这是运用autoML构建的网络结构，但因其搜索空间和架构的特点，我将其归入Inception　family.
 - intuition:根据是否降采样,将组成网络的单元分成了两种类型的cell(Reduction与Normal).每次降采样,通道数就增倍以保持计算量不变. 基本结构如下: (初始卷积核个数和N将作为控制网络计算量的超参数)
 
         input 299x299 (311*311)
         Conv 3x3 /2
         Reduction x2
         Nomral xN
         Reduction
         Normal xN
         Reduction
         Normal xN
         Ave pool
         dropout(0.5)
         FC
         softmax
         
 - autoML 搜索空间: 在先前两层cell输出与之前的融合结果之间选择两个隐层,分别进行某种操作，然后将各自结果融合. 操作空间为(1)identity;(2)Conv1x7 Conv 7x1;(3)Conv 1x3 Conv 3x1;(4)Ave pool 3x3;(5)3x3 dilated Conv;(6)Max pool 5x5;(7)Max pool 3x3;(8)Conv 1x1;(9)Max pool 7x7;(10)depthwise separable Conv 3x3;(11)Conv 3x3;(12)depthwise separable Conv 5x5;(13)depthwise separable Conv 7x7; 融合操作为eltwise add与concat.  按照上述操作B(取5)次后，将所有未使用的隐层concat作为该cell的输出.
 - 文中给出了三种比较好的结果，这里只以A为例。
    -  Normal cell
              
       ![](/home/lsq/caffe_notes/Screenshot from 2018-10-31 22-44-08.png) 
              
    - Reduction cell
    
       ![](/home/lsq/caffe_notes/Screenshot from 2018-10-31 22-43-13.png) 
    
 
 - 网络结构
    - N=6,初始卷积核252个.
    - 有depthwise separable convolution时都重复两遍
    - depthwise 和 pointwise中间不用BN,ReLU
    - 使用了lable smooth(0.1) 和辅助classifier
    - 使用scheduled droppath训练.直接elementwise使用dropout效果变差,使用固定概率的droppath没什么效果,需要使用一个线性增加的概率整条支路都drop掉
 
### ResNet and its Variants

intuition:　既然前期研究结果显示神经网络深度很重要，那么不断增加深度会怎么样？实验结果表明不断增加深度并不会使结果单调地变好，当网络变得非常深时,效果甚至很差。将深层网络看成浅层网络的堆叠，即深度增量部分只要是恒等映射，效果就不应该更差，此时作为深度增量的卷积层拟合残差０，再配合skip connection　Identity就可使更深的网络等价于对应浅的网络。深度可以极大拓展网络表示性能,wide resnet则说明网络宽度也可以起到类似的作用.
另外一种理解residual learning的角度是identity mapping提供了梯度反向传播的路径，使深层权值能更有效地优化。 因此若将网络每层都稠密连接，将引出DenseNet, 结合residual path 和inception 多分支结构将引出ResNeXt,结合residual path和Dense connection便引出DualPathNet.

#### ResNet

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, et al. "Deep Residual Learning for Image Recognition."  arXiv:1512.03385.
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, et al. "Identity Mappings in Deep Residual Networks". arXiv:1603.05027
   
- 采用Residual block堆叠而成,每个Residual block 由两个3x3卷积组成.当网络较深时为控制计算量,用1x1卷积先降低特征通道数,再经3x3卷积后用1x1卷积恢复通道数,称为 bottleneck.
- 在降采样时(该阶段的第一个卷积用strided　convolution实现),通道数相应变厚，使每层计算量保持相当.此时的Identity可以直接用0补齐深度，或者用1x1/ 2卷积扩张深度.　Identity支路需要保持干净，不要引入别的阻碍梯度传递的环节.
- 原始residual支路为Conv-BN-ReLU-Conv-BN, Eltwise add再ReLU,称为post-activation(进入下面residual的是activated feature),这种结构是后面variants经常采用的.　为了更有利于梯度传递,在Eltwise add后都不加非线性环节.非对称地将激励加入到Residual支路中,将引出一种BN-ReLU-Conv-BN-ReLU-Conv的Residual支路结构,称为pre-activation(进入下面residual需要先activate),将进一步提升post-activation结构的性能.
- 遵循以上三点可以设计一系列ResNet, 下面以ResNet-152为例(重复卷积层都是Residual支路)

        Conv 7x7 /2 64
        Max pool 3x3 /2
        -------------------- stage II x3
        Conv 1x1 　64
        Conv 3x3  64
        Conv 1x1  256
        -------------------- stage III x8
        Conv 1x1 　128
        Conv 3x3  128
        Conv 1x1  512
        -------------------- stage IV x36
        Conv 1x1 　256
        Conv 3x3  256
        Conv 1x1  1024
        -------------------- stage V x3
        Conv 1x1 　512
        Conv 3x3  512
        Conv 1x1  2048
        --------------------
        Ave pool
        FC
        softmax
        
    
#### Wide ResNet
   
    Zagoruyko, Sergey, Komodakis, Nikos. "Wide Residual Networks". arXiv:1605.07146
    
- ResNet将网络设计的很深，甚至将Residual支路设计成bottleneck结构,更显网络细长.而在如此深的网络中,信息通过skip conection流动,很多卷积层贡献很小甚至是多余的.
- 为了更大化利用Residual支路,让它变得更宽.为防止过拟合,在堆叠的两个3x3卷积之间加入dropout . *notes*既然要宽,就不考虑bottleneck那种1x1,3x3,1x1结构,且使用的pre-activation结构.
- 在训练中更宽的网络可以更有效地利用GPU并行计算能力,使得计算更快(8倍).
- 网络只需将对应的Residual支路加宽k倍即可. 16层WRN,k=2时[取前面ResNet结构为基准k=1,共三个block,每个block由两个3X3卷积堆叠,每个stage重复5次],在CIFAR-10/100上就超过了ResNet-1001(参数量相当).
        
#### ResNeXt
    
    Xie S, Girshick R, Dollar P, et al. Aggregated Residual Transformations for Deep Neural Networks. arXiv:1611.05431.
     
- intuition:像Inception那样先拆分,再变换,最后融合方式高效且有效.但结构复杂,难以迁移去指导别的网络设计. 而像VGG,ResNet模块化的设计，结构均匀.
- 将ResNet中bottleneck改造成多分支结构,分支个数称为cardinality.　
- 将1x1卷积全部concat起来,3x3卷积各自涉及到对应的通道计算后再经1x1卷积全部相加,实际上等价于将3x3卷积改成了group convolution, cardinality就是组数.设通道数d,组数c.　常规卷积计算量正比于d^2^ ,改造后为d^2^/c. cadinality越大,计算量越小.因此在计算量一定的限定下, 3x3卷积可以比原先更厚一些(d×c更大).
- ResNext-101(64x4d):将ResNet-101略微修改(上图各stage重复次数分别为3,4,23,3),第一阶段组数即cardinality为64,每个Conv 3x3 分支4channel,即共256通道,此后随着下采样,通道数逐渐翻倍变厚.
    
####DenseNet

    Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger."Densely Connected Convolutional Networks".  arXiv:1608.06993v5
    
- DenseNet由Dense Block和Transition Block组成. DenseBlock中，每层输入由前面所有层输出concat而成,每层输出新的k层feature. Transition Block由Conv 1x1和Ave pool　2x2/2组成.
- 在Dense block中基本单位是Conv 1x1　4k; Conv 3x3 k 组成bottleneck(此称为DenseNet-B),在Transition block中Conv 1x1 压缩通道(压缩比取0.5)(此成为DenseNet-C),两者兼有的称为DenseNet-BC (k可取32,48等)
- Dense connection可以理解为每一层离最终的loss function 都很近，相当于共用辅助分类器.　这里强调了特征的显式重用,而不像ResNet那像特征相当于一个状态而隐式重用(从这个角度类似于展开的RNN).
 
    
####DualPathNet

    Yunpeng Chen, Jianan Li, Huaxin Xiao, Xiaojie Jin, Shuicheng Yan, Jiashi Feng. "Dual Path Networks". arXiv:1707.01629v2
    
- intuition:　结合了ResNet特征重用和DenseNet发现新特征的能力,沿用ResNet的bottlenet时,最后Conv 1x1　输出split成两部分,一部分Eltwise　add用作residual　path,另一部分concat到前面本底特征作为Dense connection path.　再结合ResNeXt中cardinality效果(Conv 3x3 用group convolution表示).
   
## efficient models

在算力有限的硬件平台难以支撑庞大的模型，因此需要用更小的模型来达到胜任的效果.间接方法是将大模型剪裁,压缩,量化达到减少模型大小和加速推断的效果,直接方法是直接设计高效模型.这一节仅聚焦于高效模型的设计出发点和结构.
###SqueenzeNet

    Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer. "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size". arXiv:1602.07360v4

- 是第一个慎重考虑在移动端部署并优化的模型，压缩模型且保持性能所考虑要点:
    - 将3x3卷积用1x1卷积替代,减少运算量和参数
    - 减少3x3卷积输入的通道数,减少运算量和参数
    - 降采样阶段尽量后延,希望在大尺度上得到更好的特征
- firemodule
基于以上出发点，设计的block称为firemodule.这个block本质上是一种mini型的inception，结构为:
        
        Conv 1x1                # squeeze layer
        Conv 1x1  Conv 3x3      # expand layer 
        Concat
        
 - fire module 超参数
 在squeeze layer中用1x1卷积降低通道数,在expand layer中将部分3x3卷积用1x1卷积代替.几个超参数: 3x3卷积占expand　layer的比例,squeeze layer 占expand layer 通道的压缩比,  expand layer 的通道每隔若干module增加若干通道.
 - SqueezeNet 
 选择第一个fire module expand layer　128 通道,每隔2个module增加128个,　3x3占比0.5, 压缩比0.125.实验表明这两个系数越大性能越好，但是会饱和,最后选在了膝点上.进一步引入skip connection,　即在相同通道数的支路引入residual　path会进一步提高网络性能.　值得指出的是,因为每两个module要增加通道数不能直接加入identity mapping作为skpi connection，若用1x1变换通道数时,还不如不增加这些1x1的skip coonnection. (当然,性能还是比移除全部skip connection要好)
 
        Conv 7x7/2 96
        max pool 3x3 /2
        fire module
        residual fire module  
        fire module
        max pool 3x3 /2
        residual fire module  
        fire module
        residual fire module 
        fire module
        max pool 3x3 /2
        residual fire module 
        Ave pool
        FC
                
### MobileNet family

#### MobileNet V1

    Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam. "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications". arXiv:1704.04861v1 
    
- 采用seperable　convolution带来的效益
假设输入blob　为NxC_{i}xHxW,输出为NxC_{o}xHxW,卷积大小为KxK,那么卷积核参数量便为C_{o}xC_{i}xKxK,计算量(FLOPs)为HxWxC_{o}xC_{i}xKxK. 若将卷积变成group convolution，即输入blob在channel维上分成g组,每个卷积核在组组内做卷积后把所有结构在channel维上concat得到最终结果.那么卷积核参数量就变成C_{o}xC_{i}/gxKxK,计算量为HxWxC_{o}xC_{i}/gxKxK. 当分组数与channel数相同时,便蜕化成depthwise　convolution,每个输入channel一个卷积核得到一个特征,再经1x1卷积(pointwise convolution)便成为seperable convultion，是把常规卷积在通道数和空间维度解耦的轻量级实现.　seperable　coinvolution的参数量为C_{i}xKxK+C_{o}xC_{i}　,计算量为HxWxC_{i}xKxK+HxWxC_{i}xC_{o}.
- 用两个参数控制整个网络的大小: 输入分辨率(空间维度的缩放系数)和网络宽度因子(通道维度的缩放系数).计算量都大约他们平方成正比.
- 小模型训练时weight decay比较小,也不需要辅助分类器和label smoothing.
- 网络结构　基本上就是用seperable convolution实现的VGG,max pooling 均由该阶段第一个卷积stride为2表示.

        Conv 3x3 /2 32
        sep Conv 3x3 64
        ----------------------- stage II x2
        sep Conv 3x3 128
        ----------------------- stage III x2
        sep Conv 3x3 256
        ----------------------- stage IV x6
        sep Conv 3x3 512
        ----------------------- stage V x2
        sep Conv 3x3 1024
        Ave pool
        FC
　　　　　　　 
#### MobileNet V2

        Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. "MobileNetV2: Inverted Residuals and Linear Bottlenecks". arXiv:1801.04381v3
    
- intuition:在feature维度较少时,使用ReLU会破坏信息,且难以恢复.而当先升维经过ReLU再降维,信息将大概率保留. 因此本文提出了一种上边厚,下边薄的inverted　bottleneck. 本来botttleneck设计初衷就是为了降低3x3卷积的计算量,但是现在将其改造成seperable convolution后计算量减少很多,不再需要bottleneck减少输入通道数.
- 基本组成单元: inverted bottleneck. 就是一个上边厚，下面小的bottleneck(post activation)，其中3x3卷积都是separable convolution. 为了降低信息损失，在通道数较少的eltwise　add之后移除了ReLU.
- 网络结构　虽然其比V1层数多,基本组成单位还多了1x1卷积,但每个block的输入输出通道数很少(极大地减少了conv 1x1计算量),总体上比V1计算量更少. 定义扩张比t为inverter bottleneck中间1x1输出通道比输出通道数,降采样都是通过每个阶段第一个block中separable convolution 3x3 stride=2实现. *notes*　该网络的宽度系数小于1时将不应用于最后一层Conv 1x1以保持性能

        Conv 3x3/2 32
        inverted bottleneck 16 t=1
        --------------------------    stage　II  x 2
        inverted bottleneck 24 t=6
        --------------------------    stage　III x 3 
        inverted bottleneck 32 t=6
        --------------------------    stage IV x 4 
        inverted bottleneck 64 t=6
        ---------------------------     x3 
        inverted bottleneck 96 t=6
        --------------------------    stage V x 3 
        inverted bottleneck 160 t=6
        -------------------------- 
        inverted bottleneck 320　t=6
        Conv 1x1　1280
        Ave pool
        FC
        
#### Mnasnet

    Mingxing Tan1, Bo Chen2, Ruoming Pang1, Vijay Vasudevan1, Quoc V. Le1. "MnasNet: Platform-Aware Neural Architecture Search for Mobile". arXiv.1807.11626v1
    
   作为autoML的结果,其搜索的结果跟mobileNet V2非常相似,因此我将其归入mobileNet family 
- 不管是计算量还是数据访问量都是理论上所需时间,这里直接用硬件测模型前向的耗时,结合准确率，作为强化学习的reward。多目标优化的cost function是一个weighted product
- 因为数据集之间有差别,Mnasnet直接在ImageNet上训练.但训练集仅在一个子集(proxy)上训练5轮，在验证集上测准确率
- 以往的自动搜索都是搜索一个单元cell后堆叠起来，这里将整个网络划分成7个组(block),每个阶段搜索一种结构,每个阶段第一个负责降采样.4/5，6/7都在相同阶段,即5,7没有降采样
- 搜索空间:
    - 卷积操作：常规卷积及其核大小,seperable卷积其核大小,inverted bottleneck及其扩张比
    - skipconnection操作: max 或 ave pooling, identity mapping，None
    - 输出通道数
    - 组内该结构重复次数
- 网络结构 结构中不同的是找到了一些5x5的卷积核，虽然两层3x3卷积和5x5感受也相同,但在seperable convolution中 2*(9+N)>1*(25+N) -> N>7 使用的多了,5x5计算量也是经济的。相同准确率下MnasNet比MobileNet V2 top-1高2%

        Conv 3x3/2  32
        SepConv 3x3 16
        ---------------------------------stage II x3
        inverted bottleneck 3x3 24  t = 3
        ---------------------------------stage III x3
        inverted bottleneck 5x5 40  t = 3
        ---------------------------------stage IV x3
        inverted bottleneck 5x5 80  t = 6
        --------------------------------  x2
        inverted bottleneck 3x3 96  t = 6
        ----------------------------------stage V x4
        inverted bottleneck 5x5 192 t = 6
        --------------------------------  x1
        inverted bottleneck 3x3 320 t = 6
        Ave pool
        FC 
　　　　
### ShuffleNet family

#### shuffleNet V1

    Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun. "ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices". arXiv:1707.01083v2

-  将seperable convolution中1x1　卷积改造成group convolution将进一步降低运算量以便加宽网络
- 将1x1conv也改造成group后,各个group之间就没有信息交换了(一开始就将各个通道分组，别对做depthwise 3x3 ,1x1卷积后concat),因此这些操作完后需要增加一个channel shuffle操作,即每个group中第i个channel　拼接成group　i.即　blob.reshape(num,group,channel//group,height,width).transpose(0,2,1,3,4).reshape(num,channel,height,width)
- 在post-activation bottleneck中,有两个1x1 conv，但只在第一个后面添加channel shuffle(在第2个1x1 conv 后面再加没有什么提升，这里仅起到恢复通道数作用),且与Xception相同,在depthwise 3x3 conv 后不加ReLU.

        shuffle unit:
        1x1 group convolution 
        BN,ReLU
        channel shuffle
        3x3 depthwise convolution
        BN
        1x1 group convolution 
        BN
        Eltwise add
        ReLU
        
- 降采样模块借鉴了inception,将3x3/2　Ave pool 和residual path(3x3/2)concat,达到降采样和升维的目的.

        shuffle unit donwsampling:
        1x1 group convolution         Ave pool 3x3/2
        BN,ReLU
        channel shuffle
        3x3 depthwise convolution
        BN
        1x1 group convolution
        BN 
        Eltwise add
        ReLU
        
- shuffleNet结构:与惯例相同，降采样发生在每个阶段的第一个模块,每个bottleneck中间层的压缩比为4,.下面以组数为3为例:

        Conv 3x3/2 24
        Max pool 3x3/2
        ------------------ stage II x 4
        shuffle unit 240
        ------------------stage III x 8
        shuffle unit 480
        ------------------stage IV x 4
        shuffle unit  960
        Ave pool
        FC
      
#### shuffleNet V2
    
    Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun. "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design". arXiv:1807.11164v1
    
- 在以往高效模型设计时,仅考虑了计算量,但计算量仅是计算耗时的近似估计,忽略了别的因素:
    - 数据的获取　
    - Element wise操作
    - 并行化程度
    
    因此需要在计算量不变的情况下,从以下几个方面优化网络:
    
    - 输入输出通道尽量均衡: C_{1}+C_{2} >= 2 \sqrt{C_{1}C_{2}} \alpha \sqrt{FLOPs}
    - group conv时，group不能取太大,取较大时,通道数必然很多
    - Elementwise 操作尽量减少,例如Eltwise add等
    - 减少复杂分支结构,一个block尽量简洁,方便并行计算.
    
- V2对V1的改进 
    - 减少Elementwise操作,将Residual 中的Eltwise add改成了concat.输入在channel上split成两半,一半做identity mapping,另一半卷积后cancat.
    - 减少分组卷积组数. conv 1x1 不再是分组卷积,(split成半后实际上厚度已经减少了，相当于分了两组).为保持通道数均衡,bottleneck结构改成中间层数与输入输出相同厚度.
    - channel shuffle 移到concat之后,促使两支进行信息交换融合
    - 在降采样模块中，分别经改进bottleneck(3x3/2)和seperable conv 3x3/2后concat (相当于将基本模块中split去掉,identity支路变成seperable convolution)
    - Ave pool前加了一层Conv 1x1 1024. 每个阶段输出通道数减少为到116,232,464.
    
### boost block in CNN: attention

作为CNN中作为轻量级组件，self-attention机制的引入将进一步提升CNN的表达能力,且可以嵌入到之前所述任何一种CNN中.根据空间和通道两个维度,本文总结了四种attention机制，分别为spatial transformer, squeenze and excitation, residual attention和 convolution block attention module. 

#### spatial transformer

    Max Jaderberg, Karen Simonyan, Andrew Zisserman,Koray Kavukcuoglu. "Spatial Transformer Networks". arXiv:1506.02025v3.
    
- 在FasterRCNN中proposal可以看成一种spatial attention,但它需要根据额外的真值框标注训练RPN,且在训练时因难以对坐标求导而只能近似或者采用多阶段训练. 采用spatial transformer后，就可以仅利用图片类别标签自主学习仿射变换参数和实现对坐标的求导.（建立了输出坐标和输入坐标之间的关系,因此可求得输出像素对输入坐标的梯度,可以反向传播坐标预测对RPN的梯度,但在后期改为RoiAlign后,也不再需要这样)
- 特征图上做仿射变换后得到输出,由三部分组成
    - localization 用一般的神经网络参数化坐标变换矩阵,如得到放射变换矩阵的参数，初始化时使得变换矩阵为identity
    - grid generator 建立输出与输入特征图的坐标映射 torch.functional.affine_grid(theta, size)#size为输出特征图尺寸,theta为变换矩阵,返回对应于输出特征图各个像素坐标的输入图像坐标
    - sampler 根据映射与变换规则(双线性变换或者最近邻等)得到输出特征图
    torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros') #根据输入特征图和上步得到的坐标计算输出特征图像素值.
- 可以解决输入图像扭曲变形问题,还可以自动发现需要关注的区域,正因为如此,缺点便是仅能处理实例较少的图片.

#### squeenze and excitation

    Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu. "Squeeze-and-Excitation Networks". arXiv:1709.01507v3
    
- 这是在通道维度做attention.考虑到卷积局限在自身感受野内,难以感知整张图,因此用一个global pool来对各个channel做权值调整.考虑到调整需要含非线性,各个通道之间不是排它的,因此选用含有隐层的全连接网络,用sigmoid做gating.为减少开支,将隐层维数压缩r倍.
- 基本结构 以一个block为单位,例如inception或者residual path,将SE单元接在最后调整各个channel，然后再输出.
        
        ave pooling        # ->Cx1x1
        FC                 # ->C/rx1x1
        ReLU            
        FC                 # ->Cx1x1
        sigmoid               
        product  # 各个channel乘对应权重 
        
- 设计出发为克服感受野局限性,但参数和计算量主要集中在网络后期channel多的阶段,在网络后期移除SE模块可以以较少损失换取较多的计算量节省. 调整r逐渐变小计算量的增大并不是单调的使性能变好,实验中r=16为佳.
- 在主流网络中加上SE,无论原来深度如何,都可以获得近1%的分类准确率和AP的提升.

#### residual attention

    Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Cheng Li, Honggang Zhang, Xiaogang Wang, Xiaoou Tang. "Residual Attention Network for Image Classification". arXiv:1704.06904v1

- 是一种spatial-channel mixed attention,即将blob所有元素统一考虑作为gating.结构为encoder-decoder结构(前期encoder为快速捕捉全局信息),并且结合乘性的residual learning,即feature为F,attention为M时,refined后输出(1+M)*F.
- 实验表明attention采用普通结构,乘法中不加identity效果都不佳.
- residual attention结构

        max pooling 
        residual unit -->
        max pooling     |
        residual unit   |
        residual unit   |
        upsample        |
        residual unit   |
        upsample        |
        eltwise add  <--
        Conv 1x1
        Conv 1x1
        sigmoid
        
#### convolution block attention module(CBAM)

    Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon. "CBAM: Convolutional Block Attention Module". arXiv:1807.06521v2
    
- 同时考虑channel和spatial，但将两者解耦,减少计算量
- channel attention与SE相同,但同时考虑了global max pooling和global Ave pooling,它们共享一个含一层隐层的FC后相加作sigmoid
- spatial attention:沿channel维度分别作Ave pooling和max pooling 后concat,再接convolution(感受野大一些为佳,采用7X7),最后sigmoid
- 结合:两个attention module串联,且先channel后spatial效果最好