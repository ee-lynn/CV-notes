# deep convolutional nerual network architecture
 &nbsp;　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　       by sqlu@zju.edu.cn
## overview
 本文总结2012年以来深度卷积神经网络的结构，分为两条主线:
 
- 性能角度的发展 
- 效率角度的发展

## more powerful!
随着深度学习的主干网络结构的变迁,　作为视觉两大基础任务的图片分类和人目标检测任务，在大型数据集上(ImageNet, coco)的指标被逐渐推高.这里就以一些明星网络结构为例,罗列其结构的要点。
###AlexNet

    Krizhevsky, Alex, I. Sutskever, and G. E. Hinton. "ImageNet classification with deep convolutional neural networks." International Conference on Neural Information Processing Systems Curran Associates Inc. 2012:1097-1105.

AlexNet是2012年ImageNet竞赛冠军获得者Hinton和他的学生Alex Krizhevsky设计的，吹响了CNN大规模进军计算机视觉的号角，其成功的因素可以归结为:

- 激活函数均使用了ReLU(Rectfier Linear Unit)
- data augmentaion: 
horizontal flipping, random crop[256->224], PCA jittering
- GPU training
accelarate trainning spped
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

- ImageNet 结果
**Top-1 36.7% Top-5 15.4%**

### VGG
    Simonyan K, Zisserman A. Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv:1409.1556, 2014.
    
 VGG是2014年ImageNet场景分类第二，其成功的因素归结为
 
 - 3x3的卷积核代替大的卷积核
 事实上，两层3x3卷积跟一层5x5卷积具有等价感受野,但参数更少,且其中可以插入激活函数添加非线性.
 - multi-scale training  resize 到不同大小的图片后再crop
 - 网络结构，以最常使用的VGG-19为例
        
        Conv 3x3 64    ReLU
        Conv 3x3 64    ReLU
        maxpool 2x2  /2
        Conv 3x3 128  ReLU
        Conv 3x3 128  ReLU
        max pool 2x2 /2
        Conv 3x3 256  ReLU
        Conv 3x3 256  ReLU
        Conv 3x3 256  ReLU
        Conv 3x3 256  ReLU
        max pool 2x2 /2
        Conv 3x3 512  ReLU
        Conv 3x3 512  ReLU
        Conv 3x3 512  ReLU 
        Conv 3x3 512  ReLU
        max pool  2x2 /2
        Conv 3x3 512  ReLU    
        Conv 3x3 512  ReLU      
        Conv 3x3 512  ReLU
        Conv 3x3 512  ReLU
        max pool 2x2 /2
        FC  4096 dropout(0.5)
        FC  4096 dropout(0.5)
        FC  
        softmax
        
- ImageNet 结果
**muticrop ensemble Top-1 24.4% Top-5 7.1%**
  
###GoogLeNet and Inception family


自2014年ImageNet 第一名GoogLeNet提出后,又在之后相继提出了Inception V2,V3,V4.他们的共同点都是使用了Inception module. 这种module是一种多路拼接结构。训练时加入了Batch Normalization更加稳定，使之成为训练深度网络的必备元素。将多路拼接结构以可分组卷积(group convolution)的观点重新阐释并推向深度分离卷积(depthwise seperatable convolution)可导出Xception。Inception module也是后续神经网络架构自动搜索技术(NasNet)的雏形结构。

#### GoogLeNet & Inception V1
    Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich. Going Deeper with Convolutions arXiv:1409.4842v1
    Sergey Ioffe, Christian Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift arXiv:1502.03167v3

- intuition: 来自神经科学的hebbian theory: 两个神经元或者神经元系统，如果总是同时兴奋，就会形成一种‘组合’，其中一个神经元的兴奋会促进另一个的兴奋。比如狗看到肉会流口水，反复刺激后，脑中识别肉的神经元会和掌管唾液分泌的神经元会相互促进，“缠绕”在一起，以后再看到肉就会更快流出口水 
- 文中指出该种结构是神经网络结构构建算法的早期结果，算法使用一些已有的模块堆叠组合而成, 此为NasNet出现埋下伏笔.
- 为了减少计算量，Inception module在3x3,5x5前使用１x1卷积来降低输入通道数. max pooling后的1x1卷积目的在不同通道重组, 结构如下(卷积层后自带ReLU)
        
        input
        Conv 1x1 n1 Conv 1x1 n31 Conv 1x1 n51 Max pool 3x3
                    Conv 3x3 n3  Conv 5x5 n5  Conv 1x1 np
        Concat
        
- 为了使梯度更有效地传播，训练时在网络更浅出插入了两个辅助输出分支, 损失分别以0.3的权重辅助传到梯度. 该举措在后续被证明没什么特别大的改善作用.
- 预处理和数据增强.　图片只做了减均值预处理. 随机宽高比在3/4~4/3, crop 8%~100%，随机的resize差值方法,等。测试时实用144 crop: resize短边到256,288,320,352; 左中右/上中下的正方形;四个角和直接resize到224;镜像.所有144个结果上做平均.
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
       
             Ave pool 5x5 /3 valid padding
             Conv 1x1 128
             FC 1024
             dropout(0.7)
             FC 1000 
             softmax
**Top-1 ~29%, Top-5 10.07%**
             
      - Inception
      在ReLU前均插入了BatchNorm层，此后Conv-BatchNorm-ReLU为标配.　
      将Inception module中5x5卷积改成两个3x3卷积, max pool 改成ave pool
      在inception之间的max pool 用inception表示,其中3x3conv stride 为2, pool 仍为max pool.　inception 内部参数具有调整，现为
        
            inception(3a) n1=64 n31=64 n51=64 np=32 n3=64 n5=64
            inception(3b) n1=64 n31=64 n51=64 np=64 n3=96 n5=64
            inception (3c) n1=0 n31=128 n51=64 np=0 n3=160 n5=64 
            inception(4a) n1=224 n31=64 n51=96 np=128 n3=96 n5=96
            inception(4b) n1=192 n31=96 n51=96 np=128 n3=128 n5=96
            inception(4c) n1=160 n31=128 n51=128 np=128 n3=160 n5=128
            inception(4d) n1=96 n31=128 n51=160 np=128 n3=192 n5=160
            inception (4e) n1=0 n31=128 n51=192 np=0 n3=192 n5=192 
            inception(5a) n1=352 n31=192 n51=160 np=128 n3=320 n51=160
            inception(5b) n1=352 n31=192 n51=192 np=128 n3=320 n5=192
**Top-1 25.2%, Top-5 7.8%**
#### Inception V2 & V3
    Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna. Rethinking the Inception Architecture for Computer Vision. arXiv:1512.00567v3
    
- inception(3)把大于3x3的卷积核全部用3x3卷积堆叠实现等效的感受野,减少计算量
       
        input
        Conv 1x1  Conv 1x1  Conv 1x1  Max pool 3x3
                  Conv 3x3  Conv 3x3  Conv 1x1
                            Conv 3x3
        Concat
        
- 在网络的中间阶段将inception(4) nxn卷积分解成1xn,nx1卷积的堆叠,进一步减少计算量(这里n使用了7)

        input
        Conv 1x1 Conv 1x1 Conv 1x1 Max pool 3x3
                 Conv 1x7 Conv 1x7 Conv 1x1
                 Conv 7x1 Conv 7x1
                 Conv 1x7
                 Conv 7x1
        Concat
        
- 用strided convolution+pool的concat代替单纯的pool 或者conv(　feature　map减少时,单独用pool做不到通道数增加,扩大的strided　conv计算量又太大)
- 将Inception(5)改的更宽了，将1x3,3x1堆叠改成并联结构
       
        input
        Conv 1x1         Conv 1x1         Conv 1x1       Max pool 3x3
                      Conv 1x3   Conv 3x1  Conv 3x3      Conv 1x1
                                       Conv 1x3   Conv 3x1
        Concat
        
- 输入分辨率改成299(初始主干部分和inception　stride时候都valid pad,主干部分删去 Conv 1x1,改成两个Conv 3x3 /2, Conv 3x3)
以上结构称为**Inception V2**
**Top-1 23.4%, Top-5 -%**
- 将最开始的7x7卷积也改成3层3x3卷积堆叠(conv 3x3 /2; conv 3x3; conv  3x3 padded),使用RMSProp训练，添加先验为均匀分布的交叉熵损失(权重0.1,作为label　smoothing)　称为**Inception V3**
**Top-1 21.2%, Top-5 5.6%**
#### Inception V4 & Inception-ResNet V1 & V2
    Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi. Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning.  arXiv:1602.07261v2 
    
- 发布了３个模型，分别为Inception V4, Inception-ResNet V1 & V2, 其中Inception-ResNet V1 与Inception V3计算量相当,　nception-ResNet V2 与Inception V4计算量相当
- 结果显示,Residual learning 可以让模型更快收敛,但是最终结果还是由模型容量决定.
- 在Residul 支路加一个0.1的恒定常数scale，可以稳定训练过程
*notes* 文中因内存的限制,Inception-ResNet V1在residual之后elementwise add之后　ReLU之前没有BatchNorm.
- 模块单元结构
(1)

(2)

(3)

(4)

(5)

(6)




- Inception-ResNet V1　网络结构

        stem
        5 x inception-resnet-A
        Redution-A
        10 x inception-resnet-B
        reduction-B
        5 x inception-resnet-C
        Ave pool
        dropout(0.2)
        FC
        softmax
**Top-1 21.3%, Top-5 5.5%**

- Inception V4　网络结构

        stem
        4 x Inception-A
        Reduction-A
        7 x Inception-B
        Reduction-B
        3 x Inception-C 
        Ave pool
        dropout(0.2)
        FC
        softmax
**Top-1 20.0%, Top-5 5.0%**

- Inception-ResNet V2　网络结构

        stem
        5 x inception-resnet-A
        Redution-A
        10 x inception-resnet-B
        reduction-B
        5 x inception-resnet-C
        Ave pool
        dropout(0.2)
        FC
        softmax
**Top-1 19.9%, Top-5 4.9%**

####Xception
    François Chollet. Xception: Deep Learning with Depthwise Separable Convolutions.  arXiv:1610.02357v3
    
#### NasNet


###ResNet and its Variety
2016年K.M. He 自提出residual　learning便一举成名。
它的intuition很自然:　既然前期研究结果显示神经网络深度很重要，那么不断增加深度会怎么样？
实验结果表明不断增加深度并不会使结果单调地变好，当网络变得非常深时,效果甚至很差。将深层网络看成浅层网络的渐进堆叠，即深度增量部分只要是恒等映射，效果就不会更差。此时作为深度增量的卷积层拟合0，再配合skip connection就可使更深的网络等价于对应浅一些的网络。
另外一种理解residual structure 的角度是identity mapping 提供了梯度反向传播的路径，使深层权值能更有效地优化。 因此若将网络做成稠密连接，将导出**DenseNet**,融合ResNet和DenseNet便导出**DualPathNet**

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, et al. Deep Residual Learning for Image Recognition.  arXiv:1512.03385.
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, et al. Identity Mappings in Deep Residual Networks.	arXiv:1603.05027
    Zagoruyko, Sergey, Komodakis, Nikos. Wide Residual Networks.  	arXiv:1605.07146
    Xie S, Girshick R, Dollar P, et al. Aggregated Residual Transformations for Deep Neural Networks. arXiv:1611.05431.
    
    

 
### Attention in CNN
SEnet CMBA??

## more efficient!

###SqueenzeNet
###MobileNet family
###ShuffleNet family
