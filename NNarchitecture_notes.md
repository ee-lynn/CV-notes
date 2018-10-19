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
- 网络结构

        type          size    outnum     padding  stride   group
        Conv         11x11     96          5        4       -
        ReLU
        Conv          5x5     256          2        1       2  
        ReLU
        Max pool      3x3      -           -        2       -
        Conv          3x3     384          1        1       2
        ReLU
        Max pool      5x5      -           -        2      
        Conv          3x3 　　 384          1        1       2
        ReLU
        Conv          3x3   　 256         1         1       2
        ReLU
        Max pool      3x3      -           -         2       
        FC                    4096                     
        ReLU
        dropout(0.5)
        FC                    4096
        ReLU
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
 - 网络结构，以最常使用的VGG-16为例
        
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
**Top-1 24.4% Top-5 7.1%**
  
###GoogLeNet and Inception family


自2014年ImageNet 第一名GoogLeNet提出后,又在之后相继提出了Inception V2,V3,V4.他们的共同点都是使用了Inception module. 这种module是一种多路拼接结构。训练时加入了Batch Normalization更加稳定，使之成为训练深度网络的必备元素。Inception module也是后续神经网络架构自动搜索技术的雏形结构。





###ResNet and its Variety
自2016年K.M. He 提出residual　learning便一举成名。
它的intuition其实很自然:　既然前期研究结果显示神经网络深度很重要，那么不断增加深度会怎么样？
实验结果表明不断增加深度并不会使结果单调地变好，当网络变得非常深时,效果甚至很差。将深层网络看成浅层网络的渐进堆叠，即深度增量部分只要是恒等映射，效果就不会更差。此时作为深度增量的卷积层拟合0，再配合skip connection就可使更深的网络等价于对应浅一些的网络。
另外一种理解residual structure 的角度是identity mapping 提供了梯度反向传播的路径，使深层权值能更有效地优化。 

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, et al. Deep Residual Learning for Image Recognition.  arXiv:1512.03385.
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, et al. Identity Mappings in Deep Residual Networks.	arXiv:1603.05027
    Zagoruyko, Sergey, Komodakis, Nikos. Wide Residual Networks.  	arXiv:1605.07146
    Xie S, Girshick R, Dollar P, et al. Aggregated Residual Transformations for Deep Neural Networks. arXiv:1611.05431.
    


###DenseNet
### Xception
### SENet
###DualPathNet
###NasNet
## more efficient!

###SqueenzeNet
###MobileNet family
###ShuffleNet family
