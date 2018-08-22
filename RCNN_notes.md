# R-CNN(region based CNN)的前世今生
## 基本思想
顾名思义,该算法采用CNN来对图像进行特征提取，作为两阶段法的经典算法,　很直白地将目标检测任务分成在哪里(坐标回归)和是什么(将候选框进一步分类)两个子任务，而其两阶段则是(1)提出候选框; (2)对候选框进一步处理得出目标种类和位置.

## R-CNN
    arXiv:1311.2524v5: Rich feature hierarchies for accuate object detection and semantic segmentation
- 采用候选框算法(selective search)对图像中可能存在目标的区域进行提取作为候选框(proposal) ,一张图像中约提取2000张
- 采用CNN中卷积层对候选框进行特征的提取.
    - 由于CNN最后是FC,　CNN对输入图片有要求,但proposal的形状各种各样. 采取的resize策略为: 首先将proposal向外pad(文中pad 16个像素),然后直接resize到CNN需要的尺寸
- finetune　CNN
    - 首先将CNN在ImageNet上pretrain
    - 将最后FC换成N+1输出的FC后，在检测数据集上finetune. 将IOU大于0.5的proposal作为正样本,其余的作为背景类别进行分类.为保证正负样本的均衡，minibatch组成正负样本在采样时比例保持为3:1
    - 采用这个正负样本的定义一方面在IOU为0.5~1这一段作为数据增强来用，另一方面也是实验做出来的结果。
- 检测器　linear SVM
     - 共有N个one vs. rest linear SVM
     - 正样本取为真值框,负样本为只要与该类真值框IOU小于0.3极为负样本
     - 采用额外的分类器的主要原因是实验中softmax效果比SVM更差
- 回归期
    - 将全连接层前的feature map作为特征,做一个Ridge回归. 且每一类都会做一次
    - 作为待回归的系数，定义为 $$t_x = (G_x-P_x)/P_w$$ $$t_y = (G_y-P_y)/P_h$$ $$t_w = log(G_w/P_w)$$  $$t_h = log(G_h/P_h)$$ 
    - 训练时的真值框和候选框的匹配: 与候选框最大IOU，且大于0.6的真值框预支匹配.不符合条件的候选框弃用.
## Fast R-CNN
    arXiv:1504.08083v2
    sppNet: arXiv:1406.4729v4
- 前奏sppNet　(相关部分)
    - 提出ROI Max　pooling, 解决了CNN需要制定输入图像的问题: 无论输入多少图片，在POI Max pooling层都输出相同大小的feature map(NxM 需要得到 axa的特征图, 取 ceil(N/a) ceil(M/a)大小的窗,stride为floor(N/a),floor(M/a)做传统的Max pooling即可)
    - 不需要在图像上提出候选框后都过一遍CNN, 而是整张图片过CNN后在feature map 上ROI Max pooling
- CNN改动
　　　 - 将ImageNet上pretrain的模型中最后一个pooling 层改成ROI pooling,将输出改成两个分支: 分类和回归
　　　 - 为防止一个finetune 一个proposal需要计算整张图, 在组成minibatch中采用N张图片中随机选取R/N个proposal,这样只需要前向N次即可
　　　 - loss
　　　 
