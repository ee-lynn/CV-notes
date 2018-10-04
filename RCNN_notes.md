# R-CNN(region based CNN)的前世今生
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　sqlu@zju.edu.cn
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
    - 作为待回归的系数，定义为
	
	$$t_x = (G_x-P_x)/P_w$$  
	$$t_y = (G_y-P_y)/P_h$$
	$$t_w = log(G_w/P_w)$$
	$$t_h = log(G_h/P_h)$$ 
	
    - 训练时的真值框和候选框的匹配: 与候选框最大IOU，且大于0.6的真值框预支匹配.不符合条件的候选框弃用.
	
## Fast R-CNN
    arXiv:1504.08083v2
    sppNet: arXiv:1406.4729v4
- 前奏sppNet　(相关部分)
    - 提出ROI Max　pooling, 解决了CNN需要制定输入图像的问题: 无论输入多少图片，在POI Max pooling层都输出相同大小的feature map(NxM 需要得到 axa的特征图, 取 ceil(N/a),ceil(M/a)大小的窗,stride为floor(N/a),floor(M/a)做传统的Max pooling即可)
    - 不需要在图像上提出候选框后都过一遍CNN, 而是整张图片过CNN后在feature map 上ROI Max pooling
- CNN改动
    - 将ImageNet上pretrain的模型中最后一个pooling 层改成ROI pooling,将输出改成两个分支: 分类和回归
    - 为防止一个finetune 一个proposal需要计算整张图, 在组成minibatch中采用N张图片中随机选取R/N个proposal,这样只需要前向N次即可
    - loss
    最后一个FC改成两个FC,分别是分类(K+1类softmax)和坐标回归(每一类[不包括背景]都有４个回归系数[回归系数与R-CNN相同],做smoothL1回归).两个loss权重相同. 实验证明,这里用softmax效果比SVM更好
    - train 
    训练时,与RCNN相同,正负样本比例采样为3:1，IOU>=0.5时定义为正样本.略有不同的时，将IOU在[0.1,0.5)的定义为负样本,可以看成一种ard example mining.　训练时有两种策略:　固定尺寸训练与image pyramids. 实验证明固定尺寸训练的模型就可以有较好的性能: 固定短边s,保持图片宽高比resize同时限制长边小于L(若超过,则用长边确定缩放比例)
    - trick
    FC可在SVD分解下成为两个FC，当忽略其中一些奇异值时,可以加快计算速度,即
	
	$$W_{mn} = U_{m} \Sigma_{mn} V_{n}^T$$
	
	奇异值取前t大个，可以将FC由m输入,n输出分解为m输出,t中间层,n输出.计算量由mn降低到t(m+n)
	
## Faster R-CNN
    arXiv:1506:041497v3 Faster R-CNN:Towards Real_Time Object Detection with Region Proposal Networks
- RPN:在CNN上叠加简单的卷积层推断基于网格化anchor的回归系数和是否有物体的概率,并以此作为proposal,作为Fast R-CNN进一步检测的基础. 在CNN卷积层最后加3x3,ReLU,再叠加两支1x1,分别为每个anchor预测存在目标的概率和四个回归系数.　在推测时,根据概率值先做一次NMS(None maximum supreesion)，降低proposal个数
- anchor:　即先验框,具有不同的大小和宽高比.全卷积网络平移不变性带来的优势是伴随feature map每个元素的anchor按照网格形式覆盖整张图片.
- loss 与Fast R-CNN的loss类似，分为logistic regression+smmothL1 regresion两支.
- train 将两类anchor定义为正样本，并计算回归系数目标值: (1)与某个Ground Truth具有最大IOU;(2)　与ground truth的IOU大于一个阈值(0.7). (1)存在的意义仅在于保证每个ground truth都有对应的anchor. 与所有ground truth 的IOU小于一个阈值(0.3)的定义为负样本. 训练中正负样本采样为1:1，且采样方式沿用Fast R-CNN
- RPN联合Fast R-CNN 训练
    - 交替训练 先训练RPN,然后用它的proposal训练Fast R-CNN，然后将CNN权值移到RPN再finetune3x3,1x1卷积层，再finetune Fast R-CNN的FC,
    - 近似的联合训练 将RPN的loss和Fast R-CNN的loss加起来直接反向传播.其中两者交互部分认为RPN输出proposal位置固定不变，即忽略ground truth的坐标对RPN回归系数经过Fast R-CNN这路的反向传播路径(主要是ROI Max pooling 对坐标不可导)
    - 严格的联合训练 将ROI Max pooling改成ROI warp(即将ROI resize, arxViv 1506.02025v3),对坐标是可导的. 在pytorch中有直接的变换(torch.nn.functional.affine_grid,torch.nn.functional.grid_sample)，可以不关心其中的细节
    - 超越图片边缘的anchor在训练时是忽略的,而在推断时计入其中,检测框的最终形成需要限制框在图像范围内.


    

　　 
　　　 
