# yolo(you look only once)前世今生　 
 &nbsp;　　　　　　　　　　　　sqlu@zju.edu.cn
 
## 基本思想
 - 讲目标检测问题看成回归问题，图像经过CNN后直接得到是什么(类别概率),在哪里(目标框)
 - 根据CNN平移不变性，最终得到feature map为NxN时，就是将原图分割为NxN的网格进行检测，每个点感受野为以该网格中心的区域
 - 检测模型得到MxNxN 的tensor，M包含４个坐标(目标框相对网格左上角的偏移量,目标框的长宽)，1个网格有物体的概率，检测任务中Ｃ类物体的概率
## yolo v1
- backbone 直链，输入448x448
<center>conv 7x7/2 64
maxpooling 2x2/2
conv 3x3 192
maxpooling 2x2/2
conv 1x1 128
conv 3x3 256
conv 1x1 256
conv 3x3 512
maxpooling 2x2/2
conv 1x1 256 conv 3x3  512   ｘ4 
conv 1x1 512
conv 3x3 1024
maxpooling 2x2/2
conv 1x1 512  conv 3x3  1024  ｘ2 
conv 3x3 1024
conv 3x3/2 1024
conv 3x3 1024
conv 3x3 1024
FC 4096
FC 1470
</center>
- feature map 说明
每个网格预测B(B=2)个框，最终的feature map 形状为[2x(4+1)+20]x7x7
- train
在3x3 1024 前　global ave pooling+FC ImageNet pretrain,224x224输入: top-5 :88%
加入４层卷积层和２层全连接层,输入扩展成448x448 训练detection.
leaky ReLU(0.1),drop out(0.5) after first FC
- loss function
... to be completed
one bounding box to be responsible for one onject(with highest IOU with GT)
只有当网格中匹配物体时，类别概率会计算损失,根据对应目标对目标框坐标计算损失
## yolo v2
- frame adjustment
输入图像改为416x416，使得最后feature map 为奇数
使用anchors，为每个anchors均分配独立的类别概率，anchors采用k-means得到.　根据训练集中目标框的,将其全部移到坐标原点,定义距离为1-IOU(box,centroid),得到k类聚类中心的长宽，可以作为anchor priors
修改目标框坐标预测形式　
...
加入了shortcut(见backbone)
- train
添加batch norm, 在ImageNet上pretrain时最后10　epoch采用448x448
- backbone
...
##yolo v3
- backbone
...
- 多标签分类，采用sigmoid输出类别概率
- feature pyramid
...
