# yolo(you look only once)前世今生　 
 &nbsp;　　　　　　　　　　　　sqlu@zju.edu.cn
 
## 基本思想
 - 讲目标检测问题看成回归问题，图像经过CNN后直接得到是什么(类别概率),在哪里(目标框)
 - 根据CNN平移不变性，最终得到feature map为NxN时，就是将原图分割为NxN的网格进行检测，每个点感受野为以该网格中心的区域
 - 检测模型得到MxNxN 的tensor，M包含４个坐标(目标框相对网格左上角的偏移量,目标框的长宽)，1个网格有物体的概率，检测任务中Ｃ类物体的概率
## yolo v1
- backbone 直链，输入448x448

    conv 7x7/2 64 
    maxpooling  2x2/2 
    conv 3x3 192
    maxpooling 2x2/2 
    conv 1x1 128 
    conv 3x3 256
    conv 1x1 256
    conv 3x3 512 
    maxpooling 2x2/2
    conv 1x1 256, conv 3x3  512    ｘ4
    conv 1x1 512 
    conv 3x3 1024
    maxpooling 2x2/2 
    conv 1x1 512,  conv 3x3  1024   ｘ2
    conv  3x3 1024
    conv  3x3/2 1024 
    conv 3x3 1024 
    conv 3x3 1024 
    FC 4096 
    FC 1470
    
- 最后得到的feature map结构说明
每个网格预测B(B=2)个框，最终的feature map 形状为[2x(4+1)+20]x7x7
- train　detals (VOC有20类,coco有80类)
在3x3 1024 前global ave pooling+FC 在ImageNet pretrain,224x224输入: top-5 :88%
加入４层卷积层和２层全连接层,输入扩展成448x448 训练detection.
leaky ReLU(0.1),drop out(0.5) after first FC

- loss function
.$$\lambda_{coord}\sum_{i=0}^{M^2}\sum_{j=0}^B1_{ij}^{obj}[(x_i-\hat{x}_i)^2+(y_i-\hat{y}_i)^2]+\lambda_{coord}\sum_{i=0}^{M^2}\sum_{j=0}^B1_{ij}^{obj}[(\sqrt{w_i}-\sqrt{\hat{w}_i})^2+(\sqrt{h_i}-\sqrt{\hat{h}_i})^2]$$
$$+\sum_{i=0}^{M^2}\sum_{j=0}^B1_{ij}^{obj}(C_i-\hat{C_i})^2+\lambda_{noobj}\sum_{i=0}^{M^2}\sum_{j=0}^B1_{ij}^{noobj}(C_i-\hat{C}_i)^2+\sum_{i=0}^{M^2}1_i^{obj}\sum_{c\in{classes}}(p_i(c)-\hat{p}_i(c))^2$$
每个网格中一个框仅负责一个目标, (with highest IOU with GT, loss中以j表示)
只有当网格中匹配物体时，类别概率会计算损失,根据对应目标对目标框坐标计算损失
为克服大框产生的loss更大，这里以长宽开方作为回归目标,一定程度上加大了小目标位置的错误惩罚
$$\lambda_{coord}=5,\lambda_{noobj}=0.5$$加大了坐标回归惩罚,降低了无目标的概率惩罚(有些类似于focal loss，这里多分类中没有背景这一类)
## yolo v2
- frame adjustment
输入图像改为416x416，使得最后网格数为奇数(13x13),有中心点预测很大的目标
使用anchors，为每个anchors均分配独立的类别概率，anchors采用k-means得到:
根据训练集中目标框的,将其全部移到坐标原点,定义距离为1-IOU(box,centroid),得到k类聚类中心的长宽，可以作为anchor priors
修改目标框坐标预测形式　
$$b_x = \sigma(t_x)+c_x \\ b_y = \sigma(t_y)+c_y \\ b_w = p_we^{t_w} \\b_h =  p_he^{t_h}$$
$$p_w,p_h$$是anchor的长宽,用对数空间更大区分大小目标框惩罚力度,网格的左上角坐标为$$c_x,c_y$$,用sigmoid限制目标框中心在该网格内
加入了shortcut(见backbone)
- train
添加batch norm, 在ImageNet上pretrain时最后10　epoch采用448x448
在训练detection时，随机resize图片尺寸为32倍数
- backbone 删去FC,网络为全卷积网络
    conv 3x3 32
    maxpooling 2x2 /2
    conv 3x3 64
    maxpooling 2x2 /2
    conv 3x3 128
    conv 1x1 64
    conv 3x3 128
    maxpooling 2x2/2
    conv 3x3 256
    conv 1x1 128
    conv 3x3 256
    maxpooling 2x2 /2
    conv 3x3 512
    conv 1x1 256
    conv 3x3 512
    conv 1x1 256
    conv 3x3 512 --->   reshape: 2x2x512 ->1x1x2048 [shortcut]
    maxpooling 2x2 /2
    conv 3x3 1024
    conv 1x1 512
    conv 3x3 1024
    conv 1x1 512
    conv  3x3 1024
    |------------------------|:--------------------------:|
    |classifer                | 　   detection              |
    |conv  1x1 1000  　  |    conv 1x1 1024       |
    |global ave pooling|    conv 3x3 1024       |
    |                              |    concat　　　with [shortcut]|
    |                              |   conv 3x3　 1024       |
    |                              |   conv 1x1 125 (5个anchor,20类) |
##yolo v3
- backbone
    conv 3x3 32
    conv 3x3/2 64
    residual conv 1x1 32,  conv 3x3 63  
    conv 3x3/2 
    residual conv 1x1 64, conv 3x3 128  x2
    conv 3x3/2 256
    residual conv 1x1 128,  conv 3x3 256  x8
    conv 3x3/2 512    [shortcut scale0]
    residual conv 1x1 256, conv 3x3 512  x8
    conv 3x3/2 1024   [shortcut scale1]
    residual conv 1x1 512, conv 3x3 1024  x4
    |----------------------- |: --------------------:|
    |classifier               |      detection       |
    |global ave pooling| conv 1x1 512, 3x3 1024   x3  -> 1x1 75 出大目标框|
    |FC 1000                 | conv1x1 256|
    |                              |upsample x2|
    |                              |concat with [shortcut scale1]|
    |                              |conv 1x1 256 , 3x3 512   x2  ->1x1 256, 3x3 512, 1x1 75 出中目标框|
　　　　|　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|conv 1x1 128|
　　　　|　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|upsample x2|
　　　　|　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|concat with [shortcut scale0]|
　　　　|　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|conv 1x1 128, conv 3x3 256   x2|
　　　　|　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|conv 1x1 75 出小目标框|
    
- 多标签分类，采用sigmoid输出类别概率
- feature pyramid
在三个尺度上各出３个anchor(32x32,16x16,8x8),feature　map不一样大,最后一起做预测时可以以预测向量concat.