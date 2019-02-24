# 图像处理
　&nbsp;　　　　  　　　　　　　 sqlu@zju.edu.cn

## 引言
本笔记记录了关于图像的基本概念,方法和实现它们的技术手段:opencv和PIL的使用.
 本文分成以下几个部分:
- 颜色空间
- 图像变换与操作
- 图像压缩与输入输出
- 背景提取
- 图像特征点和描述
- 直方图
- 光流
- 相机模型与立体视觉
## 颜色空间
彩色模型是坐标系统和子空间的说明,位于系统中的每种颜色都是由单个点表示.部分彩色模型跟具体硬件相关,这里仅讨论数学模型.
### RGB
[Red/Green/Blue]人眼感受光主要由锥状细胞,分别对红绿蓝光做出感受，波长吸收峰值分比为575nm,535nm,445nm.在硬件中也容易对此颜色空间做出控制.
RGB构成笛卡尔坐标系,立方体斜对角线是灰度级. 具体表示时,一般每个像素采用3个字节表示,但也有采用2个字节表示,用5,6,5位(RGB565)和5,5,5位(RGB555)表示各个分量
### CMYK
是RGB的补色，也是颜料的三原色(对应颜色原料吸收RGB).为克服将CMY混合产生黑色不纯，单独设置黑色通道K,常用语打印机设备.
### HSV/HSI
[Hue/Saturation/Value]符合人对颜色的解释.亮度表达了无色的强度概念,色调是光波混合中与主波长有关的属性,与表示观察者感知的主要颜色(赤橙黄绿蓝靛紫),饱和度指的是相对纯净度,或一种颜色混合白光的数量.色调和饱和度一起称为色度
将RGB的笛卡尔坐标系以其斜对角线为轴垂直放置,与颜色的平面与红色的夹角为色度,轴与颜色的距离为饱和度.颜色位于轴的高度为I,RGB的最大值为V
### YUV/YCbCr/YUV420/NV12/NV21/I420/YV12
Y代表的是亮度，UV代表的是色度,其他都是YUV的存储方式变种，硬解码一般都会输出这种格式.
Y = 0.299R+0.587G+0.114B
Cb = 0.564(B-Y)  里面含0.5B,范围在-127~127
Cr = 0.713(R-Y)  里面含0.5R,范围在-127~127
UV分别是Cb+128与Cr+128
- 视觉系统对亮度更敏感而对色度不敏感，因此可以压缩色度通道起到压缩的作用.
  - 4:4:4表示完全取样
  - 4:2:2表示2:1的水平取样，垂直完全采样
  - 4:2:0表示2:1的水平取样，垂直2:1采样
  - 4:1:1表示4:1的水平取样，垂直完全采样。
- 根据存放形式不同，还有plane(YUV分别存放,对应CxHxW)和pack(交替存放，对应HxWxC)形式
  - YU12(I420):420,先U后V的plane形式
  - YV12:420,先V后U的plane形式
  - NV12(YUV420sp):420,Y plane,先U后V的pack(Semi-Planar)形式
  - NV21:420,Y plane,先V后U的pack形式
  - YUYV(V422,YUNV,YUY2): 422, 按照YUYV的pack形式
  - UYVY(Y422,UYNV): 422, 按照UYNV的pack形式
### 其他彩色空间及相互转换
除上述以外还有XYZ,Lab等颜色空间.不再详述.均可由```cv::cvtColor(InputArray src, OutputArray dst, int code, int dstCn=0)[or dst=cv.cvtColor(src, code[, dst[, dstCn]]) or PIL.Image.Image.convert(mode=None)，"L"(gray)/"RGB"/"CMYK/YCbCr"```转换，其中code标记转换的两个表示空间，可以在RGB/GRAY/XYZ/YCrCb/HSV/Lab/Luv/HLS(HSI)/YUV family(上面列出的420和422格式)
## 图像变换与操作
### 像素级变换

### 空间滤波与卷积

### 形态学
- 两种基本操作
    - 腐蚀
    - 膨胀
- 导出操作
    - 开操作
    - 闭操作
    - 击中击不中变换
- 一些常见使用场景

### opencv基本数据结构

## 图像压缩与输入输出
若将图片在某个颜色空间的数组全部存储将消耗太多存储资源,考虑到编码(不需要都采用a完整字节表示),时间空间(时空上具有一定可预测性),不相关信息(可以取消对视觉效果影响不大的存储)的冗余，有很多压缩方法。图片类型的不同，采取的压缩方式也不同。
### 一些主要的压缩方法：
- 霍夫曼编码：霍夫曼编码采用变长编码,图片中出现概率高的像素值采用短的编码,概率小的采用长编码,使得总体上编码长度缩短，平均像素编码长度接近于图片的信息熵(香农第一定理).采用霍夫曼树的构建方式即可.
- 行程编码:基本思想是存储像素值和连续像素数量,在存储二值图像比较有用.
- 块变换编码:将一张图片分成多个NxN的不重叠小块,分别使用可逆二维线性变换,对变换后的系数进行截取,量化,编码. 二维线性变换可选取DFT,DCT(离散余弦变换)等.截取时可按照系数方差(区域编码,方差大的系数带有更多信息)或者模(阈值编码,更加常用)大小排序后取前几个.量化可以具有全局阈值/每个子图采用不同阈值，使得子图保留的系数个数相同(最大N编码)/每个子图像每个系数不同(跟量化结合起来,不同位置使用不同的基数求余。缩放量化表格可以使结果具有不同的量化阶梯高度,调节压缩率)
- 预测编码:基本思想是采用空间或时间上的线索预测图像(e.g.使用前面像素/图像的线性组合),对预测残差进行编码,由于残差的信息熵较原来小,可以使编码长度更短.其中视频一般采用运动补偿编码:独立帧(I)，类似于jpeg,编解码不依赖于其余帧,是生成预测残差的理想起点,可以阻止误差累积.I帧周期性地插入到视频码流中.视频帧被分成NxN的宏块,编码每个宏块相对于前帧的运动向量.基于前一帧的编码帧为P帧(预测称),基于后一帧的编码帧为B帧(双向帧).基于L1的块匹配用于计算运行向量,该向量也可用DCT系数进行量化.
### 一些主要多媒体文件的压缩标准:
- JPEG:对图像使用YUV420描述,使用块变换编码,将图像分成不重叠的8X8像素块,每个像素减均值(128)后进行DCT后使用由视觉效果确定的量化基数表格(8x8,亮度和色度不同)进行量化,将量化后系数进行Zig-Zag排列成一维，进一步使用行程编码(非零AC的系数和前面0系数的个数),规范为这种形成编码二元组制定了霍夫曼编码表,DC系数系数是相对前一副图像的差值，同样也有霍夫曼编码表.
- MPEG-4运动补偿编码,运动向量精度1/4像素.P,B帧宏块16x16(可变),I帧变换块8x8(DCT),.H.264在I帧内还用空间预测编码,且不使用整数变换(可变变换块)而不是DCT
### 多媒体文件IO
工具都已经将压缩/解压缩集成,对于文件IO来说是不可见的，工具自动按照文件**内容**进行相应的解码,根据后缀进行编码.
- opencv文件IO和编辑
    - 读图片 `Mat cv::imread(const String& filename,int	flags = IMREAD_COLOR)` 按照**HxWxC,其中C中BGR**的顺序解码. *notes* opencv-python读取中文路径的图片时会有问题,使用`cv2.imdecode(np.fromfile(filename,dtype=np.uint8),cv2.IMREAD_UNCHANGED)`
    - 写图片 `bool cv::imwrite(cconst String& filename,InputArray img,const std::vector< int >& params = std::vector<int>())`,
    - 读视频 `VideoCapture& videoCapture.operator>>(Mat &image)`/`bool videoCapture.read(OutputArray image) `/`bool videoCapture.grab (); bool videoCapture.retrieve(OutputArray image, int flag=0)`
    - 写视频 `VideoWriter& videoWriter.operator<<(const Mat& image)`/`void cv::VideoWriter::write(InputArray image)`
    - 文字   `cv::putText(InputOutputArray img,const String& text,Point	org,int	fontFace,double fontScale,Scalar color,int thickness = 1)`                
    - 画框   `cv::rectangle(InputOutputArray img,[Point pt1,Point pt2]/Rect rec,,const Scalar& color,int thickness = 1)` 
- PIL文件IP和编辑
    - 读图片 `PIL.Image.open()` 
    - 写图片 `PIL.Image.save()`
    - 文字 `PIL.ImageDraw.Draw(PIL.Image).text(position, string, options[font])`   #这里再细化一下
    - 画框 `PIL.ImageDraw.Draw(PIL.Image).rectangle(box, options[outline])`
## 背景提取

### 混合高斯模型背景建模(参数化模型)
认为每个像素点在过去一段时间内符合混合高斯分布
1.初始化。对第一帧，以随机像素值为均值,给定方差,建立K个高斯模型，权重均为1/k  K一般取3~5
2.更新。匹配高斯分布(以小于D个标准差为判据,D一般取2.50-3.5)，
若匹配，则
    w = (1-a)*w+a
    mean = (1-p)*mean+p*X
    std2 = (1-p)*std2+p*(X-mean)**2
    a为学习率 p = a* gaussian(X)
若不匹配，则
w = (1-a)*w
若所有模式都不匹配，则
创建新的高斯分布，以该像素值为均值，给定方差，替换掉权重最小的高斯分布
最后对所有权重进行归一化
3.预测。按照w/std从大到小排序，并且给定背景所占比例T(T>0.7), 当对权重求cumsum时达到T时，匹配分布在T以内高斯时，为背景，否则为前景
### ViBe(Visual Background Extractor)
认为每个背景像素值在一个样本集在领域内
1.初始化  对于一个像素点，随机的选择它的邻居点的像素值作为它的模型样本值
2.更新 保守的更新策略+前景点计数方法。保守的更新策略前景点永远不会被用来填充背景模型。前景点计数：对像素点进行统计，如果某个像素点连续N次被检测为前景，则将其更新为背景点。
连续多次选取前景的像素更新为背景，每一个背景点有φ的概率去更新自己的模型样本值，同时也有φ的概率去更新它的邻居点的模型样本值。
在选择要替换的样本集中的样本值时候，随机选取一个样本值进行更新
3.预测。
比较样本集合中各点与预测点L2,统计符合条件的点数，小于阈值时为前景，否则为背景


## 图像特征点和描述
## 直方图
### 概念和计算方式
### 在跟踪上的应用:meanshift
## 光流

### 稀疏光流

### 稠密光流

### 采用深度学习计算光流
    Fischer P., Dosovitskiyz A. , Ilgz E., et al, FlowNet: Learning Optical Flow with Convolutional Networks. ICCV 2015
    Ilg  E., Mayer N., Saikia T., et.al. FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks, CVPR 2017
    Zhu Y., Lan Z., Newsam S., Hauptmann A,Hidden Two-stream Convolutional Networks for Action Recognition. ACCV 2018
## 相机模型与立体视觉
### 相机内参数
### 相机畸变
### 外参数矩阵与Rodrigues变换
### 仿射变换与透视变换
### 双目相机标定
### 深度图



参考书籍:
学习OpenCV3
数字图像处理(第三版)
https://docs.opencv.org/master/
http://effbot.org/imagingbook/pil-index.htm



