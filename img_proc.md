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
#### 直方图
直方图是对图像像素级概率建模的手段,表示为像素在图像中出现的概率,并且不含空间信息.对于偏暗的图像,小像素值概率较高,对比度强的读像素值分布范围较广.
- 直方图规定化
为了达到特定的效果，达到指定的直方图,可对像素进行变换.根据概率的观点,就是寻找一个变换函数r->s[记为H]，使概率密度函数从原来的f_r(r)变换成f_s(s)
根据 
$$
\int^{r_0}f_r(r)dr=\int^{H(r_0)}f_s(s)ds -> F_r(r_0)=F_s(H(r_0)) 可得H(r)= F_s^{-1}(F_r)
$$
特别地当想要直方图均衡化时(指定的直方图是均匀分布),变换函数为F_r,即按照概率分布函数进行变换.
- 计算直方图和直方图均衡的方式
`void cv::calcHist(InputArrayOfArrays images,const std::vector<int>& channels,InputArray mask,OutputArray hist,const std::vector<int >& histSize,const std::vector<float>& ranges,bool accumulate = false)` 类似还有将其中vector换成C风格的指针和指定个数的dims参数的重载版本. 返回的hist是channel.size()维的,每个元素是各个维度上的联合分布像素个数
在opencv可以直接进行直方图均衡化:
`void cv::equalizeHist(InputArray src,OutputArray dst)`
还可以进行将直方图直接投射到原图上:(空间像素转换为该像素[属于一个bin中]的概率),通过追踪目标框hue的直方图投影的重心,即可实现meanshift以及camshift跟踪.
`void cv::calcBackProject(InputArrayOfArrays images,const std::vector<int>& channels,InputArray hist,OutputArray dst,const std::vector<float>& ranges,double scale)`类似还有将其中vector换成C风格的指针和指定个数的dims参数的重载版本.
`PIL.Image.histogram()` 返回的是各个像素的个数,多通道时是单个通道边际分布叠加形成的一维列表,在PIL中别的相关操作需要根据得到的直方图自行实现.
### 空间滤波与卷积
卷积作为一种线性操作,是图像空间滤波的概括表述:
$$ f \star b(x,y) = \Sigma_{s=-a}^{a}\Sigma_{t=-b}^{b}{w(s,t)f(x+s,y+t)} $$
其中卷积核关于原点的反转已经省略(实际上上实定义了相关操作,但反转操作只是卷积核表示的问题,可以等价起来.这种定义方式也convolution neural neworks中定义方式) 卷积核都是单通道的,处理多通道图片仅在C维上做广播.
`void cv::filter2D(InputArray src,OutputArray dst,int ddepth,InputArray kernel,Point anchor=Point(-1,-1),double delta=0, int borderType=BORDER_DEFAULT)`
opencv中还有大量专用的空间滤波函数,如blur,GaussioanBlur,bilateralFilter,Sobel,Laplacian等等.
`PIL.Image.filter(PIL.ImageFilter.Kernel(size, kernel, scale=None, offset=0))`
### 形态学(morphology)
- 两种基本操作
    - 腐蚀: 去除图片中小于结构元b尺寸的小高亮区域
    $$ f\ominus b(x,y) = min_{(s,t)\in b}{f(x+s,y+t)} $$
    - 膨胀: 联结小于结构元b尺寸的断开小区域
    $$ f\oplus b(x,y) = max_{(s,t)\in b}{f(x+s,y+t)} $$
    - 腐蚀和膨胀的关系: 对ROI腐蚀就是对ROI补集的膨胀后再取补集  
` void cv::dilate(InputArray src,OutputArray dst,InputArray kernel,Point anchor=Point(-1,-1),int iterations=1,int borderType=BORDER_CONSTANT,const Scalar& borderValue=morphologyDefaultBorderValue())`	
` void cv::erode(InputArray src,OutputArray dst,InputArray kernel,Point anchor=Point(-1,-1),int iterations=1,int borderType=BORDER_CONSTANT,const Scalar& borderValue=morphologyDefaultBorderValue())`
`PIL.Image.filter(PIL.ImageFilter.MinFilter(size))`
`PIL.Image.filter(PIL.ImageFilter.MaxFilter(size))`
- 导出操作
    - 开操作: 先腐蚀再膨胀,消除向外凸的小尖角(空间尺寸上和灰度高度上)
    - 闭操作: 先膨胀在腐蚀,消除向内凹的小尖角(空间尺寸上和灰度高度上),可以看成闭操作的互补
    - 击中击不中变换: 用两个结构元,一个对图形腐蚀,一个对图形补集膨胀,交集部分.
    - 帽顶帽底变换: 冒顶变换就是图形减去开操作，剩下向外凸的小尖角，帽底就是闭操作减去图形，剩下向内凹的小尖角.
统一接口`void cv::morphologyEx(InputArray src,OutputArray dst,int op,InputArray kernel,Point anchor=Point(-1,-1),int iterations=1,int borderType=BORDER_CONSTANT,const Scalar& borderValue=morphologyDefaultBorderValue())`		
`op = MORPH_ERODE/MORPH_DILATE/MORPH_OPEN/MORPH_CLOSEMORPH_GRADIENT/MORPH_TOPHAT/MORPH_BLACKHAT/MORPH_HITMISS`其中`iterations`指复合操作中基本操作的次数，而不是复合操作的次数,因为根据定义性质，复合操作重复多次并没有用.
PIL中只能复合基本操作来实现
### 几何变换
#### 经典二维插值方法:bilinear
一维情形时,对于整数格点f(x1),f(x2)之间分数坐标的点f(x),由拉格朗日插值公式得到 (x-x1)f(x2)/(x2-x1)+(x2-x)f(x1)/(x2-x1) = (x-x1)f(x2)+(x2-x)f(x1)  [x2-x1=1]
二维时对各维分别做线性插值,得到f(x,y) = (x-[x]])f([x]+1,y)+([x]+1-x)f([x],y) = (x-[x]){(y-[y])f([x]+1,[y]+1)+([y]+1-y)f([x]+1,[y])} +([x]+1-x){(y-[y])f([x],[y]+1)+([y]+1-y)f([x],[y])} = (x-[x])(y-[y])f([x]+1,[y]+1)+(x-[x])([y]+1-y)f([x]+1,[y])+([x]+1-x)(y-[y])f([x],[y]+1)+([x]+1-x)([y]+1-y)f([x],[y])
在对整张图进行变换时,一般而言是反向进行的,即目标图映射回原图中,利用原图整数格点进行插值,得到目标图上的像素值.使用的场合包括变形(resize),仿射变换(affine),透视变换(perspective)等
#### 齐次坐标
将一个原本是n维的向量用一个n+1维向量来表示,二维点(x,y)的齐次坐标表示为(hx,hy,h),计算得到的齐次坐标点(x,y,z)与(x/z,y/z,1)是同一个点.投影时,在一个投影平面上z都取1.因此透视变换其实只有8个自由变量. (x,y,1) -> (x',y',z') <-> (x'/z',y'/z',1). 而仿射变换因要求保持变换前后平行的关系,确定变换前后3个点后,第4个点便确定了,因此只有6个自由变量.
#### opencv中api
`void cv::resize(InputArray	src,OutputArray	dst,Size dsize,double fx = 0,double	fy = 0,int interpolation = INTER_LINEAR)`,其中dsize/fx,fy需要指定其中一套即可.	
affine()
perspective()
#### PIL中api
`PIL.Image.resize(size,filter)` filter可取`Image.NEAREST(default)/Image.BILINEAR/Image.BICUBIC/Image.ANTIALIAS`
`PIL.Image.transfrom(size, method, data, filter)` 
method可取:
Image.EXTENT:即实现crop+resize，data是原图中需要crop box的两个点
Image.AFFINE:此时data为包含6个元素的从输出到输入的仿射变换矩阵
Image.QUAD:此时data包含原图4个顶点坐标(8个值),形变至size的输出.
Image.PERSPECTIVE:此时data包含8个元素的从输出到输入的透视变换矩阵(T33已经归一化为1)
### opencv/PIL基本数据结构
#### Opencv 核心数据结构
Mat封装了稠密张量数据结构,采用引用计数进行内存管理,当内存引用次数归零后内存自动释放.
几种数据结构的继承关系:
Mat <- Mat_<_Tp> 增加了圆括号索引,对于常用的形状和数据类型有很多别名: 类型有unchar(b),double(b),int(i),short(s),float(f),元素类别是数字时为1,Vec时数字为Vec元素个数,例如 Mat_<float> 即 Mat1f，Mat_<Vec3b>即 Mat3b. 作为特化的Mat，它的一些函数不需要再加类名.例如 Mat::at<_Tp>(row,col) 可以直接写成 Mat::at(row,col).
Matx<_Tp,m,n> 小矩阵模板,能初始化1,2,3,4,5,6,7,8,9,10,12,14,16个元素，别名为Matx{Tp}{m}{n} <- Vec<_Tp,cn> (即令基类n=1) 增加了方括号索引,也具有别名VecTpcn <- Scalar_<_Tp> (即令基类cn=4)
除此以外还有Rect_<_Tp>,Size_<_Tp>,Point_<_Tp>,不加模板时均是_Tp为int时的别名.
Ptr<_Tp>提供了类似shared_ptr<_Tp>的智能指针,将自我进行内存管理. Mat::ptr<_Tp>(row)将返回指向row单个元素的指针.MatConstIterator,MatIterator作为迭代器,会自动跳过图片存储中pitch和width之间的gap.
#### PIL 数据结构
`Image.fromarray(numpy.ndarray)  <->  numpy.asarray(PIL.Image)`可使图片与numpy互相转换.
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
  - 文字 `PIL.ImageDraw.Draw(PIL.Image).text(position, string, options[outline,font])   #outline 颜色"rgb(red%,green%,blue%)"/hsl(hue,saturation%,lightness%)",font是PIL.ImageFont实例,可以加载ttf文件：ImageFont.truetype("*.ttf",size)`
  - 画框 `PIL.ImageDraw.Draw(PIL.Image).rectangle(box, options[outline,fill])  #fill内部填充颜色`

## 背景建模

背景建模是一种分割静止背景和运动前景的方法.可用于提取运动缓慢的目标.

### 混合高斯模型背景建模(参数化模型)

每个像素以混合高斯分布建模.
1.初始化:对第一帧，以随机像素值为均值,给定方差,建立K个高斯模型，权重w均为1/K,K一般取3~5
2.更新:匹配高斯分布(以小于D个标准差为判据,D一般取2.50-3.5)，
若匹配，则 [a为学习率 p = a*gaussian(X)]
  更新权重:w = (1-a)*w+a
  更新均值方差:mean = (1-p)*mean+p*X; std2 = (1-p)*std2+p*(X-mean)**2
若不匹配，则
  更新权重:w = (1-a)*w
若所有模式都不匹配，则
  创建新的高斯分布替换掉权重最小的高斯分布,以该像素值为均值,给定一个方差.
最后对所有权重进行归一化
3.预测:按照w/std从**大**到**小**排序,对其求cumsum,达到给定背景所占比例T(T>0.7)时,匹配分布在T以内高斯的像素为背景，否则为前景.

### ViBe(Visual Background Extractor,非参数化模型)

    O. Barnich and M. Van Droogenbroeck. ViBe: A universal background subtraction algorithm for video sequences. In IEEE Trans. Image Processing, 2011.
    Brutzer S , Hoferlin B , Heidemann G . Evaluation of background subtraction techniques for video surveillance, CVPR 2011
认为每个背景像素以邻域像素组成的样本集表示,通过概率更新样本集和距离阈值来判断.
1.初始化:对于一个像素点，随机地选择它的邻域像素值作为它的模型样本值
2.更新:保守的更新策略或前景点计数方法。每一个背景点有φ的概率去更新自己的模型样本值，同时也有φ的概率去更新它的邻居点的模型样本值。在选择要替换的样本集中的样本值时随机选取一个样本值进行更新
保守的更新策略：前景点永远不会被用来填充背景模型。
前景点计数：对像素点进行统计，如果某个像素点连续N次被检测为前景，则将其更新为背景点。
3.预测。
比较像素点与其样本集合中各点的L2距离,统计符合条件的点数,小于阈值时为前景,否则为背景.

### opencv中提供的接口
<opencv2/video/background_segm.hpp>
<opencv2/bgsegm.hpp>
Video analysis/Improved Background-Foreground Segmentation Methods

## 图像特征点和描述

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



参考资料:
[1]学习OpenCV3
[2]数字图像处理(第三版)
[3]https://docs.opencv.org/master/
[4]http://effbot.org/imagingbook/pil-index.htm



