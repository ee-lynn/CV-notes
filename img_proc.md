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

### 直方图

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
稠密仿射变换:`void cv::warpAffine(InputArray src,OutputArray dst,InputArray M,Size dsize,int flags = INTER_LINEAR,int borderMode=BORDER_CONSTANT,const Scalar&	borderValue=Scalar())`		 flags除了设置插值方式外,还可以组合`WARP_INVERSE_MAP`表示M(2x3)矩阵是dst至src的仿射变换矩阵
稀疏仿射变换:`void cv::transform(InputArray src,OutputArray dst,InputArray m)` *note* m作用在输入的channel维度,当维度相同时,不补1,否则用齐次坐标. m可以是2x2或3x3. src 可以1-4通道.	
获得仿射变换矩阵:`Mat cv::getAffineTransform(InputArray src,InputArray dst)` src和dst提供三个2维点
特别地旋转作为一种特殊的仿射变换:`Mat cv::getRotationMatrix2D(Point2f center,double angle,double scale)`
稠密透视变换`void cv::warpPerspective(InputArray src,OutputArray dst,InputArray M,Size dsize,int flags = INTER_LINEAR,int borderMode=BORDER_CONSTANT,const Scalar& borderValue = Scalar())`		M为3x3单应性矩阵,最后的结果将重新化为齐次坐标Z=1的平面.
稀疏透视变换`void cv::perspectiveTransform(InputArray src,OutputArray dst,InputArray m)` m可以是3x3(2D点)/4x4(3D点)
获得透视变换矩阵:`Mat cv::getPerspectiveTransform(InputArray src,InputArray dst,int solveMethod=DECOMP_LU)`	src,dst提供四个2维点
`findHomography`是一种优化版本,即提供大于四个点，解超定方程.
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

   Zoran Zivkovic, Ferdinand van Heijden. Efficient adaptive density estimation per pixel for the atsk of background subtraction. 2006

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
认为每个背景像素以邻域像素组成的样本集表示,通过概率更新样本集和距离阈值来判断.
1.初始化:对于一个像素点，随机地选择它的邻域像素值作为它的模型样本值
2.更新:保守的更新策略或前景点计数方法。每一个背景点有φ的概率去更新自己的模型样本值，同时也有φ的概率去更新它的邻居点的模型样本值。在选择要替换的样本集中的样本值时随机选取一个样本值进行更新
保守的更新策略：前景点永远不会被用来填充背景模型。
前景点计数：对像素点进行统计，如果某个像素点连续N次被检测为前景，则将其更新为背景点。
3.预测。
比较像素点与其样本集合中各点的L2距离,统计符合条件的点数,小于阈值时为前景,否则为背景.

### opencv中提供的接口
opencv中的背景建模在`<opencv2/video/background_segm.hpp>`共同的基类为`cv::BackgroundSubtractor`提供了`virtual void cv::apply((InputArray image,OutputArray fgmask,double learningRate=-1)`，其子类有:
cv::BackgroundSubtractorKNN(非参数模型),cv::BackgroundSubtractorMOG2(参数模型)
调用方式均是先createBackgroundSubstractor<xxx>(param),然后调用apply得到前景fgmask.创建后还可以get/set各种参数.

## 图像特征点和描述

### 特征点(角点)

- 一些具有辨识能力的点,特别地可应用到多视角图片的匹配等；
harris测度:一个点作为关键点,位置移动一下像素会变化很多.基于这个思想构造
$$
\Sigma x,y \in (w,h) w_xy[I(x+\delta x,y+\delta y)-I(x,y)]^2
$$
在一个小窗口(w,h)内，在w_xy系数加权下L2距离. 线性化后展开后可以表示为\delta x,\delta y的二次型，其hessian阵H的特征值表征了角点好坏. harris测度定义为
$$
det(H)/tr^2(H)
$$
即两个特征值的比值,太小时不是一个好的角点(Harris),或者仅判断较小的特征值大于一定阈值就是一个好的特征点(Shi-Tomasi)
- opencv api 
`void cv::goodFeaturesToTrack(InputArray image,OutputArray corners, int maxCorners,double qualityLevel,double minDistance,InputArray mask=noArray(),int blockSize=3,bool useHarrisDetector=false,double k=0.04)`就是找出这样的角点,其中使用评价标准在`minDistance`做非极大值抑制(NMS),最多输出`maxCorners`个角点,`qualityLevel`是最低可接受的评价标准与最好角点的评价标准的比值.`blockSize`是计算角点hessian时使用的窗口大小`useHarrisDetector`表示使用评价标准是harris还是Shi-Tomasi,`k`是阈值.
- 找到角点后,还需要构造一个特征向量来表征这个角点,成为特征描述符.

### 尺度不变特征转换(Scale-invariant feature transform或SIFT)

SIFT描述符具有平移、缩放、旋转不变性，同时对光照变化、仿射及投影变换也有一定的不变性,由四部分构成.
- 高斯差分（DoG）滤波：搜索所有尺度上的图像位置。通过高斯微分函数来识别潜在的对于尺度和旋转不变的兴趣点。
  - 高斯模板在空间方向是可分的,二维模板被拆成了x,y两个方向的两个模板,加快计算.
  - 形成金字塔时,产生几组(octave,共o组)为一个分辨率,每组内又有几层(interval,共s层).每一组都是上一组倒数第3层的降采样1倍(选取原因见方差分布规律).
  - 对高斯平滑的图像做拉普拉斯算子,可以近似用不同方差的高斯算子得到.因此在每组内相邻层做差分,得到差分图.因下一步需要确定极值需要上下图，因此差分图需要n+2个,对应地,需要s=n+3层
  - 一组内方差逐次放大2^(1/n)倍.因共有s=n+3层,倒数第三层对应正好是2*\sigma,降采样一倍后正好恢复方差.
  - 两个高斯滤波级联,等价于方差和的方差高斯滤波. 摄像机拍摄时会对图像形成一定的平滑,定义这种平滑方差为0.5,规定第一组第一层方差为1.6,则对摄像机拍摄的图像直接做方差为1.52的高斯滤波即可.实际操作中,为得到更多特征点,将图像resize上采样一倍,作为金字塔低端,此时相机方差为1,第一层方差为1.25(\sqrt{1.6^2-1^2})

- 尺度空间的极值检测和关键点位置确定：对DoG金字塔中的每一层，进行尺度空间的极值检测(极大值[正]和极小值[负])，把每一个极值点作为候选点，在每个候选的位置上，通过一个拟合精细的模型来确定位置和尺度。关键点的选择依据于它们的稳定程度。
  - step 1: 在某一层上选择极值,与该层8邻域,上下层9个点比较,得到极值
  - step 2: 采用牛顿法迭代得到亚像素的极值点(关于x,y,\sigma三种变量).牛顿法迭代时需要采用差分求梯度和hessian.其中亚像素偏移大于0.5格时更换插值起点.过滤掉迭代次数超越限制,极值绝对值小于阈值的极值点.
   - 使用Harris测度过滤掉测度小于0.1的特征点

- 关键点方向确定：基于图像局部的梯度方向，分配给每个关键点位置一个或多个方向。所有后面的对图像数据的操作都相对于关键点的方向、尺度和位置进行变换，从而提供对于这些变换的不变性。
  - 以特征点所在层的1.5\sigma 窗口内统计像素的梯度和方向.梯度计算采用中心差分.梯度采用方差为1.5\sigma 的高斯加权
  - 计算梯度方向的直方图,统计的是加权后的梯度模.直方图的峰值作为该特征点的方向.对于多峰直方图,次峰大于主峰80%的也作为方向,此时将该特征点复制一份,增强特征的鲁棒性.当然也需要在直方图中插值(二次插值)得到具体多少度.
- 构建关键点特征描述符：在每个关键点周围的内，在选定的尺度上测量图像局部的梯度。这些梯度被变换成一种表示，这种表示允许比较大的局部形状的变形和光照变化。
  - 得到特征点h(x,y,\theat)后,在图上画BxB(4x4)个网格,每个网格由3\sigma个像素点组成.该网格旋转至\theta(所有像素均由插值得到).并且计算BxB网格内每个网格的梯度方向直方图(以特征点距离为高斯加权,直方图均分为8个方向),得到8xBxB维描述符.
  - 为了去除光照变化的影响，需要对它们进行归一化处理，对于图像灰度值整体漂移，图像各点的梯度是邻域像素相减得到，所以也能去除。非线性光照，相机饱和度变化对造成某些方向的梯度值过大因此设置门限值(向量归一化后，一般取0.2)截断较大的梯度值。然后再进行一次归一化处理，提高特征的鉴别性。
- SIFT与其他特征的关系
  - 构造描述符时,采用的就是HOG的方式,可以把SIFT理解为尺度和方向自适应的HOG.
  - 因在过滤特征点时也用到了Harris测度，因此SIFT也是一种Harris角点
  - SURF是改进版的SIFT:框架与SIFT相同,只不过细节上用计算更加高效的技术来替代.如不构建金字塔而是用一组不同大小的盒式滤波器,微分用Haar小波代替,描述符仅分为4个分量,x,y梯度和与x,y梯度绝对值和等等
- opencv api
  - `<opencv2/features2d.hpp>`中各种形式的特征都有统一的基类接口`cv::Feature2D`
使用`static Ptr<xxx>create(params)`来创建,有基类实现多态.
  - `virtual void detect(InputArray image,std::vector<KeyPoint>& keypoints,InputArray mask=noArray())`
  `virtual void detect(InputArrayOfArrays images,std::vector<std::vector<KeyPoint>>&keypoints,InputArrayOfArrays masks=noArray())`
  detect计算特征点,第二个是第一个的批量计算版本
  - `virtual void compute(InputArrayOfArrays images,std::vector<std::vector<KeyPoint>>&keypoints,OutputArrayOfArrays descriptors)`
  `virtual void compute(InputArrayOfArrays images,std::vector<std::vector<KeyPoint>>&keypoints,OutputArrayOfArrays descriptors)`
  compute根据特征点计算描述符(特征向量),第二个是第一个的批量计算版本
  - `virtual void detectAndCompute(InputArray image,InputArray mask,std::vector<KeyPoint >& keypoints,OutputArray descriptors, bool useProvidedKeypoints=false)`
  将detect和compute一起做了,一般情况需要描述符时直接调用这个函数更加高效,不需要重复计算中间变量.
  -与特征相关的概念还有
  `cv::KeyPoint()`表示特征点的基本类型,`CV:::DMatch`表示特征匹配的基本类型,
`cv::DescriptorMatcher`用于匹配特征点的基类,`void cv::drawKeypoints(InputArray image, const std::vector<KeyPoint> &keypoints, InputOutputArray outImage, const Scalar& color=Scalar::all(-1), DrawMatchesFlags flags=DrawMatchesFlags::DEFAULT)`可视化特征点

## 光流
光流算法的理想输出是两针图像中每个像素的位移矢量。图像中每个像素都是用这种方法,则通常称其为"稠密光流",仅仅对图像中某些点的子集计算则被称为"稀疏光流".
opencv中光流计算均在`<opencv2/video/tracking.hpp>`中
### 稀疏光流
- Lucas-Kanade算法的三个假设:(1)亮度恒定:像素值加上速度后值恒定;(2)时间较快,运动微小:所求位移矢量就是速度矢量;(3)空间一致性:空间小窗口内位移矢量相同,用于正则化，环节孔径效应(透过一个小孔观察更大无图的运动,无法得知真实运动信息,只知道垂直边缘的运动速度).
$$
\nabla I \cdot u=-dI/dt  (由假设1,2导出)
$$
结合假设3,在小窗口(例如5x5)中各个像素均满足以上等式,用最小二乘法求的速度矢量.
迭代:因为像素不变假设,复用空间梯度.使用计算得到的矢量叠加上原始图像得到中间图像,修正像素时间梯度,修正计算速度矢量
$$
u=u_{pre}-(\nabla I)^{-1} dI/dt
$$
金字塔式算法:
构建图像金字塔后,由粗到细逐层计算,粗尺度的计算结果上采样后作为细化层的初值估计.
- opencv api
稀疏光流的共同基类为`cv::SparseOpticalFlow`,计算的统一接口为:`virtual void cv::SparseOpticalFlow::calc(InputArray prevImg,InputArray nextImg,InputArray prevPts,InputOutputArray nextPts,OutputArray status OutputArray err=cv::noArray())`
其中金字塔LK光流类为:`cv::SparsePyrLKOpticalFlow`.
`static Ptr<SparsePyrLKOpticalFlow> cv::SparsePyrLKOpticalFlow::create(Size winSize=Size(21, 21),int maxLevel=3,TermCriteria crit=TermCriteria TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),int flags=0,double minEigThreshold=1e-4)` 指定窗口大小,金字塔层数,迭代结束条件,和flag = OPTFLOW_USE_INITIAL_FLOW | OPTFLOW_LK_GET_MIN_EIGENVALS,特征值用于评估最小二乘法问题的奇异性(或者理解为harris测度过滤不好的跟踪点)
另外一个接口函数直接计算
`void cv::calcOpticalFlowPyrLK(InputArray prevImg,InputArray nextImg,InputArray prevPts,InputOutputArray nextPts,OutputArray status,OutputArray err,Size winSize=Size(21, 21),int maxLevel=3,TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01),int flags=0,double minEigThreshold=1e-4)`.		
### 稠密光流
- 总变分法(使得匹配后的两张图像上每个点尽可能相同_
$$
\min_{u(x,y),v(x,y)} \int \psi(I(x,y)-I(x+u,y+v))dxdy
$$
其中$$\psi$$是误差函数,根据不同需求可取L1,L2函数.
  - 对纯色区域只要求尽可能匹配是不够的,还需要加入正则项
$$
\min_{u(x,y),v(x,y)} \int \phi(\nabla u, \nabla v)dxdy
$$
于是总体函数为
$$
\min_{u(x,y),v(x,y)} \int [\psi(I(x,y)-I(x+u,y+v))+\lambda \phi(\nabla u, \nabla v)]dxdy
$$
  - dual TV-L1

        Zach C , Pock T , Bischof H . A Duality Based Approach for Realtime TV-L1 Optical Flow, 2007.

早期的H-S算法是上述优化问题中误差函数都定义为L2,再使用变分法得到欧拉方程经过数值解法解,效果不是很好.TV-L1将误差函数都设为L1，且将上述问题拆成两个问题.
引入新优化变量v，优化问题变成
$$
\min_{u,v} \int [|I(x+u)-I(x)|+\lambda |\nabla u|+(u-v)^2/\theta]dxdy
$$
再次将I(x+u)-I(x)在x+v附近用一阶tailor展开成线性 \rho =I(x+v)+dI(x+v)/dx(u-v)-I(x),代入优化目标.因为u与v很接近,采用交替求解的方式:

  - step1:固定v，求u
$$
\min_{u} \int [\lambda |\nabla u|+(u-v)^2/\theta]dxdy
$$

  - step2:固定u，求v
$$
\min_{v} \int [|\rho (v)|+\lambda (u-v)^2/\theta]dxdy
$$
第二个子问题是pixelwise的,可以分别求解.求解方式就是讨论\rho的符号,就可以得到解析解(实际上在数值上根据u得到v的过程中,\rho绝对值小于一定阈值就直接认为是0,该阈值由更新步长之间比较可以得出),记为为TH操作
第一个子问题是denoising问题,有现成的方法.迭代公式为:
$$
u = v - \theta div p  \quad  p\in R^{2} \\
p = (p+\tau/\theta\nabla u))/(1+\tau/\theta|\nabla u|)
$$
实际求解时候,构建金字塔结构,在最粗的尺度上初始化u=0,每个尺度上u用上个尺度resize初始化,v初始化为u,p初始化为(0,0),反复使用TH->更新v->更新p 迭代固定次数得到一个尺度的光流.
- opencv api
稠密光流共同基类为`cv::DenseOpticalFlow`,统一接口为`virtual void cv::DenseOpticalFlow::calc(InputArray I0,InputArray I1,InputOutputArray flow)`
可以使用的方法有`cv::DISOpticalFlow; cv::FarnebackOpticalFlow;  cv::optflow::DualTVL1OpticalFlow; cv::VariationalRefinedment`
- 采用深度学习计算光流

      Fischer P., Dosovitskiyz A. , Ilgz E., et al, FlowNet: Learning Optical Flow with Convolutional Networks. ICCV 2015
      Ilg  E., Mayer N., Saikia T., et.al. FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks, CVPR 2017

  - FlowNet使用CNN有监督学习来预测光流，输入为前后帧,输出光流.
  FlowNetS(imple):就是encoder-decoder,先降采样后升采样,最终得到光流输出.其中升采样feature map还会concat对应大小降采样的feature map 
  FlowNetC(orrelation):考虑光流的定义,引入了correlation操作去匹配特征
  对于featuremap1中中心位置x1(stride s1),计算在D=2d+1邻域内featuremap2中中心位置x2(stride s2)的卷积,卷积核大小K=2k+1.输出结果安排为([D/s2]+1)^2x([h/s1]+1)x([w/s1]+1).

  - FlowNetS网络结构

        input 6xhxw
        Conv1 7x7/2  64   ReLU
        Conv2 5x5/2  128  ReLU
        Conv3 5x5/2  256  ReLU
        Conv3_1 3x3  256  ReLU
        Conv4 3x3/2  512  ReLU
        Conv4_1 3x3  512  ReLU
        Conv5 3x3/2  512  ReLU
        Conv5_1 3x3  512  ReLU
        Conv6 3x3/2  1024 ReLU
        ConvTrans5 4x4/2 512
        concat Conv5_1              ->ConvTrans1 4x4/2 2
        ConvTrans4 4x4/2 256  
        concat Conv4_1 ConvTrans1   ->ConvTrans2 4x4/2 2
        ConvTrans3 4x4/2 128  
        concat Conv3_1 ConvTrans2   ->ConvTrans3 4x4/2 2
        ConvTrans2 4x4/2 64
        concat Conv2 ConvTrans3
        Conv 3x3  2  ReLU
        bilinear

  - FlowNetC网络结构
    image1,image2两支单独输入,直到conv3.权重共享.
    correlation输出441通道,concat image1 conv3再conv 3x3 输出的32通道,合计473通道。之后与FlowNetS相同,上采样的conv2来自image1这一支.
    总参数数量与FlowNetS相同

    为了训练深度网络，只做了flying chairs数据集,该数据集试是将椅子(808张椅子,每张62个视角)放在背景图(共964张来自Flickr,切成4块)上，共生成了22872个pair
    数据增强:平移,旋转,缩放,高斯噪声,亮度对比度gamma扰动.
    FlowNetC比FlowNetS更容易拟合到训练集,且因为d的设置,在特大位移上性能略逊一筹.

    FlowNet2.0是FlowNet高度调优的升级版,包括3个方面:
    (1)训练时使用不同数据集,且性能与数据集的顺序相关：Flying chairs仅包括平面移动,Things3D是更实际的3D运动.训练时,schedule比flowNew更长,且首先在chairs数据集上训练,再在Things3D上finetune效果更好.即首先学习颜色匹配，再学习3d运动效果好，直接上来学困难的3d运动效果逊之.
    (2)多个网络级联,并且引入warping操作: 网络级联,底层网络先训练,再固定权重,提供给后级I2(x+u,y+v),|I2(x+u,y+v)-I1(x,y)|,uv(x,y)。因为级联,可减少通道数加快速度.可以得到一系列模型.
    (3)融合了一个专门针对小位移设计的网络:为应对小位移效果不好的问题,额外建立了一个数据集、原来chairs的光流是按照Sintel中直方图设计的,现在按照UCF-101的直方图构造,配合均匀背景,称为ChairsSDHom(small displacement homogeneous)。将FlowNetS骨架中7x7/2改成3x3 3x3/2 3x3 5x5改成3x3/2 3x3, conv6后添加conv6_1 3x3 1024  转置卷积后都添加3x3卷积. 称为FlowNetSD。融合网络输入两支光流大小,光流,误差,再加上Image1,共(1+2+1)*2+3=11输入通道。结构:

        Conv 3x3     64
        conv 3x3/2   64
        conv 3x3     128
        conv 3x3/2   128
        conv 3x3     128     -> Conv 3x3 2
        convtrans 4x4/2  32
        conv 3x3         32  -> Conv 3x3 2
        convtrans 4x4/2  16
        conv 3x3         16
        Conv 3x3         2

    FlowNetCSS+FlowNetSD融合称为FlowNet2

        Zhu Y., Lan Z., Newsam S., Hauptmann A,Hidden Two-stream Convolutional Networks for Action Recognition. ACCV 2018

  - MotionNet与flowNet最大差别在于其将计算光流按照基本概念建模成一个无监督学习问题
  损失函数包括三部分组成:
    - 像素重建误差:
    $$
    \Sigma_i^j \Sigma_j^w \rho(I_1(i,j)-I_2(i+u(i,j),j+v(i,j)))/hw
    $$
    - 平滑性惩罚:
    $$
    \rho(\nabla_x u)+\rho(\nabla_y u)+\rho(\nabla_x v)+\rho(\nabla_y v)
    $$ 
    - 重建结构误差:
    $$
    1-\Sigma \mu_1\mu_2\sigma_1\sigma_2/[(\mu_1^2+\mu_2^2)(\sigma_1^2+\sigma_2^2)]
    $$
    其中\mu \sigma 均是8x8像素块内的均值方差,下标1,2分别是前帧和后帧与光流重建的图像中按照stride=8切割的图像块.MotionNet整个网络结构与FlowNetSD相同,在升采样中每个尺度上产生的光流均计算损失.

## 相机模型与立体视觉

### 相机内在参数[固有参数]

- 几何模型:将小孔成像物理模型中成像平面放到小孔前面,成像大小相同但不再倒立.f'是焦距(物理尺寸)再乘上换算系数变成f(图像尺寸相关,像素/毫米)。进一步考虑成像各项不均匀和偏心,得到相机的几何模型:
$$
  x_{screen} = f_xX/Z+c_x \\
  y_{screen} = f_yY/Z+c_y
$$
- 内参矩阵: 应用齐次坐标后,便有投影变换
$$
  M = \left(
  \begin{array}{ccc} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{array}
  \right) \\
  Q = \left( X  Y  Z \right)^T \\
  q = MQ
$$
q是投影后像素坐标的齐次坐标形式,可约去z形成二维坐标.
- 畸变:使用透镜而不是小孔能更快汇集更多光线从而快速成像,但会引入畸变.
  - 径向畸变:光线弯曲引起,远离透镜中心的光线比靠近中心的光线弯曲更多.建模成关于r的泰勒展开,参数化为[k1,k2,k3]
$$  
  x_{correct} = x(1+k1r^2+k2r^4+k4r^6) \\
  y_{correct} = y(1+k1r^2+k2r^4+k4r^6) \\
$$
  - 切向畸变:安装过程引起,透镜平面与成像平面不平行.参数化为[p1,p2]
$$  
  x_{correct} = x+x(2p_1xy+p_2(r^2+2x^2)) \\
  y_{correct} = y+y(p_1(r^2+2y^2)+2p_2xy) \\
$$
  - 畸变作用在相机坐标上(投影前)

### 外参数 
- 实际物体位于世界坐标系,需要转换到相机坐标系中,一个坐标系统变换到另一个坐标系统,需要经历旋转(R)和平移(T).
- R可以分解为绕着空间三个轴(X,Y,Z)的旋转
$$
  R_x(\theta) = \left(
  \begin{array}{ccc} 1 & 0 & 0 \\ 0 & cos(\theta) & sin(\theta) \\ 0 & -sin(\theta) & cos(\theta) \end{array}
  \right) \\
  R_y(\theta) = \left(
  \begin{array}{ccc} cos(\theta) & 0 & -sin(\theta) \\ 0 & 1 & 1 \\ sin(\theta) & 0 & cos(\theta) \end{array}
  \right) \\
  R_z(\theta) = \left(
  \begin{array}{ccc} cos(\theta) & sin(\theta) & 0 \\ -sin(\theta) & cos(\theta) & 0 \\ 0 & 0 & 1 \end{array}
  \right) \\
  R = R_x(\theta_1)R_y(\theta_2)R_z(\theta_3)
$$
- 用Rodrigues变换表示旋转:向量的模表示旋转的弧度,方向表示旋转轴.向量表示和矩阵表示的旋转采用Rodrigues变换互相转化.

### 单目相机标定

- 理论分析:给定棋盘格,其中有N个角点,K个视图.则构成2NK个约束[每个点坐标有2个分量],共有参数4+6k(忽略畸变,4个内参,每个视图6个外参),参数小于约束的情况下,可得(N-3)K>=2.实际上一个单应性矩阵只需要4个点就能确定，因此无论角点个数N多少,有效N都不超过4.因此 K>=2,即至少有2个视图才能标定.实际使用时,一般使用7x7棋盘格10多个视图.
- 把棋盘格放在z=0的世界坐标系中,则
$$
\left(\begin{array}{ccc} x \\ y \\ z \end{array}\right)  = M\left(\begin{array}{ccc}r1 & r2 & r3 & t \end{array}\right) \left(\begin{array}{ccc} X \\ Y \\ 0 \\ 1 \end{array}\right) = M\left(\begin{array}{ccc}r1 & r2 & t\end{array}\right)\left(\begin{array}{ccc}X \\ Y \\ 1 \end{array}\right)  
$$
于是M[r1,r2,t]对应于单应性矩阵H = [h1,h2,h3]
$$
r1 = \lambda M^{-1} h1\\
r2 = \lambda M^{-1} h2\\
t = \lambda M^{-1} h3\\
$$
R为正交矩阵,因此
$$
r1r2^T = 0 \rightarrow h2^TM^{-T}M^{-1}h1 = 0 \\
|r1|= |r2| \rightarrow h2^TM^{-T}M^{-1}h2 = h1^TM^{-T}M^{-1}h1 
$$
令B = M^{-T}M^{-1},可得到B解析解的形式.将前述二次型方程展开
$$
\left(\begin{array}{ccc} h_{i1}h{j1} & h_{i1}h{j2}+h_{i2}h{j1} &  h_{i1}h_{j3}+h_{i3}h_{j1} &  h_{i2}h_{j2} &  h_{i2}h_{j3}+h_{i3}h_{j2} &  h_{i3}h_{j4}  \end{array} \right) \left(\begin{array}{ccc} B_{11} &  B_{12} &  B_{13} &  B_{22} & B_{32} & B_{33}  \end{array} \right) ^T  \\
\rightarrow \left(\begin{array}{ccc} V_{12}^T \\ V_{11}^T-V_{22}^T\end{array} \right) b = 0
$$
这是一个视图方程,有k个视图可得k个这样的方程.从而得到B,进而根据B与M之间的解析关系得到内参矩阵.
\lambda由|r1| = 1得到
$$
\lambda = 1/|M^{-1}h1| \\ r3 = r1\times r2
$$
此时还忽略了畸变,但求了估计的内参和外参,在外餐转到相机坐标后为无畸变坐标,再联立畸变方程估计畸变参数.固有参数、外参和畸变参数采用反复迭代的方式求解.
- opencv api
  - 标定
  `double cv::calibrateCamera(InputArrayOfArrays objectPoints,InputArrayOfArrays imagePoints,Size imageSize,InputOutputArraycameraMatrix, InputOutputArray distCoeffs,OutputArrayOfArrays rvecs,OutputArrayOfArrays tvecs, int flags = 0,TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, DBL_EPSILON))`
  - 计算世界坐标系坐标的旋转和平移矩阵[标定的子过程]
  `int cv::solveP3P(InputArray objectPoints,InputArray imagePoints,InputArray cameraMatrix,InputArray distCoeffs,OutputArrayOfArrays rvecs,OutputArrayOfArrays tvecs,int flags)`
  - 稠密去畸变重映射
  `void cv::initUndistortRectifyMap(InputArray cameraMatrix,InputArray distCoeffs,InputArray R,InputArray newCameraMatrix,Size size,int m1type,OutputArray map1,OutputArray map2)`+`void cv::remap(InputArray src,OutputArray dst,InputArray map1,InputArray map2,int interpolation,int borderMode = BORDER_CONSTANT,const Scalar& borderValue=Scalar())`  R是相机坐标的全局补偿,newCameraMatrix是相机坐标至像素坐标,可来自stereoRectify P1/P2
  `void cv::undistort(InputArray src,OutputArray dst,InputArray cameraMatrix,InputArray distCoeffs,InputArray newCameraMatrix =noArray())` 
  - 稀疏去畸变重映射
  `void cv::undistortPoints(InputArray src,OutputArray dst,InputArray cameraMatrix,InputArray distCoeffs,InputArray R = noArray(),InputArray P=noArray())` R是相机坐标的全局补偿,P是相机坐标至像素坐标,可来自stereoRectify
  - 世界坐标系坐标投影至像素坐标
  `void cv::projectPoints(InputArray objectPoints,InputArray rvec,InputArray tvec,InputArray cameraMatrix,InputArray distCoeffs,OutputArray imagePoints,OutputArray jacobian=noArray(),double aspectRatio=0)` 

### 双目相机立体视觉
- 双目相机深度测量的步骤
(1)使用数学方法消去畸变 
(2)调整相机之间的角度和距离,使两图像平面共面.这一过程称为Rectification
(3)左右相机中找到相同的特征,输出视差图,即相同特征在左右相机中的坐标差,这一过程称为匹配.
(4)将视差转化为深度图,称为重投影. Z = fT/d
- 对极几何
 - 本征矩阵(essential matrix),描述两坐标中同一物理坐标之间关系.左(右)相机物理坐标点为p_l(p_r),左右坐标系之间的变换为R,T.则
$$
T\times p_l = S \cdot p_l \\  S = \left(\begin{array}{ccc} 0 &-T_z & T_y \\ T_z & 0 & -T_x \\ -T_y & T_x & 0 \end{array} \right),rank(S) = 2 \\
(p_l-T)^T(T\times p_l) = 0 ,p_r = R(p_l-T) 
\rightarrow (R^{-1}p_r)^TSp_l = 0 \rightarrow p_r^{T}RSp_l = 0 \rightarrow p_r^{T}Ep_l = 0  (E\equiv RS)
$$
  - 基本矩阵(fundamental matrix),描述左右相机像素坐标之间的关系.有本征矩阵出发,投影到像素坐标,则
$$
  (M^{-1}q_r)^TEM^{-1}q_l = 0 \rightarrow q_r^TM^{-T}EM^{-1}p_l = 0 \rightarrow q_r^TFp_l = 0 (F\equiv M^{-T}EM^{-1})
$$
  - 本征矩阵和基本矩阵的直观理解:E,F的秩均为2,因此给定E,F和其中一个相机中坐标,其二次型展开后解空间为一条直线,表示单目相机某一成像点在另一台相机中的可能位置是一条直线.即真实相点和投影中心在另一项相机中投影的连线——级线(epiline) 
  `void cv::computeCorrespondEpilines(InputArray 	points,int whichImage,InputArray F,OutputArray lines)`
- 计算基本矩阵
  - 根据图像匹配直接计算
  F为3x3矩阵,但因其秩为2,因此自由度只有8.本征矩阵和基本矩阵关于两相机的坐标约束均为二次型,其中坐标已知时可转换为线性方程,因此至少需要8个匹配像素坐标. `Mat cv::findEssentialMat(InputArray points1,InputArray points2,InputArray cameraMatrix,int method = RANSAC,double prob=0.999,double threshold = 1.0,OutputArray mask = noArray())`
`Mat cv::findFundamentalMat(InputArray points1,InputArray points2,OutputArray mask,int method = FM_RANSAC,double ransacReprojThreshold = 3.,double confidence=0.99)`
  - 根据标定信息计算 
  根据世界坐标系中坐标分别投影至像素坐标后表示同一物体的约束:
$$
q_r = R_rq+T_r  , q_l = R_lq+T_l  , q_l = R(q_r-T)\\
\rightarrow R = R_r^TR_l , T =T_r-R^TT_l 
$$
- 双目矫正Rectification
在计算得到两相机坐标系变换的R,T后,需要利用其将两个成像平面变换至共面.首先将R平分一半,此时两相机平行.再次构造R_rect,沿着两投影中心连线方向旋转,使得两成像平面共面 
$$
R_l = R_r^T = R^{1/2} \\
e_1 = T/|T|, e_2 = (-e_{1y},e_{1x},0), e_3 = e1 \times e2  R_rect = \left(\begin{array}{ccc}e_1 & e_2 & e_3 \end{array} \right) 
$$
将图片根据所得旋转矩阵(R1 = R_lR_rect, R2 = R_rR_rect)矫正后,得到对齐的图像,通过匹配得到视差图,然后重投影(X,Y,Z = X,Y,ft/d,写成矩阵乘法形式W=Q[4x4]P)得到深度图,[stereoRectify中P1,P2是3D坐标转到像素坐标投影矩阵,但我不知道跟相机内参除了平移外有何差异,应该是转动后改了]
- opencv中api
  - 双目标定[单目标定+两相机之间的关系]
  `double cv::stereoCalibrate(InputArrayOfArrays objectPoints,InputArrayOfArrays imagePoints1,InputArrayOfArrays imagePoints2,InputOutputArray 	cameraMatrix1,InputOutputArray distCoeffs1,InputOutputArray cameraMatrix2,InputOutputArray 	distCoeffs2,Size 	imageSize,InputOutputArray 	R,InputOutputArray T,OutputArray E,OutputArray F,int flags = CALIB_FIX_INTRINSIC,TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 1e-6))`
  - 双目矫正
  `void cv::stereoRectify(InputArray cameraMatrix1,InputArray distCoeffs1,InputArray cameraMatrix2,InputArray distCoeffs2,Size imageSize,InputArray R,InputArray T,OutputArray R1,OutputArray R2,OutputArray P1,OutputArray P2,OutputArray Q,int flags =CALIB_ZERO_DISPARITY,double alpha=-1,Size newImageSize=Size(),Rect* validPixROI1=0,Rect* validPixROI2=0)`
  - 视差图转成深度图
  `void cv::reprojectImageTo3D(InputArray disparity,OutputArray 3dImage,InputArray Q,bool handleMissingValues=false,int ddepth = -1)`


参考资料:  
  [1]Adrian Kaehler,Gary Bradski."学习OpenCV3". 2018  
  [2]RafaelC.Gonzalez,Richard E.Woods."数字图像处理(第三版)",2011  
  [3]https://docs.opencv.org/master/  
  [4]http://effbot.org/imagingbook/pil-index.htm  



