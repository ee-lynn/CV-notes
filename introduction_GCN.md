# Introduction to Graph Convolution Networks

&nbsp;     　　　　　　　　　　   sqlu@zju.edu.cn

## 引言

卷积神经网络(CNN)基于时域或空域信号处理中的卷积操作，使用卷积核的参数进行学习(实际上CNN的卷积核出于方便的考虑被定义成了理论上卷积核的中心对称)。它是连续函数卷积操作的离散化，对栅格离散化的序列或图像具有天然的适应性。但对于非线性的数据结构,如树,图这样的数据结构，如何有效利用其拓扑信息需要将CNN进一步拓展，图卷积神经网络便是其中一种模型。对于N个特征为 
$$ h \in \mathbb{R}^d $$ 
的节点组成的图G，图卷积操作统一表述为
$$ F(G_l,W_l)=Update(Aggregate(G_l,W^{agg}),W^{update}) $$
Aggregation函数用于聚集邻居节点的特征,update函数用于更新节点的特征，
$$ W^{agg}, W^{update} \qquad 分别是两个函数中可学习参数。$$
该操作可以简单理解成常规卷积的拓展。在常规卷积中，G是HxW网格,每个格点具有C维的特征, Aggregate函数是以W^{agg}为权重的线性函数, Update是平均函数。但要深刻理解其理论基础，需要有更多的背景知识。本文将以图信号处理的角度阐释图上的傅里叶变换，卷积定理，CNN常规部件在GCN中的对应操作以及GCN在计算机视觉中的典型应用.

## 拉普拉斯算子与拉普拉斯矩阵的关系

首先定义一些术语。对于图G=(V,E), V是节点集合，E是边集合，D是顶点的度矩阵（对角阵），对角线上依次是每个节点的度，A是图的邻接矩阵，定义Combinatorial laplacian L=D-A，下面说明Combinatorial laplacian与拉普拉斯算子之间的关系。
拉普拉斯算子定义为连续函数梯度的散度
$$ \triangle f = \nabla \cdot \nabla f $$
二维情况下离散化后
$$ \frac{\partial^2f}{\partial x^2} = \frac{\partial}{\partial x} f(x+1,y)-\frac{\partial}{\partial x} f(x,y) = f(x+1,y)-f(x,y)-f(x,y)+f(x-1,y)=f(x+1,y)+f(x-1,y)-2f(x,y)$$
$$ \frac{\partial^2f}{\partial y^2} = \frac{\partial}{\partial y} f(x,y+1)-\frac{\partial}{\partial y} f(x,y) = f(x,y+1)-f(x,y)-f(x,y)+f(x,y-1)=f(x,y+1)+f(x,y-1)-2f(x,y)$$
即
$$ \triangle f(x,y) = f(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1)-4f(x,y) $$
若将离散化情况下二维平面看成一张图，图是四联通的，那么
$$ \triangle f = -Lf $$
推广f定义在图域上，图的每条边都是独立的方向, 拉普拉斯算子作用域f等价于图的拉普拉斯矩阵作用于f。

## 图上的傅里叶变换

由连续函数的傅里叶变换定义，很容易验证
$$ \mathcal{F}(\triangle(f)) = -\omega^2\mathcal{F}(f) \qquad \omega是频率,\mathcal{F}是fourier变换 $$ 
即傅里叶变换在拉普拉斯算子的作用下呈现线性拉伸关系，这种关系很容易由实对称矩阵L得到，因为实对称矩阵L，存在单位正交矩阵P，使得
$$ P^TL=\Lambda P^T \qquad 其中\Lambda 为对角阵（实对称矩阵的对角化）$$
即
$$ P^TLf=\Lambda P^Tf \qquad $$
在P^T变换下,拉普拉斯矩阵的作用呈现线性拉伸的关系，因此可定义P^T为L表示的图上的傅里叶变换,P相应地表示傅里叶逆变换，对角阵\Lambda对角线上的值为图上的特征频率

## 图卷积的频域和空域表达

由图的傅里叶变换和卷积定理，图上的卷积便为
$$ f*g = P(P^Tf \odot P^Tg) \qquad  \odot为elewise prod(Hadamard product). $$
- 当参数化卷积核时，第一代GCN[]直接将卷积核的傅里叶变化参数化为diag(\theta).即
$$ f*g = Pdiag(\theta)P^Tf $$
此时模型的卷积核与图大小相同，具有N个参数，需要在初始化阶段对L进行对角化。
为克服这些问题，第二代GCN将卷积核参数化为
$$ \Sigma_{k=0}^K \theta_kT_k(\Lambda) \qquad 这里T_k(\Lambda)为频率矩阵不超过k次的正交多项式$$
- 第二代GCN采用截断的多项式来表示卷积核的频率，这里的技巧是采用特征频率展开，可以验证
$$ T_k(\Lambda) = P^TT_k(L)P \rightarrow g=\Sigma_{k=0}^K\theta_kT_k(\Lambda) = P^T\Sigma_{k=0}^K\theta_kT_k(L)P $$
$$ Pdiag(\theta)P^Tf = PP^T\Sigma_{k=0}^K\theta_kT_k(L)PP^Tf=\Sigma_{k=0}^K\theta_kT_k(L)f $$
L是稀疏的且L^K只会导致相隔K个点以内的元素非0，即节点的状态只会受相隔K个点以内的元素影响。这样第二代GCN避免了对L的正交分解，且将参数降低到了K个。
- 第三代GCN将K限制为1，采用线性来表示卷积核，则图卷积将表示为\theta_0f+\theta_1Lf,并进一步采用\theta_0=\theta_1正则化模型，\theta(I+L)f.当f为多实值函数(向量)时\theta也由实值拓宽成向量，对所有的节点，参数是共享的。拉普拉斯矩阵在使用时，一般使用归一化后的形式，即
  
Normalized laplacian
$$  Ls=D^{-\frac{1}{2}}LD^{-\frac{1}{2}} $$
Random walk Normalized laplacian
$$  Lr=D^{-1}L $$ 
对于第三代GCN,对I+L归一化可以进一步提高性能

## 深度图卷积神经网络

堆叠多层图卷积操作便构成了深度图卷积神经网络模型。Aggregate可采用ave pooling,max pooling,LSTM等对输入变量个数没有要求的操作,有时候还要求对输入顺序无要求。Upadate一般都采用FC来实现。模型深度加大后，GCN也会面临CNN加大深度后面临的问题，可将CNN中的部件迁移至GCN:
- Residual Learning for GCNs
$$G_{l+1} =F(G_l,W_l)+ G_l$$
- Dense Connections in GCNs
$$G_{l+1}=concat(F(G_l,W_l), G_l)$$
- Dilated Convolution
在Aggregate时需根据中心节点与邻居节点之间距离排序，然后每相隔dilated rate个节点采样进行聚合

## 图卷积在CV中的应用

CV应用中，一般没有显而易见的图结构，需要根据数据形态定义图。不过在这种情况下图的定义要比使用人为规则将不规则数据映射成线性数据结构效果要好很多，图结构能够最大程度保留数据的几何和统计性质。
### 点云语义分割

在很多立体视觉任务中物体由点云表示，点云作为一种无序，非均匀的数据形式在无人为干预的投影规则下难以采用CNN直接提取特征，可以考虑采用GCN处理。以[5]为例
- 图建立:直接在采用点云组成的欧式空间中的范数，当距离范数小于一定值时认为相连(ball quering)。
- 输入信息: 点云坐标和点云的特征(颜色或雷达回波强度)
- 图卷积核: 相连的节点共享权重,并且采用maxpooling聚合。后续还有一个节点pooling 操作，采用使用最远点采样(fathest point sampling,递归地从点集中选取离已选点集最远的点)采样得到一系列中心点.它比随机采样更能覆盖整个点云)。实现上先pooling进行sampling再提取特征.
- 模型:为克服点云密度不均衡的特点，在模型中注入了多尺度特性。在训练时以不同概率随机drop掉单元内的点模拟不均匀的点云训练集，并且在ball quering是距离阈值采用多个尺度, 所有结果concat后作为更新特征，不断降采样后接softmax,该模型可用于点云分类。
在点云语义分割需要输出节点级别的预测，还需要上采样。采用K(K=3)近邻点的特征根据距离倒数加权平均得到上采样点的特征并与.
浅层对应节点特征concat后经过共享的FC更新点特征(模仿1x1卷积),最后每个点节接softmax.

### 基于骨架的行为识别

传统的行为识别一般直接采用视频帧进行识别，当使用光流作为补充模态进行ensemble时，可以进一步提高性能，人体骨架也可作为一种额外的模态，但单独依靠骨架进行行为识别性能还远不如直接采用视频帧。以[6]为例

- 图建立:同一帧中人体关键点相连，时域上同名关键点相连，构成了一张时空同。
- 输入信息:关键点的坐标和关键点置信度
- 图卷积核:空域中直接相邻的关键点作为图卷积核的感受野，空域上权重共享方案在ablation study中显示 全部节点共享<中心节点+邻居节点两套权值<自身+离重心比中心节点更近的节点+离重心比中心节点更远的节点三套权值。时域上前后a/2帧，不同时间空域中的参数不共享，直接相加融合
$$ \Sigma_{i=-a/2}^{a/2}D^{-\frac{1}{2}}LD^{-\frac{1}{2}}W_{iv}f, \qquad W_{iv}为i帧空域中权值(FC) $$
- 模型: 使用resGCN形式堆7层，最后global pooling后FC+softmax

## 应用中问题的讨论

- 权值如何共享对性能产生影响较大
[3]在ablation study中说明Aggregate对所有直接相连的邻居节点都采用共享的权重效果最佳，这种聚合形式在CNN中等价于Conv3x3+maxpooling且限制Conv3x3卷积核9个数字完全相同,[6]在设计Aggregate时就考虑将近邻节点分组，每组中共享权值。在ablation study中说明了较复杂的分组方案性能更好。
- 在Aggregate中输入变量形式繁多
Update操作一般都是将聚合特征(有的再与中心节点特征concat)送入FC，形式比较统一.相比之下aggregate操作送入pooling的输入形式多种多样:
  - EdgeConv[7]:  mlp(h_u-h_v), $$v \in N(u)$$
  - GraphSAGE[8]：mlp(h_v), $$v \in N(u)$$
  - Graph Isomorphism Network[9]: sum(h_v) , $$v \in N(u)$$
  - Max-Relative GCN[4] maxpooling(h_u-h_v) $$v \in N(u)$$
N(u)表示u的邻居节点。这种形式一方面可以将多种实现方式设计成单元部件，依靠NAS技术在数据集上搜索最佳的GCN结构，另一方面由于多种形式的操作在不同数据集上性能是各异的,对在大型数据集上搜索得到的结构泛化能力还需要考验。
- 建图时的人为干预
一般而言非线性数据结构在建图时节点的建立是非常方便的,例如立体视觉中每个点云，行为识别中每个关键点，社交网络中每个用户,但在建立边时需要一定的技巧，需要充分考虑数据本质的形态和特点。
以点云处理为例，可以将点云所处的3维空间中K近邻个节点相连,可以将在距离阈值范围以内的点相连，甚至可以将3维空间改成节点特征的高维空间中的K近邻或者距离阈值范围以内，计算量和性能都有所不同。
- 关于在deep learning框架中实现GCN的问题
一般而言，GCN比较特殊的操作为选出需与中心节点聚合的邻居节点，将这些邻居节点(经过共享权重的变换)聚合。都可以复用deep learning框架中为CNN设计的操作。
将图以D_{NxC}形式保存每个节点，并保存邻接矩阵W_{NxN},N为节点数,C是每个节点的特征维数，对于动态变化的图，实时更新邻接矩阵。
前向时,使用W index出D中对应Ni行，共享权重可以使用FC(linear)或者conv1d。使用FC时tensor以N_ixC形式安排，使用conv1d时tensor以CxN_i形式安排。聚合操作复用maxpooling1D即可。

**Reference**
- [1] Bruna, J. , Zaremba, W. , Szlam, A. , & Lecun, Y.  Spectral networks and locally connected networks on graphs. ICLR 2014
- [2] Defferrard, Michaël, Bresson, X. , & Vandergheynst, P. Convolutional neural networks on graphs with fast localized spectral filtering.  NeuIPS 2016
- [3] Thomas N. Kipf, Max Welling. Semi-Supervised Classification with Graph Convolutional Networks. ICLR 2017
- [4] Guohao Li, Matthias Müller, Ali Thabet, Bernard Ghanem. DeepGCNs: Can GCNs Go as Deep as CNNs? ICCV 2019
- [5] Charles R. Qi, Li Yi, Hao Su, Leonidas J. Guibas. PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. NeuIPS 2017
- [6] Sijie Yan, Yuanjun Xiong, Dahua Lin. Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition. AAAI 2018
- [7] Y. Wang, Y. Sun, Z. Liu, S. E. Sarma, M. M. Bronstein, and J. M. Solomon. Dynamic graph cnn for learning on point clouds. arXiv:1801.07829v1
- [8] W. Hamilton, Z. Ying, and J. Leskovec. Inductive representation learning on large graphs. NeurIPS, 2017
- [9] K.Xu, W.Hu, J.Leskovec, andS.Jegelka. How powerful are graph neural networks? ICLR-2019
