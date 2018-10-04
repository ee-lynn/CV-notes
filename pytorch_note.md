# Pytorch
　&nbsp;　　　　　　　　　　　　　　　　　　　　　　　Author : sqlu@zju.edu.cn
## 目录
我将pytorch整个包归类为以下8类，分别为 

** 基本数据与动态图 **
-  torch.Tensor
在0.4.0之前的版本具有求导功能的数据还有一层torch.autograd.Variable的封装，现在已经弃用，可以直接为Tensor求导，但Variable仍然可用
在相关的类有nn.Parameter,torch.autograd.Function 

** 网络单元 **
- torch.nn.Module 使用时实现forward即可
及其子类　nn.Sequential,nn.ModuleList

** 函数辅助 **
- torch.nn.functional，是nn.Module的重要补充

** 求解器 **
- torch.optim

** 初始化 **
- torch.nn.init

** 数据准备 **
- torch.util.data

** 数据处理 **
- torchvision

** 杂项 **
- 加速计算
- 模型的保存

##  基本数据与动态图
-  **基本特性**
与numpy.ndarray类似, 大部分操作都相同，可以互相转换,且共享内存 
       numpy<-->from_numpy
 一些操作具有两个版本func_/func 　func_是func的inplace版本.一般情况下，少使用inplace操作，因为结果对求导可能会有用.
一些易混淆操作:
      view/reshape                  #数据不变，改变形状/大小变大时，再次分配内存，变小时，原数据仍不丢弃
      expand/repeat                #只是广播操作/数据拷贝　
      unsqueenze/squeenze 　#产生/压缩指定维上为１的维度
*notes* : numpy广播原则:让所有输入与最长的shape看齐，shape不足的前方补１(升维), 在对应维数地方要么长度一致，要么长度为１复制扩展成一致，否则不能计算
- 内存机制
 对tensor进行操作时，其storage其实一直没变,变化的只是对storage的解释,即封装的头部.求转置时,会造成内存不连续,用contiguous后将重新拷贝内存数据
  `a = torch.ones(2,5)`
  `b = a[:,1]`
	 `id(a.storage()) == id(b.stroage())   #yeilds True`	 
- 自动求导的机制 
tensor(Variable)基本属性:data， grad， grad_fn，requires_grad, volatile
data存储数据，grad存储导数,grad_fn大约是反向传播函数对象，包括了next_function属性，是此grad_fn的前驱，由此记录了反向传播的计算图，实现了动态求导
requires_grad, volatile均是求导属性.默认情况tensor是不可求导的，volatile优先级比requires_grad高,
volatile为True时，计算图中其所有依赖变量都不求导,
requires_grad为True时,计算图中所有依赖变量都求导,但只有叶子节点会保留grad，其余在计算结束后都会被清空
求导是累积的，因此多次求导时需要zero_grad来清空grad
- backward参数解析
     Variable.backward(grad_variables = None,retian_graph = True,Create_graph = None)
     grad_vatriables:从后面传过来的梯度,与Variable形状相同
     retain_graph:　前向传播时缓存在反向传播后将被清空，若要多次反向传播，需要保存前向图
     Creat_Graph:  为反向传播建立计算图，可以求高阶导数
- 扩展autograd
当builtin自动求导无法满足要求时(基本上当基于原子操作的反向传播效率低下时使用，因此该高级用法不是必须掌握的)
需要定义torch.autograd.Function的子类，实现静态方法forward,backward,调用该类的apply方法(此时与caffe有些相似)
## 网络单元
- 所有网络都是torch.nn.Module子类，使用时实现forward即可。 __call__()基本上就调用了forward
- 已将所有可学习参数封装起来，通过parameters/named_parameters访问 
- 网络单元是嵌套的,childen/named_children 访问模块下一层，modules/named_modules 递归访问所有子模块，包括自身
- nn.Module实现中有三个内置字典: \_\_setattr\_\_ ,　\_\_getattr\_\_ 均已对类型最初判断,否则存储在\_\_dict\_\_中
      _parameters = OrderedDict() ＃手动生成的Parameter对象都会放在其中
      _modules = OrderedDict()　　＃nn.Module子类都会放在其中
      _buffers = OrderedDict()　　＃计算中临时变量,例如BN层的running_mean，Adam的momentum等
      training = True
由此说明将nn.Module构成的list做为属性将不起作用，而应该用ModuleList
- Sequential
可用模块名 add_module(name,module) 或者id号做索引 Sequential((module1.module2..))
## 函数辅助
基本上所有nn.Module都有其functional实现,但有参数时需要预先手动生成Parameter对象，因此该类更适用于没有参数的模块
## 求解器
包含了各种优化器，传入网络参数和学习率lr.
e.g. SGD(iterable({'param': Parameters,'lr':float},))
每一次迭代中
      Solver.zero_grad()
      Module.backward()
      Solver.step()
## 初始化
集成了一些方法对Parmeter的data进行赋值
## 数据准备
继承torch.util.data.Dataset, 需要覆写\_\_len\_\_,\_\_getitem\_\_ 以支持迭代协议,一次返回一个样本(及标签)　  
然后用torch.util.data的DataLoader将batch_size个样本拼接成一个batch, 支持迭代协议
## 数据处理
- 数据预处理：Transforms
Compose是一个pipline，可内嵌一系列处理操作
包括但不限于Resize,CenterCrop,RandomCrop,Pad,Normilize,ToTensor(以255归一化),ToPILImage,以及自实现的任意预处理
- 保存图片
make_grid,save_img
## 杂项
- 加速计算
　1) Tensor.cuda()，返回新对象
　2) Net.cuda()，原位操作,也返回自身
 以上两者本质都是将Tensor移到GPU上
多GPU计算:
torch.nn.DataPrallel(Module,inputs,device_ids = None,outputs = None,dim = 0)　
- 模型的保存
torch.save(obj,filename),torch.load(filename)可以保存任意序列化对象，特别是tensor
model有state_dict，其实就是named_parameters，将其保存，加载时用model.load_state_dict(state_dict)