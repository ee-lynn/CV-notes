#Pytorch
　&nbsp;　　　　　　　　　　　　　　　　　　　　　　　Author : sqlu@zju.edu.cn
## key class
- torch.Tensor
- autograd.Variable
- torch.nn

- Variable equvalent
` class Variable:` 
 `public:`
 　`Variable(Tensor data None, requires_grad = Flase,volatile = False)`
 　`:data(data),diff(),grad_fun(NULL){}`
 　`Tensor data;`
 　`Variable diff;`
 　`func* grad_fun; `
 - Tensor-> numpy   Tensor.numpy()
 - Tensor ->cuda    Tensor.cuda()
 - numpy ->Tensor   from_numy()
 - func_ 　for Tensor is a func version of `inplace = True`
 torch.Tensor()　<->np.array()　该有的都有，方法也一样,reshape ->view(),resize()
 numpy广播原则:让所有输入与最长的shape看齐，shape不足的前方补１(升维)
 在对应维数地方要么长度一致，要么长度为１复制扩展成一致，否则不能计算
 在torch中手动实现这个过程为unsqueenze()和expand()
 
 -  details of Variable
 　属性具有遗产性,且volatile优先级比requires_grad高
 　only Variable have gradiant
 　requeres_grad:false in default, gradients are not computed　
 　accumated gradients in runtime. so clear them if needed: a.zero_grad()
 Variable.backward(grad_variables = None,retian_graph = True,Create_graph = None)
 grad_vatriables:从后面传过来的梯度,与Variable形状相同
 retain_graph:　缓存
 Creat_Graph为求导建立图模型，可以求高阶导数
 将计算图看成一个有向无环图(RNN则是展开后)那么次有向无环图对应一棵树，计算图的建立过程是叶子节点到根节点(loss)的过程，只有叶子节点会保留计算梯度grad,其他变量计算后都将grad成员置为None
 grad_fn是函数指针，指向得到该变量的函数，它有next_function成员，指向计算图中孩子节点,save_variables保存缓存
 
 
- torch.nn.Module is super class of all userdefined model class
　def forward() and __init__() (**there are learnable parameters**) in the user defined class
parameters(),named_paramters():{name,parameter}返回迭代器
所有可学习的参数都是Parameter类(Variable子类,但其requires_grad一定为True)
无须实现backward已经由autograd根据froward自动求导，这也要求forward中变量必须是Variable
调用时直接model(input)即可，这样会调用model类的__call__()，会自动调用self.forward(),且做一些别的细微操作
nn.Module实现细节:
self._paramters = OderedDict()
self._modules = OrderedDict()
self._bnuffers = OrderedDict()
self.training = True
def setattr(self,obj):
'''
self.obj[0] = obj[1]
'''
if isinstance(obj[1],Parameter):
    _parameter[obj[0]] = obj[1]
elif isinstance(obj[1],Module):
_Modules[obj[0]] = obj[1]
elif dropout..batchnorm...
   _buffer...#需要保留的参数
  else:
  __dict__[obj[0]] = obj[1]
 def __getattr__:
 #从__dict__中取
 .train(),.eval()＃递归设置所有模型training = False
 .parameters(),named_parameters() 递归检索出所有参数
 .modules(),named_modules()　递归检索出所有模型
 Sequential,ModuleList是nn.Module子类,后者相当于vector\<Module\>
 def getattr(): Module子类
Sequential可以在构造时加层,也可以add_module(string name,Module module)，实现了级联的forward方法


- torch.optim.Optimizer<-各种优化器的基类　e.g.
SGD(iterable(dic{'param': Parameters,'lr':float}))
还可以不同参数指定不同的学习率
optim.SGD([{'param':model.base.prameters()},{'param':model.classifier.parameters(),'lr':1e-3}],lr = 1e-2,momentum = 0.9)

nn.fucntional　里面有Module对应的各种函数,这些只是纯函数，不会被提取可学习参数，需要人为初始化Parameter，因此一般将无参数的层用该模块实现，可以不放在模型的构造函数里
nn.autograd.Function 用于扩展autograd，自定义的函数func需要继承Function,不需要构造函数，但是要实现forward()和backward()，并且以@staticmethod 装饰
此时output.backward()会调用self.backward()分别计算input的梯度

- 加速计算
1) 将输入数据.cuda()，返回新对象
2) 讲模型.cuda()，原位操作 
以上两者本质都是将Variable内部的Tensor移到GPU上
多GPU计算:
torch.nn.DataPrallel(Module,inputs,device_ids = None,outputs = None,dim = 0)　
- 数据集制作
数据集继承torch.util.data.Dataset, 需要覆写__len__, __getitem__返回长度和索引的数据和标签　
然后用torch.util.data的DataLoader生成batch的迭代器，长度为floor(epoch/batch)(drop_last = True)
- 模型的保存
torch.save(obj,filename),torch.load(filename)可以保存任意序列化对象
model有state_dict属性，其实就是named_parameters，然后可以将其保存，加载时用model.load_state_dict(state_dict)
*notes*: modules()与children()的区别是modules递归了所有children,children只是当前层孩子节点
- 一些工具包:
torchvision.transforms Resize,CenterCrop,RandomCrop,Pad,Normilize ToTensor(会以255归一化)　<-> ToPILImage,  Compose():一个pipline
