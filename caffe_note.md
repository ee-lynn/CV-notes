# CAFFE(convolutional architecture for fast feature embeding )
&nbsp;     　　　　　　　　  　　　  sqlu@zju.edu.cn

## 网络可视化
[upload prototxt to this link](http://dgschwend.github.io/nescope/#/editor)  or [this link](http://ethereon.github.io/netscope/#/editor)
## caffe 内核代码
需了解Blob,Layer,Net,Solver 源码;
Net: a set of layers
name:"dummy-net"
layer {name: "data"...}
Layer: name, type, bottom, top, parameters(refer to $CAFFE_ROOT/src/proto/caffe.proto)
## how to develop new layers in CAFFE
consisted of 
- Setup:
for one-time initialization: read parameters, fixed-size allocations,etc.
- Reshape:
any other work that depends on the shapes of bottom blobs
- Forward_{cpu, gpu} 
- Backward_{cpu,gpu}
[refer to this link](http://github.com/BVLC/caffe/wiki/Development) 
- python layer
refer to  $CAFEE_ROOT/examples/pycaffe/layers/{pyloss,pascal_multilabel_datalayers}.py
class name is the python layer name
all four functions above are to be realized in this class

## pycaffe
- net 对象构建及使用
      net = caffe.Net(model_def,　  # defines the structure of the model: prototxt　　　 　
	                         model_weights, #contains the trained weights:caffemodel
	  					   caffe.TRAIN) #phase
      net.copy_from(model_weights)
      net.forward()  # 返回各个输出层的dict(name,blob), 也可以通过net.blobs来取
      net.save()  # 保存caffemodel

- data structure
`net.blobs` : OrderedDict, (NxCxHxW) 以name为键,blob为值
`net.top_names, net.bottom_name` 以layer name为键,top,bottom blob name的OrderedDict
`net.params`: OrderedDict, 以layer_name为键,blob为值
 [0] for weight blobs(output_chanel, input_chanel,filter_height,filter_width)  跟blob等价
[1]for bias blob
`net._layer_names`
`net.layers[id]` 依附于layer对象的参数blob，跟net.params相同

- caffe.NetSpec()获得net, 只需不断填充，返回的都是Top blobs,属性管理均由n.\_\_setattr\_\_("name",blob)捕获
caffe.layers提供与proto的交互, layers.type(bottomblob, *attr) , 最后.to_prototxt()输出序列化字符串，写入文件即可
例如:
      n = caffe.SpecNet()　 
      n,conv1 = layers.Convolutions(n.data,kernal_size = 5,num_output = 20,weight_filler = dict(type = "xavier"))
      n.pooling1 = layers.Pooling(n.conv1,kernel_size = 2,stride = 2,pool = caffe.parameters.Pooling.MAX)
      n.fc1 = layers.InnerProduct(n.pool2,num_output=500,,weight_filler = dict(type = "xavier"))
      n.relu1 = layers.ReLU(n.fc1,in_place = True)
      layers.SoftmaxWithLoss(n.score,n.label)
- solver对象构建
`from caffe.proto import caffe_pb2`
`s = caffe_pb2.SolverParamer()` #solver.prototxt里面的key都可以作为成员属性，最后f.write(str(s))
对偶地，proto = caffe_pb2.NetParameter()
对prototxt中内容进行修改,如proto.layer 是各layer的list,可对prototxt中成员进行访问 ,可以直接从caffenmodel二进制文件中解析(ParseFromString)。
`caffe.get_solver("*.prototxt")`
可用.net引用到训练网络,test_nets引用到测试网络列表，有.step(n)方法前向反向传播n步并更新

- 计算处理
`caffe.set_device(gpu_id)`
`caffe.set_mode_{gpu,cpu}()`

*optional* preprocess  可用任何工具包得到ndarray数据，caffe 提供caffe.io.Transformer类:
`set_chanel_swap(name)`
`set_input_scale(name)`
`set_mean(name)`
`set_raw_scale(name)`
`set_transpose(name)`
`caffe.io.loadimage()` #　导入成ndarray　(HxWxC)
`preprocess(name,img)`