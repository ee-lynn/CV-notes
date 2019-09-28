# python进阶笔记
  sqlu@zju.edu.cn
考虑到python2.x将在2020年退役，本笔记聚焦python3，在必要处指出python2.7的不同之处。

## 1.动态类型模型 
### 1.1引用计数
- python中变量都是对象的引用,当该对象的引用计数归零后，内存即被回收，由此实现了垃圾回收。类型是针对对象而言的而不是对变量而言的，由此实现了动态类型
- == 操作符递归比较变量引用的值是否相等, is操作比较两变量是否引用同一个对象
### 1.2可变/不可变对象
- 通过变量引用的对象有些是不可变的，如字符串，数字，元组。修改不可变化变量对导致变量引用另外的对象，python会缓存简单的不可变对象使相同的变量引用同一个对象，稍复杂情况下都会变成不同副本。
- 有些则可以改变，如字典,列表，类，可变对象在多引用情况下，修改一个变量可导致所有引用对象的变化. 多类型的嵌套时也递归地符合上述规则,并且普通的copy只能拷贝顶层数据，深层的成员仍旧是多引用，需要用deepcopy才能递归的复制
## 2.布尔模型
### 2.1布尔测试
- 布尔测试中,任何非零数字和非空对象将返回真，数字零或空对象即None会返回假
- 对实例判断时,优先调用__bool__,若没有实现则调用__len__
- and or 都会采取短路逻辑返回使得条件返回时的对象[不是True或False]
### 2.2三元表达式
- `b if c else d`  等价于C中 `c? b:d`
## 3.循环模型
### 3.1生成器与生成器表达式
- 生成器是一个视图，每次返回一个值，函数中用yeild代替return，表达式中用f(x) for x in S if h(x)表达.
- 当生成器耗尽后会抛出StopIterationError
### 3.2迭代器协议
- 在for a in b: dosomething 语法中,等价于
    ```
    while True: 
        try:
            b = iter(b)
            a = next(b)
            dosomething
        except StopIterationError:
            break
    ```
- python3中next()调用实例的__next__(),python2中调用实例的next()       
- iter(b)优先调用b.__iter__，迭代器是含有__next__的类,迭代一轮后便耗尽了,若要支持多迭代,则iter需要返回新迭代实例(而不是具有__next__方法的自身)
当b没有实现__next__时,调用__getitem__，直到IndexError
### 3.3推导式
- 列表,字典,集合都有其推导式,可用构造器将生成器表达式展开
`[generator]`
`list(generator)`
`{generator}`
`set(generator)`
`{key:val for zip(generator1,generator2)}`
`dict(key(x)=val(x) for x in generator)`
## 4.函数模型
- 运行时将def a：语句转化为module.a = a,内部的语句将在调用时运行,(module是文件名),a是types.functiontype,有__code__属性，有以下内省属性
    ```
        co_argcount          #位置参数个数
        co_kwonlyargcount    #关键字参数个数
        co_varnames          #被赋值的变量名
        co_freevars          #仅被引用(未赋值)的变量名
        co_nlocals           #函数中局部参数(位置参数+关键字+可变参数 + 本地变量)
        co_filename          #函数所属的文件
        co_name              #函数名
        co_firstlineno       #函数在第几行
    ```    
### 4.1作用域
- 给变量赋值的地方决定了这个变量将存在于哪个命名空间,如果在def内赋值,他被定位在函数之内(每次函数调用都生成新的作用域);如果在嵌套的def内,对于嵌套的函数而言，它是nonlocal的;如果在def之外赋值,他就是整个模块的
- 赋值[广义的赋值，包括使用def,import,参数传递等赋值方式,但不包括对可变对象的修改]的变量名除非声明为global[位于def以外的变量]或nonlocal[仅位于嵌套def中],否则均为本地变量.所有变量名都可以归纳为本地、全局和内置[由builtins模块提供,但builtins本身不是内置的,显式使用内置变量时需要导入包]的。
- python在使用变量时，依照 本地作用域(Local)->外层结构的def或lambda本地作用域(Enclosing) -> 全局作用域(Global) -> 内置作用域(Builtin) 的顺序查找变量
### 4.2变量协议
- 函数调用时参数必须以 **位置参数(value),关键字参数(name = value),\*iterable, \*\*dict** 的顺序出现
- 函数定义时参数必须以 **一般参数(name),默认参数(name= value),\*name, name或者name=value 这样的keyword-only参数，最后是\*\*name**
- python在匹配时,通过位置分配非关键字参数,通过匹配变量名分配关键字参数,其他额外的非关键字分配到\*name元组中,其他额外的关键字参数分配到**name字典中,用默认值分配给未得到分配的参数, 默认参数是在def运行时绑定的[类似于static变量],因此在调用时修改可变对象默认值会造成后续的影响
- 函数注解在def头部行,对于参数,紧随参数名之后,默认值之前的冒号,对于返回值紧随在参数列表的->. 当注解出现的时候,python将其收集到函数对象的__annotation__的字典中,参数名和"return"成键
### 4.3匿名函数
- 匿名函数是一个表达式,出现在def不能出现的地方
`lambda arg1,arg2...argN: expression using args`
## 5.模块模型
### 5.1模块运行规则
- python在导入模块时,并非像c中#include那样将代码直接插入，而是执行了三个步骤：(1)找到模块文件,(2)编译成字节码,执行模块的代码来创建其所定义的对象.
(3)将该对象保存在sys.modules中。三步骤仅在第一次导入时发生，之后便直接从sys.modules中获取.
- 在(1)寻找模块时，python在程序主目录,PYTHONPATH,标准链接库目录和.pth文件指定的目录中依次寻找,这些目录会保存在sys.path中
- 模块名不需要后缀，他会链接.py/.pyc/.so/.dll/目录[包导入]/zip等文件(该行为由__import__("module")定义,包括解压等)
`from A import B` 等价于
```
import A
B = A.B
del A
```
模块只会导入一次,需要重新导入时需要调用imp.reload(module)[python2有内置的reload函数],原位覆盖了module对象,造成全局的变化,且仅影响顶层模块(不对模块中模块递归reload)，如前from等价过程来看，也不会影响from的对象
- __all__列表将是import \*需要导出的所有对象，而单下划线**_**开头的变量不会被导出
### 5.2包
- python代码的目录称为包，包导入把一个目录变成python的命名空间,属性则是对应于目录中包含的子目录和模块文件,在路径中需要有__init__.py文件,该文件在导入目录时执行，其中被赋值的对象成为目录属性
- 包支持相对导出,即`from xx import xxx` 中`xx`可以是`.`,`..`,`.B`,`..B`这样的结构,而不以点开头的模块均会被认为是绝对导入,即从sys.path路径中寻找.相对导出使得包管理更加方便.
## 6.类模型
- 所有的类都是type类的实例,在创建类最后插入运行type实例化,用于管理类.默认情况下为
`class A:...   -> A=type(classname:str,superclasses:tuple,classdict:dict)`其中`type.__call__()`调用了
```
__new__(meta,classname,superclasses,classdict)返回新创建的类
__init__(class,classname,superclass,classdict)初始化类,无返回值
```
type是默认情况下所有类的元类，也可以自定义元类,用于替换插入实例化时运行的代码
在python2中，新式类定制元类需要
```
class A(object):
    __metaclass__ = xxx
```
python3中，定制元类需要
```
class A(super,metaclass = xxx):
```
即可在类创建末尾插入运行`xxx(classname,superclasses,classdict)` 其中`classdict`相当于`xxx.__dict__`
### 6.1实例化和继承协议
- class语句创建类对象并将其赋值给变量,class语句内所有赋值会在这个类作用域中创建属性,由所有实例共享,方法内对self属性做赋值会产生每个实例的属性,每个实例私有
- 在属性引用时首先检查实例,然后是它的类,最后是所有的超类。实例与类之间的联结通过__class__实现,子类和超类通过__bases__联结,本级的属性都在__dict__中,属性搜索通过对__dict__结构的广度优先搜素实现(注:在python2经典类中属性搜索是深度优先搜索).
- 在类中查找到属性后,instance.method(args)自动转化为class.method(instance,args)
- 对属性赋值只会修改该对象
- 下划线_仅是私有属性惯例,没有语法意义,而对于双下划线开头的变量(末尾没有双下划线,区别于魔法方法),会被转换成_classname__propertyname,用于避免冲突,但仍避免不了强行被访问
- __slots__元组规定了本级类的属性,以外的属性在运行时不可赋值. 当__slots__含有'__dict__'时,slots就没有限制作用了.这时候slot和dict共同组成了命名空间
### 6.3魔法方法
- 在python中,内置运算符会触发魔法方法的调用,它越过了实例的属性搜索而直接在类内查找(python2经典类与常规属性搜索顺序相同,因此使用经典类用getattr可以拦截魔法方法,而新式类需要重写)
- 常见的魔法方法有
```
__init__           #构造器
__add__            #重载operator+(self,others), operator+=(self,others) __iadd__专门用于重载+=
__repr__,__str__   #打印默认情况下提供对象类名+内存地址，打印会试图调用__str__,未实现时调用更底层的__repr__.需要注意的是在交互时显示对象时调用__repr__,且__str__仅能对最顶层的对象进行调用,不会被别的对象的__str__递归调用,这种情况下需要实现更底层的__repr__
__call__           #可调用对象
__getattr__,__getattribute__,__setattr__,__delattr__ #用于属性管理
__getitem__,__setitem__，__delitem__ #涉及索引的读/设/删操作,__getitem__额外支持了迭代协议,以OutofIndex替代StopIterationError
__len__            #len()或未实现__bool__时的布尔测试
__bool__(python2中 __nonzero)__    #布尔测试
__iter__,__next__  #支持迭代协议.
__contains__       #operatorin(),未实现时以 __iter__,__next__ 或__getitem__在迭代中搜索
__index__          #将类映射成一个整数,特别地，当类出现在索引处时
__enter__,__exit__ #with/as协议
__get__,set__      #用于描述符
__new__            #用于元类
```
### 6.4 静态方法和类方法
- 类的常规方法会默认映射成class.method(instance),当仅需要管理类的成员时,不希望依靠instance,在python3中可以直接跟普通函数一样调用class.method(args),
在python2中这种无绑定(unbound method)的调用方式必须额外传入instance. 可以使用静态方法声明将实例去除,使得可以使用实例来调用静态方法
#### 6.4.1 静态方法
使用装饰器@staticmethod声明静态方法,装饰器实际上把第一个self参数删去,并且代理了本身的函数。
#### 6.4.2 类方法
使用@classmethod声明类方法,装饰器实际上把第一个self参数用self.__class__代替,并且代理了本身的函数
#### 6.4.3 装饰器
装饰器有函数和类装饰器之分，在声明前@即可，表示在完全执行完后插入调用装饰,主要用于管理函数和类实例
```
@decorator`
class A/def A:
-> A = decorator(A)
```
### 6.5属性管理
```
a = A()
a.name         #会从实例a和类A中寻找name成员,类A中name的property对象;实现了__get__方法的描述符类实例,或者调用重载函数
a.name = name  #会调用property.setter;实现了__set__方法的描述符类实例,或者调用重载函数
del a.name     #会调用property.deleter;实现了__delete__方法的描述符类实例，或者调用重载函数
```
#### 6.5.1 property
实例化形式 property(get,set= None,del=None,doc=None),有setter(set),deleter(del)方法,一般情况下可用装饰器的形式把第一输入参数包装成property对象
#### 6.5.2描述符
一个实现了__get__(self,instance,owner),__set__(self,instance,owener)的类.self是描述符对象本身,instance是引用属性的对象,owner是引用属性对象的类
#### 6.5.3函数重载
`__getattr__(self,name)`捕获未定义的属性
`__setattr__(self,name,value)`捕获所有的属性
`__getattribute__(self,name)`捕获所有的属性
- 为避免循环调用(在不火属性时进一步引用属性引起的递归调用,应对__dict__才做[该成员仍会被__getattribute__捕获],或者调用父类[如object]的对应函数(__getattr__等))
- 需要指出的是在python3中重载的魔法方法已经不能被捕获，在使用代理模式下需要重新实现,而在python2中魔法方法也会被属性重载方法捕获
## 7.python的字符串模型
- python2中提供str和unicode类型. str表示每个字符可以用一个字节解释的字符串,同时任何二进制串也可以用此类型表示,unicode类型表示宽字节字符串，可以用u'spam'表示
- python3中将字符串统一用unicode[u''或U'']表示,二进制串用bytes[b'',B'']和bytearray[bytes的可变形式]类型表示
- encode("encoding")将unicode字符串(python2中unicode或者python3中str)转化成二进制串(python2中str或者python3中bytes)
- decode("encoding")将二进制串转化成unicode字符串.其中encoding指定编码方式[例如"utf-8","ascii","gbk"],缺省情况下取sys.getdefaultencoding()
- python2在读写文本时,全部都是以str形式读取,t和b差别仅是是否做行末识符转换[\r\n,\n]
- python3则根据w和t区分返回str还是bytes,返回str时根据指定的编码格式解码
- 在unicode字符串中,要打键盘无法表示的字符时,可以使用'\xNN'或者'\uNNNN','\uNNNNNNNN'表示,但二进制串中每个字符只能一个字节,即不支持\u转义
## 8.附录
- 类装饰器管理实例的典型形式
```
def wrapper(*wargs):   #带有参数的装饰器(没有参数时可以这层)，使用局部作用域保存wargs参数
    def onDecorator(aClass):   #装饰器
        class onCall(*args):   #管理实例
            def __init__(self,*args):
                self.wrapper = aClass(args)   #管理的实例都由onCall对象代理
            def __getattr__(self,name):
                getattr(self.wrapper,name)
            ...            #可使用wargs,aClass实例的信息扩展，实现装饰效果
        return onCall
    return onDecorator
@wrapper(xxx)
class myclass:...
```
- 也可以管理类
```
def wrapper(*wargs):   #带有参数的装饰器(没有参数时可以这层)，使用局部作用域保存wargs参数
    def onDecorator(aClass):   #装饰器
        aClass...   #可使用wargs,aClass类的信息扩展，实现装饰效果
        return  aClass
    return onDecorator
@wrapper(xxx)
class myclass:...
```
- 元类管理类的典型形式
```
class metaClass(type):  #这里没有定义__call__和__init__,从type中继承
    def __new__(meta,classname,supers,classdict)
        classdict... # 根据各种信息对classdict(类属性)进行加工处理
        return type.__new__(meta,classname,supers,classdict)
class myclass(metaclass = metaClass):...
```
- 也可以管理实例
```
def metaClass(classname,supers,classdict):
    aClass = type(classname,supers,classdict)
    def onCall(*args):
        aInstance = aClass(*args)
        ... #用实例信息加工实例
        return aInstance
    return onCall
class myclass(metaclass = metaClass):...
```
## 9.python的C/C++扩展 
todo