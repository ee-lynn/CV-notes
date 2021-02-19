# 在Linux上工作需要了解的基本内容
 　　　　　　　　　　　                    &nbsp;    　sqlu@zju.edu.cn
## 引言
以前一直在windows上工作,现在切换到服务器后，由于操作系统都是linux,难免会有不适应。靠着几个基本命令和网上碎片式的查询勉强度过了一段时间。现在将linux的皮毛系统地学习了一遍，现总结要点，供以后遗忘后参考。
我将linux学习笔记分成 个方面:

- 初识linux
  - 约定俗成的文件组织结构及其用途
  - 系统环境配置
- 文件系统和目录
  - 导航
  - 文件操作
  - 存储介质挂载
  - 文件搜索
  - 权限
- shell特性和命令
  - 宏替换
  - 提示符
  - 命令文档
  - 系统包管理
- 输入输出
  - 重定向
  - 打印
- 文本处理
  - 正则表达式
  - 文本处理命令
- 编译程序
  - 程序编译安装
  - Makefile书写
- shell编程
- 版本管理(git)
### C/C++程序编译调试

- Makefile整体结构
Target1:dependencies
`\t`system command...
Target2:dependencies
`\t`system command...
- 默认执行仅执行第一条规则,并由依赖项进一步触发别的规则.也可以在make后显式规定执行某条 make someTarget
- 最开始的**`\t`**是必须的,不能替换成4个空格
- Target(时间戳为T1)，dependencies(时间戳为T2)
- 若没有Target文件，则T1=0，dependencies为空时，T2=0.
    `if（T1==0） exec Target;`
    `if（T1<T2） exec Target;` 
    `else print("up to date")`
- 注释变量与函数
  - #为注释
  - 变量一般大写，类似于字符串，支持 **+=**操作符号，$()可以引用变量
  - 为了在执行时不显示命令本身，在命令前加**@**
  - $@ 指代target， $^指代dependencies列表，$<指代依赖表第一项
  - Makefile 中支持一些预定义的函数: $(函数名 参数列表) 参数以逗号分开，只能是Makefile自带的函数。 e.g. `PWD = $(shell pwd)`,其他支持的常用函数有
    - wildcard(pattern) 罗列参数形式的所有文件,e.g. wildcard(*.cpp)
    - patsubst(srcpat,dstpat,content) 把content中的srcpat代替为dstpat, e.g. patsubst(%.o,%.cpp,test.cpp)
    - foreach(iterator, iterable, apply_func) 在迭代器iterable中逐个对iterator执行apply_func
  - % 通配符, 一条命令相当于好几条命令.
  - g++ -c some.cpp表示编译, -o 表示输出
  - g++ -MMD some.cpp会生成*.d文件，其内容为some.cpp的所有依赖项，将它-include进Makefile可以保证不遗漏依赖项实现增量编译(特别针对.h文件难以手动罗列出的情况)

- 多子目录
  - 需要定义src_dir = ... .., 然后用foreach 遍历整个项目文件后生成CXX_SOURCES
  `SRC_DIR = src object`
  `CXX_SURCES = $(foreach dir, $(src_dir), $(wildcard dir/*.cpp))`
- 一种通用的Makefile结构
`TARGET=`
`SRC_DIR = src object`
`CXX_SURCES = $(foreach dir, $(src_dir), $(wildcard dir/*.cpp))`
`CXX_OBJECTS = $(patsubstr %.cpp, %.o, $(CXX_SOURCES))`
`DEP_FILES = $(patsubst %.o, %.d, $(CXX_OBJECTS))`
`CXXFLAGS #编译选项`
`LDFLAGS  #链接选项`
`$(TARGET): $(CXX_OBJECTS)`
`\tg++ $(CXX_OBJECTS) $(LDFLAGS) -o $(TARGET)`

`%.o:%.cpp`
`\tg++ -c -MMD $(CXXFLAGS) $< -o $@`
`-include $(DEP_FILES)`

`clean:`
`\trm  -rf $(CXX_OBJECTS) $(DEP_FILES) $(TARGET)`

- 库
  - 动态(共享)库名称均为lib**name=**.so,静态库名称均为lib**name**.a
  - 动态库生成:编译选项加上-fPIC(position independent code),联结阶段加上-shared
  - 静态库生成:编译得到.o文件之后打包 `ar　-rcs lib**name**.a file1.o file2.o...`
  - 使用库时,编译选项加上-Iinclude_path(#include <xxx.h>从include_path中找,而"xxx.h"先从当前文件夹找,找不到按照前者搜索顺序搜索).链接选项需要加上-L**lib_path** -l**lib_name**  使用静态库时,也可以把它当做几个.o文件的集合一样直接写链接命令
  - 编译运行时,会从/lib, /usr/lib, /usr/local/lib, LD_LIBRARY_PATH中寻找依赖库.找不到需要export LD_LIBRARY_PATH = lib_path
  - g++默认链接libc.so(ANSI C),libstdc++.so(c++,包含STL)
  - 同时存在静态库和动态库时，编译器优先选择动态库,因此两者同时存在时,用静态库的第一种使用方法
- 调试技术
  - 在编译阶段需要添加　**-g**　命令
  - gdb 后，r(run), b(break),  c(continue),  p(print),  n(next), q(quit)
  - b 函数名/文件名:行号/当前文件的行号/类名:成员函数 
  - 显示断点 info break
  
  - 删除断电 del break 断点编号
  - disp variable   相当于VC中的watch，执行后均显示值
  - 还可以看内存中的值 x/[个数][显示格式x/d/u/f/s][单位b/h/w/g] variable.  x(16进制)d(decimal)u(unsigned int)f(float)s(string) b(1)h(half word2)w(4)g(4 giant)个字节
  - bt(backtrace)显示函数调用栈, nm查看符号, T(text)表示已经定义,U(undefined)表示未定义, readelf -d program 可以查看program依赖哪些库(或者ldd)
  - 内存转储: `ulimited -c unlimited`在挂掉时就会生成core文件, `gdb program core` 即可查看挂掉时的现场信息.
  
## shell编程
- shell 文件头: #! /bin/bash
- 给变量赋值直接等号即可,等号左右不能有空格。 引用时使用$var或者${var}  字符串变量最好用引号引起来，因当其中含有空格，会被断开
- read -p "xxxx" var  <-> scanf("xxxxx",&var)
- 参数在脚本里可以用$0,$1,$2...引用,$0是脚本本身，之后为输入变量列表，$#为参数个数, $? 表示上一条命令退出状态,0成功,非0失败. $@所有变量,可迭代
- $[ 计算字符串 ]、$((计算字符串))可以用于计算表达式的值 
- $(shell 命令) 或者`shell 命令` 可以执行命令
exec >.log  2>.mistake.log 表示后面脚本里日志输出位置
- **选择分支结构**
`if  ((condition))`    
　`then　statement`
`elif [ condition ]`
`then`
　`statement` 
`else`
　`　statement`
`fi`　
- then写成一行，需要加;  把(())改成[ ]其中逻辑运算符就必须跟fortran类似　-gt -eq
 etc.   != 和==只支持字符串  多个条件使用 &&,||,！ 联结 [] 
- []比较灵活，前后需要空格　可以直接在里面写命令　e.g.　-f 是文件，-d是文件目录 -e 是否存在   -s 文件存在且不为空
- -z "$var"  判断变量为空字符串, -n "$var"  判断变量不为空字符串
- **多选择分支**
`case var in`　　
   `condition 1)`
  　　　　`statement`
　　　`;;`
　　`condition 2 | condition 3)`
　　　　`statement`
　　　　`;;`
　　`*)`
　　　　`statment`
　　　　`;;`
`esac`
- **循环结构**
python style
`for iter in iteratable`　　#{start..end..interval} 表示range(start,end+1,interval)
　　`do statement`
`done`

C style
`for((i=0;i<n;i++))`
`do statement`
`done`


`while condition`  
`do statement`
`done`

- **函数**
`func()`
`{`
`statement`
`return var`
`}`　#以宏定义的方式去理解它，其中的参数都通过$012...引用
