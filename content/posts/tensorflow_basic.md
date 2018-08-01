---
title: "Tensorflow_basic"
date: 2018-08-01T08:23:39+08:00
bigimg: [{src: "https://ws1.sinaimg.cn/large/00704eQkgy1fsazo12js7j30xc0c5auf.jpg", desc: "Flowers|Hangzhou|Apr 5,2018"}]
draft: false
notoc: true
tags: [""]
categories: ["effective"]
---

# TensorFlow基础

本文基于[TensorFlow Official Tutorial](http://cv-tricks.com/artificial-intelligence/deep-learning/deep-learning-frameworks/tensorflow-tutorial/)，结合平时使用，手把手教你入门TF。

[TensorFlow](https://www.tensorflow.org/?hl=zh-cn)是一个用于数值计算的库，其中数据沿着图（graph）而流动。在TensorFlow中数据用n维数组表示并被称之为Tensors。而图（graph）由数据(也叫Tensors)和数学运算共同组成。

* 图中的节点： 代表数学运算
* 图中的边(edges)： 代表在不同运算中流动的Tensors

TensorFlow不同于其他编程语言的另一个方面是： 在TensorFlow中无论你要构建什么，首先你需要构思整个蓝图。在创建图的时候，变量并没有被赋值。随后当已经创建了完整的图之后，还需要在一个会话(session)中去运行它，此时图中的变量才会被赋值。

现在让我们通过动手来学习。运行Python并导入tensorflow：
``` bash
sankit@sankit:~$ python
Python 2.7.6 (default, Oct 26 2016, 20:30:19) 
[GCC 4.8.4] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>>
>>> import tensorflow as tf
```

### TensorFlow中的图
图是TensorFlow的主干，所有的计算/操作/变量都位于图中。代码中发生的一切都位于TensorFlow提供的一个默认图中。可以通过如下代码访问该图：

``` python
graph = tf.get_default_graph()
```
你也可以这样得到所有操作的list：
``` python
graph.get_operations()
```
由于现在图是空的，所以该语句的输出也是空的,即[]。
如果想打印出各操作的名称，用这条语句：
```
for op in graph.get_operations():
    print(op.name)
```
这回还是空的，等我们在图中加入了操作之后再来执行该语句。
另外，我们也可创建多个图，现在先不细讲。(**TODO**)

### TensorFlow会话
图是用来定义操作的，而操作必须运行在一个会话(session)中，图和会话的创建是相互独立的。可以把图想象成是设计蓝图，则会话就是它的施工地点。

图仅仅是定义了计算或者说构建了设计蓝图。 然而，除非我们在会话中运行图或者图的一部分，否则没有任何变量也没有任何值。

可以这样创建会话：
```

sess=tf.Session()
... your code ...
... your code ...
sess.close()
```
打开一个会话时，要记得在结尾处关闭。或者可以用python中的with语句块，如此一来，它将会自动被关闭：
```
with tf.Session() as sess:
    sess.run(f)
```
### TensorFlow中的Tensors
TF将数据保存在Tensors中，它有点像numPy包中的多维数组（尽管它们和numPy数组不同）

* 常量

常量的值不能修改，定义方式如下：

```
a=tf.constant(1.0)
a
<tf.Tensor'Const:0' shape=() dtype=float32>
 print(a)
Tensor("Const:0", shape=(), dtype=float32)
```
可以看到，不同于Python之类的其他语言，这里并不能直接打印/访问常量的值，除非在会话中运行，再来试一下：
```
with tf.Session() as sess:
    print(sess.run(a))
```
这回打印了输出结果1.0

* 变量

即Tensors，和其它语言中的变量相似。
```
>>>b = tf.Variable(2.0,name="test_var")
>>>b
<tensorflow.python.ops.variables.Variable object at 0x7f37ebda1990>
```
变量（顾名思义）和常量不同，能够存储不同的值。然而，在TF中，变量需要分别进行初始化，单独初始化每个变量效率很低。但TensorFlow提供了一次性初始化所有变量的机制，具体方法如下：

对于0.11及更早的tf版本，使用initialize_all_variables()方法：
```
>>>init_op = tf.initialize_all_variables()
```
0.12及以后的版本，使用global_variables_initializer():
```
>>>init_op = tf.global_variables_initializer()
```
上述代码会把init_op添加到TensorFlow的默认图中。

现在，试图访问刚才定义的变量b之前，先运行一下init_op，打印b输出2.0：
```
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(b))
```
现在可以打印出该图中的全部操作：
```
graph = tf.get_default_graph()
for op in graph.get_operations():
    print(op.name)
```
This will now output:
```
Const
test_var/initial_value
test_var
test_var/Assign
test_var/read
init
```
如你所见，之前定义过常量a, 所以它被加到了图中。同理，对于变量b而言，许多’test_var’的状态，例如test_var/initial_value，test_var/read等也被加入了图中。你可以利用TensorBoard来可视化整个网络，TensorBoard是一个用于可视化TensorFlow图和训练过程的工具。

* 占位符

占位符，顾名思义表示占位，是指等待被初始化/填充的tensors。占位符被用于训练数据，只有当代码是在会话中运行的时候占位符才会被填充。“喂给”占位符的东西叫做feed_dict。Feed_dict是用于存储数据的（一系列）键值对：
```
>>>a = tf.placeholder("float")
>>>b = tf.placeholder("float")
>>>y = tf.multiply(a, b)
 // Earlier this used to be tf.mul which has changed with Tensorflow 1.0

//Typically we load feed_dict from somewhere else,
 //may be reading from a training data folder. etc
 //For simplicity, we have put values in feed_dict here
>>>feed_dict ={a:2,b:3}
>>>with tf.Session() as sess:
       print(sess.run(y,feed_dict))
```
上例输出结果为6.

* 在TensorFlow中应用设备
TensorFlow具有非常强大的内置功能，可以在gpu, cpu或者gpu集群上运行你的代码。 它为你提供了选项，使你能选择要用来运行代码的设备。 这里不对此进行详细介绍，随后会有单独关于这个主题的教程。先来看一下TensorFlow全貌：
![TensorFlow全貌](http://cv-tricks.com/wp-content/uploads/2017/02/xTensorflow_Graph_0.png.pagespeed.ic.U_RLEnluD2.webp)

下一篇介绍TF的简单代码样例。 我们会一起学习线性回归的代码。

