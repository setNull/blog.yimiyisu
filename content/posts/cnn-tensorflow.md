---
title: "TF实现的基于CNN的图片分类"
date: 2018-08-01T15:35:02+08:00
bigimg: [{src: "https://res.cloudinary.com/dh5dheplm/image/upload/v1533094873/samples/ecommerce/accessories-bag.jpg", desc: ""}]
draft: false
tags: ["CNN","tensorflow"]
categories: ["machine-learning"]
---

## 前言

在这篇Tensorflow入门教程中，我将用TF实现一个基于卷积神经网络（CNN）的图片分类器。
本文基于[tensorflow-tutorial](http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/)翻译整理而成，并对原文中提到的代码进行了详细分析和补充讲解。

在我们开始讲解tensorflow之前， 先介绍一下卷积神经网络的核心概念。如果你已经对CNN有所了解，你可以从第二部分开始， 即TF实现基于CNN的图片分类器。

## 第一部分： CNN Basics

神经网络本质上是解决优化问题的数学模型。神经网络的详细讲解，推荐[这个系列](https://www.zybuluo.com/hanbingtao/note/433855)。

### 层
（**TODO**）
#### CNN中层的种类
1. 卷积层
2. 池化层
3. 全连接层

### 理解训练的过程
（**TODO**）
1. 网络的结构
2. 迭代更新权重/参数

## 第二部分： TF实现基于CNN的图片分类器

本文中实现的网络，比起解决real-world问题的网络，是非常小非常简单的，甚至可以在CPU上进行训练。
该网络经过三个卷积层，然后拍平后再经过两个全连接层得到分类结果。
网络结构图：
![网络结构](http://res.cloudinary.com/dh5dheplm/image/upload/v1533135229/ml/xTensorflow-tutorial-2-1.jpg.pagespeed.ic.cetItSpDJP.png)

### 数据
本文数据集基于 [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats)。
一般的，我们把输入数据分成2部分：
1. 训练集：本文中每个分类500张图片
    * 用80%的训练集作为训练样本，用来更新网络权重/参数
    * 用20%的训练集作为验证样本，用来独立于训练过程计算准确率。
2. 测试集：用完全独立于训练集的图片(最好来自不同的数据源)作为测试集，本文中测试集一共400张图片。

### 创建网络层

* 卷积层

TF中卷积操作是用tf.nn.conv2d 函数， 它的输入是:

* input= 上一层的输出, 这是一个四维张量。典型的在第一个卷积层里，你输入了n个 width * height * num_channels 的图片， 那么input = ```[n width height num_channels]```

* filter= 可训练的变量定义了卷积核（filter）. 我们以随机正态分布开始学习这些权重。它是一个四维张量，其形状是设计网络时预定义的一部分。假设你的filter的大小为filter_size, 输入的channels是num_input_channels, 当前层有num_filters， 那么你的filter的形状如下：``` [filter_size filter_size num_input_channels num_filters]```

* strides= 定义了在做卷积运算时，以多大的步幅移动卷积核。在这个函数里， 它是一个尺寸大于等于4的张量 -- ```[batch_stride x_stride y_stride depth_stride]```,

    * batch_stride 总是1， 因为你不会想跳过任何图片。
    * x_stride, y_stride绝大多数情况下是一样的， 它的选择是网络设计的一部分。
    * depth_stride总是设为1，因为你不会在深度上跳过。

* padding=SAME的意思是需要在输入周围补0使得当前层的输出x,y 的维度和输入时一样。

卷积后， 我们在神经元上加上偏差biases，这个是可学习的参数。是以随机正态分布开始，然后在训练的过程中学习出来。

现在，我们来应用最大池化``` tf.nn.max_pool ``` , 这个函数的签名和卷积的函数类似。

``` python
tf.nn.max_pool(value=layer,ksize=[1, 2, 2, 1],
               strides=[1, 2, 2, 1],
               padding='SAME')
```
注意，这里我们k_size/filter_size选成2*2， 步长在x,y方向上也都是2。根据之前提到的公式 ``` (w2= (w1-f)/S +1; h2=(h1-f)/S +1 ) ```, 可以看到输出的维度是输入的一半。 这些是最大池化层经常选的值。

最后，我们选Relu作为激活函数， 在池化操作的输出上应用Relu -- ```tf.nn.relu```函数。

所有这些运算都是在一个卷积层上。 我们封装一个函数来定义完整的卷积层：
 ``` python
def create_convolutional_layer(input,
               num_input_channels, 
               conv_filter_size,        
               num_filters):  
    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)
    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')
    layer += biases
 
    ## We shall be using max-pooling.  
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)
 
    return layer
```

* Flattening层

The Output of a convolutional layer is a multi-dimensional Tensor. We want to convert this into a one-dimensional tensor. This is done in the Flattening layer. We simply use the reshape operation to create a single dimensional tensor as defined below:
``` python
def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])

    return layer
```

* 全链接层

Now, let’s define a function to create a fully connected layer. Just like any other layer, we declare weights and biases as random normal distributions. In fully connected layer, we take all the inputs, do the standard z=wx+b operation on it. Also sometimes you would want to add a non-linearity(RELU) to it. So, let’s add a condition that allows the caller to add RELU to the layer
``` python
def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True):
    
    #Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer
```
So, we have finished defining the building blocks of the network.



**（To Be Continue）**
* * *
* 占位及输入

Now, let’s create a placeholder that will hold the input training images. All the input images are read in dataset.py file and resized to 128 x 128 x 3 size. Input placeholder x is created in the shape of [None, 128, 128, 3]. The first dimension being None means you can pass any number of images to it. For this program, we shall pass images in the batch of 16 i.e. shape will be [16 128 128 3]. Similarly, we create a placeholder y_true for storing the predictions. For each image, we have two outputs i.e. probabilities for each class. Hence y_pred is of the shape [None 2] (for batch size 16 it will be [16 2].
``` python
x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)
```

* 网络设计
We use the functions defined above to create various layers of the network.
``` python
layer_conv1 = create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1)

layer_conv2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2)

layer_conv3= create_convolutional_layer(input=layer_conv2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3)
          
layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False)
```
* 预测

As mentioned above, you can get the probability of each class by applying softmax to the output of fully connected layer.

```y_pred = tf.nn.softmax(layer_fc2,name="y_pred")```

y_pred contains the predicted probability of each class for each input image. The class having higher probability is the prediction of the network. 
```y_pred_cls = tf.argmax(y_pred, dimension=1)```

Now, let’s define the cost that will be minimized to reach the optimum value of weights. We will use a simple cost that will be calculated using a Tensorflow function softmax_cross_entropy_with_logits which takes the output of last fully connected layer and actual labels to calculate cross_entropy whose average will give us the cost.
``` python
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
```

* 优化
Tensorflow implements most of the optimisation functions. We shall use AdamOptimizer for gradient calculation and weight optimization. We shall specify that we are trying to minimise cost with a learning rate of 0.0001.
``` python
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
```
As you know, if we run optimizer operation inside session.run(), in order to calculate the value of cost, the whole network will have to be run and we will pass the training images in a feed_dict(Does that make sense? Think about, what variable would you need to calculate cost and keep going up in the code). Training images are passed in a batch of 16(batch_size) in each iteration.


### 预测


完整的代码在[这里](https://github.com/sankit1/cv-tricks.com/tree/master/Tensorflow-tutorials/tutorial-2-image-classifier)





