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
1.卷积层
2.池化层
3.全连接层

### 理解训练的过程
（**TODO**）
1.网络的结构
2.迭代更新权重/参数

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
2. 测试集： 用完全独立于训练集的图片（）作为测试集， 本文中测试集一共400张图片。

### 创建网络层

* 卷积层


* Flattening层
* 全链接层
* 占位及输入
* 网络设计
* 预测
* 优化
* 

### 预测


完整的代码在[这里](https://github.com/sankit1/cv-tricks.com/tree/master/Tensorflow-tutorials/tutorial-2-image-classifier)





