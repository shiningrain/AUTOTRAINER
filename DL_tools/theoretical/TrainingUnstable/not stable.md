[TOC]

### 训练不稳定问题描述

	训练不稳定是指在训练过程中，loss函数震荡，从而导致神经网络一直无法收敛。

[参考链接](https://en.wikipedia.org/wiki/Stability_(learning_theory))

**gradient vanish问题出现时，权值会更新过慢，神经网络没有办法很好的收敛**

训练不稳定的原因：

1） 优化器选择问题.

2） 学习率过大

3） Batch Size选择不当

4） 过拟合问题


### 几个可能有效的解决方案


#### [ADAM自适应优化器](https://arxiv.org/pdf/1412.6980.pdf)

文章描述：
Adam 通过计算梯度的一阶矩估计和二阶矩估计而为不同的参数设计独立的自适应性学习率，通过使用该优化器使得机器学习的效率以及健壮性更好。

实验结果：
在我们的实验中，大部分梯度不稳定问题，可以利用adam缓解，且收敛的速度相比较sgd更快。

有效性分析：
通过实验分析，对于SGD优化器，所产生的训练不稳定，可以利用adam代替，Adam优化器会自适应的调整学习率，通常会有好的结果，而且对于神经网络改变较小。**是解决训练不稳定最简单最便利的方法之一**


#### [减小学习率](https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/)

博客描述：
该博客提出学习率对于神经网络的训练的重要性，当你设置了过大的学习率时，因为会可能会不停的越过loss最小的那个点，所以loss曲线可能会一直处于震荡模式。

实验结果：
在我们的实验中，减小学习率，在大部分实验中都可以解决训练不稳定。

有效性分析：
因为单纯的只考虑学习率的大小而不考虑训练数据本身，是存在缺陷的，我们对于一个不稳定问题，我们很难设计一种模式去自动判断需要减少多少学习率才合适——过小的学习率会引发不收敛问题或者为后续的训练带来时间代价，而大的学习率又无法解决问题。尽管大部分实验中，我们发现单纯的减小学习率是可以解决的，但该解决方案仍只作为一种**较优的方案**，而非首选方案。


#### [Momentum](https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/)

博客描述：
Momentum算法借用了物理中的动量概念，它模拟的是物体运动时的惯性，即更新的时候在一定程度上保留之前更新的方向，同时利用当前batch的梯度微调最终的更新方向，动量限制了loss收敛局部范围内的抖动。

实验结果：
在我们的实验中，Momentum只在特定的实验中，可以解决梯度不稳定问题。在大部分实验中，仅仅先比较原来有改进，但是，没有前面两种方法好。

有效性分析：
通过实验可以得出结论，Momentum暂时作为一个**备选方案**，去尝试解决训练不稳定的问题。

#### [改变batch size](https://machinelearningmastery.com/how-to-control-the-speed-and-stability-of-training-neural-networks-with-gradient-descent-batch-size/)

博客描述：
文章通过改变batchsize的大小，解释了在较小的batchsize可能会出现快速学习，但学习过程不稳定。大批量的学习会使得训练的时间增多，因此需要选择较为合适的批次让模型更好的训练。

实验结果：
在我们的实验中，在少量的实验中，尤其是未利用初始化的32作为batchsize的值，解决效果较好，

有效性分析：
BatchSize的改变可以通过改善模型每次学习到的知识来缓解收敛稳定性的问题，事实上，该方法存在一定的局限之处，例如我们很难确认一个模型合适的训练batch是多少，我们只能根据当前表现尝试去对batch进行增大或者缩小，因此该方法作为一个**备选方法**。


#### [Gaussion Noise](https://machinelearningmastery.com/how-to-control-the-speed-and-stability-of-training-neural-networks-with-gradient-descent-batch-size/)

博客描述：
Gaussion Noise是一种将噪声添加到具有少量训练数据集的约束不足的神经网络模型中可以产生正则化效果并减少过度拟合的方法。这里我们利用Gaussion Noise 主要是缓解**test loss的震荡的问题**。

实验结果：
在我们的实验中，在test loss曲线震荡而train loss曲线不震荡的实验情况下，有很好的结果。

有效性分析：
通过实验，我们可以得出结论，对于这种**过拟合导致的训练不稳定问题**的情况，可以采用此解决方案。

#### [ReduceLROnPlateau(callback)](https://keras.io/zh/callbacks/)

回调函数描述：
ReduceLROnPlateau是一种通过监督test loss的变化，从而改变学习率的函数。每当test loss不再改变时，将学习率减小。

实验结果：
在我们的实验中，该回调函数的解决结果一般和减小学习率这种方法的相同。

有效性分析：
ReduceLROnPlateau检测Loss的变化，一旦长期loss不衰减，则减小学习率，使模型在更小的解空间内搜寻更优秀的参数组合，可以作为我们**减小学习率的一种实现方法**，在大部分情况是有效的。

#### 总结：

对于不稳定的问题，一般的解决方案有四种，使用自适应优化器、减小学习率、使用Momentum、使用ReduceLROnPlateau回调函数。对于特定问题batchsize问题，可以通过设置合适的一次训练样本决定。对于test loss不稳定的情况，可以使用Gaussion Noise解决。
