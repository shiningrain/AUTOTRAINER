[TOC]

### 问题描述

> DyingReLU发生时，一定比例（问题发生时该比例一般较大）的神经元输出为0，对于任何输入，权重都难以继续更新。 可能是通过为其权重学习一个较大的负偏差项来实现的。
>一旦ReLU以这种状态结束，就不太可能恢复，因为0处的函数梯度也为0，因此梯度下降学习不会改变权重。 对于负输入，具有较小正斜率的Leaky ReLU（当x <0表示，y = 0.01x）是改善此问题并提供恢复机会的一种尝试。
>sigmoid和tanh神经元的值饱和时可能会遇到类似的问题，但始终至少存在一个小的梯度，可以使它们长期恢复。

[参考链接](http://theprofessionalspoint.blogspot.com/2019/06/Dying-ReLU-causes-and-solutions-leaky.html)

**DyingReLU出现时，层的output和gradient都是0，对于预测没有贡献。**

死亡的原因：

1） 学习率过高

2） 存在不合理的样本或者不恰当的初始化，一个比较大的负bias导致进入DyingReLU



### 几个潜在的j解决方案


#### [SeLU](http://papers.nips.cc/paper/6698-self-normalizing-neural-networks.pdf)

scaled exponential linear units 比例指数线性单位，具备自规范化特性。文章SeLU的paper证明了通过许多网络层传播的接近于零均值和单位方差的激活将**收敛**于零均值和单位方差（即使存在干扰和扰动），这使得selu构成的神经网络具备：**强大的正则化能力和鲁棒性**；他们的工作还证明了方差的上下限——梯度消失和爆炸不可能在这个网络中发生。selu初始化要求均值等于0，方差var等同于1/n，该[博客](https://link.medium.com/UNwRMiEEI2)中使用lecun_normal初始化了selu。优化器方面，论文中提到SGD、Adam、Adamax、Adadelta均训练的很好。

SeLU特点：**自标准化，自归一化，尤其适用于深层网络（16层以上）。**

有效性分析：
复现的单例中出现的DyingReLU问题很大程度上与初始化器初始化的数值相关，即因为较大的负向bias引发的死亡。初步认为是与selu配套使用的lecun_norm初始化器改善了这个问题——**事实上，在测试中仅用selu激活函数不改变配套的其他设置是不能消除DyingReLU问题的。**


#### [改变初始化器](https://arxiv.org/abs/1903.06733)

这篇文章提出使用非对称随机分布（RAI）来进行初始化，并在试验中取得了远超对称初始化的效果（问题出现几率he_uniform76.8%，RAI6.3%），这也说明了对于DyingReLU问题，初始化器可能带来的影响。这篇文章对DyingReLU进行了分类，分成了训练前就die和训练中die。
在我们的实验中，复现DyingReLU使用的是random_uniform初始化器，该初始化器随机按照均匀分布生成随机张量，我们设定的是-1到+1区间，随机生成的值比较大，因此在迭代中，会很快的触发问题（训练前die，认为是计算中溢出了）；如果使用默认设定的RandomUniform也可以触发dyingrelu问题，不过触发几率相对较低。如果使用he_uniform等基于截断正态分布的初始化器（直接使用默认参数），那么该问题会得到较为有效的改善。

有效性分析：之所以改变初始化器（仅仅从random_uniform改编为he_uniform）就会有效果，在一定程度上和初始化的权值有关。一开始random_uniform的权值初始化得绝对值初始化的太大了，得到的loss非常大（甚至可能溢出了）且难以有效的进行训练，计算时可能会出现值为负数的情况从而被ReLU变成0，最终导致训练前就死亡了。事实上我们在改变random_uniform的上下界后，该问题得到了一定的改善，但训练中依然更新困难，存在训练中的die问题。这里使用he_uniform以及lecun_uniform都可以改善问题，**我认为一定程度上得益于截断的正态分布（该结论仍缺少理论依据）**


#### LeakyReLU

LeakyReLU改善DyingReLU问题是从最根本的方法入手的——为小于0的死区增加了微小的梯度以避免陷入停止更新。

有效性分析：事实上，在本问题中，leakyReLU并不能改善问题。如果使用random_uniform（-1,1），那么梯度依然为全零（溢出？），如果使用默认的配置的random_uniform，则虽然没有0梯度，但是更新很慢且几乎陷入不更新（迭代6倍于原本的次数依然无效）。**虽然看起来没有神经元die，但是事实上网络中的问题并未得到改善。**


#### [Resnet](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
文章描述：
残差网络是一种通过添加恒等块，即他们利用将前面层的输出x，以及上一层的输出结果f(x),进行简单的加或者减作为新一层的输入值H(x)=f(x)+(-)x,这样可能使得本来可能出现的DyingReLU问题的概率减小。分析原因：**当前层没有发生Dying ReLU现象则大部分输出为正,则对后面层加上该层的输出值Dying ReLU的概率小于原概率。**

实验结果：在我们的实验中，利用残差网络的**收敛结果要好于其他改善方案**，可能是一个**较为复杂但是最优**的方案。

有效性分析：
Resnet的[paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)中并未提及其关于DyingReLU问题的改善效率，只是说明了该网络在执行识别和训练任务时具备较高的效率。paper中提及了一个情况：在Resnet与普通网络对比训练效果时，*普通网络在未发生梯度消失问题的情况下，依然出现了较高的错误率与较差的训练效果，而Resnet并没有出现这个问题，似乎这个可能是DyingReLU的情况* 。
Resnet通过残差块的设计，使得即使发生了DyingReLU也会被传入H(x)=f(x)+(-)x中x的正值覆盖。因此对于每个残差块的输出，层的含零量不会超过残差块接受的输入层的含零量。对于**网络初始化未出现Dying ReLU的情况**，这个方案可以减小到Dying ReLU的概率。。**reset本身作为一个训练效果优秀的网络，可以作为一个备用选项，在其他改善方案效果不好时提供给用户。**


#### [优化器问题（与学习率相关）](https://arxiv.org/abs/2005.06195)

文章认为，DyingReLU现象可以依靠SGD优化器的动量设计来进行缓解，并在文章结尾给出了在单层ReLU模型上的SGD与Adam优化器DyingReLU的比例随训练次数变化的曲线，在一定程度上说明了对于DyingReLU问题，优化器之间的差异。

上文提到，造成DyingReLU的原因不仅仅在于负向的bias，学习率过大也会一定程度上引发这类问题，过大的学习率可能导致过大的权重改变量从而导致训练中梯度意外进入死区。

有效性分析：现有的实验并未展示出改变优化器所带来的明显效果以及搜索到的改善方案并未提及改变优化器（SGD、Adam等）有效性以及具体原因。可以考虑在其他改善方案的情况下替换优化器，来进行进一步测试（仅作考虑选项）。


#### [Batch_Normalization](https://en.wikipedia.org/wiki/Batch_normalization)

文章描述：
归一化处理-Batch_normalization，论文(https://arxiv.org/abs/1502.03167)首次应用到神经网络的学习中，它是在原来的每个隐含层的训练过程中添加新的归一化层，将上一层的输出重新规范到均值为0方差为1的数据，在一定程度上可以对DyingReLU进行有限的缓解。

实验结果：
在我们的实验中，BN虽然能缓解Dying ReLU问题，但是最终的**收敛速度较慢、收敛情况较差**。

有效性分析：
可以作为一个备选方案，但是由于其表现效果并不理想，不作为缓解Dying ReLU问题的主要方案。

#### 总结

> DyingReLU的潜在改善方案中，对原始网络修改由高到低Resnet>SeLU>initializer>leakyReLU>优化器改变；测试中有效性从高到低为：SeLU>Initializer>LeakyReLU>优化器改变（Resnet当前缺乏实验）