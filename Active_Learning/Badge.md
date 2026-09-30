
## ICLR 2020 
## Jordan T. Ash(Princeton University)等
## Code base: https://github.com/JordanAsh/badge


## 解决了什么问题？

**Active Learning**就是在每一轮的Label Budget下选择性标注一些样本进行训练，重点是选择样本的策略。

传统Active Learning过度依赖于Batch Size、模型架构和超参，不直观而且工作量大；此外只关注**不确定性**和**多样性**中的一个，本文旨在同时面向不确定性和多样性，并减少调参工作。

然后讲一下为什么传统方法依赖Batch Size好吧，就是如果我们只去关注Uncertainty的话，这个其实很简单对吧，只需要把每个Unlabel Pool中的样本去前向一次，算出来预测概率之后找置信度低的（也就是预测的最大概率小的）样本去query一下不就行了吗，但是这个问题在于，你有可能一个batch中全部都是这个梯度下降方向的样本，第一个样本还有点价值，后面的样本的价值就降低了，所以参数调整的方向应该是不同的，这就需要多样性来进行调整了。

## 解决思路

比较核心的claim其实就是，如果一个样本能够带来大的参数调整，那么这个样本我们认为他是有价值的，这个样本也代表着高不确定性，所以文章也提出了gradient embedding这一个量来度量参数调整的剧烈程度和调整方向。

#### 如何获取gradient embedding（什么是gradient embedding）

我们考虑一个分类网络$f(x;\theta)=\sigma(W\cdot z(x;V))$ ，输入一个样本，经过参数为$V$的神经网络得到特征$z$，$z$的维度是自定义的$d_z$然后经过一个线性投影层$W$，维度为$d_c*d_z$投影到类别的维度上(这里不加偏置量$b$，加上也不影响结果)，最后经过一个*Softmax*层映射到概率维度，也就是：

***Note:*** 后面的文章是把这个线性层权重拆成一些向量来的，确实直观一点: $W=(W_1,\ldots,W_K)^\top\in\mathbb{R}^{K\times d}$,那么这个每一个$W_i$跟特征$z$作积就是这个样本在$i$类上面的预测概率
$$\sigma(z)_i=e^{z_i}/\sum_{j=1}^Ke^{z_j}$$
那么我们既然要做分类，我们就要做一次CE Loss的计算：
$$\ell_{\mathrm{CE}}(f(x;\theta),y)=\ln\left(\sum_{j=1}^Ke^{W_j\cdot z(x;V)}\right)-W_y\cdot z(x;V).$$
上面都没什么可说的，下面我们求gradient embedding，其实也就是CE Loss对最后一层线性投影层$W$求个偏导就行。和上面一样，也是把这个gradient embedding拆成向量来计算的：
$$(g_x)_i=\frac{\partial}{\partial W_i}\ell_\mathrm{CE}(f(x;\theta),\hat{y})=(p_i-I(\hat{y}=i))z(x;V)$$
好，这个就是gradient embedding了，这个里面有一个$\hat{y}$，这个东西其实就是模型对于样本$x$，他认为最有可能的类，也就是预测概率最大的（因为当时算这个的时候是没有ground truth的$y$的）

#### 怎么筛选样本
算法流程图：
![[Badge_algorithm.png]]

中文解释一下：
1. 算法的输入是：神经网络、一个待筛选的样本池、M个预先随机好的样本、当前的标注轮数T和一个batch内的样本数B
2. 初始化标注的样本集S
3. 对神经网络参数初始化
4. 对于每一个轮次：
    1. 对于每一个unlabel样本:
        1. 找到他预测概率最大的标签$\hat{y}$
        2. 根据这个$\hat{y}$算一下gradient embedding $g_x$
    2. 对这个gradient embedding进行一个*k*-MEANS++，选出对应数量的样本，然后标注
    3. 然后把这些样本并入标注集
    4. 训练下一轮模型
5. 循环回去到轮次结束
6. 模型训练结束

#### 下面说一下这个为什么这么做(Motivation)

1. 使用gradient embedding度量不确定性：*Since deep neural networks are optimized using gradient-based methods, we capture uncertainty about an example through the lens of gradients. In particular, we consider the model uncertain about an example if knowing the label induces a large gradient of the loss with respect to the model parameters and hence a large update to the model.（原文）* 意思就是这个越大，说明不确定性越大，比较直观。
2. 使用$\hat{y}$来算gradient embedding: 我们是没有ground turth标签的，所以算不了真正的gradient embedding，只能使用$\hat{y}$，这是现实原因，下面讲讲为什么可以这么做（理论原因）：
因为这个用$\hat{y}$算出来的东西其实是真正的gradient embedding L2 范数的***Lower Bound***，证明：
$$\|g_x^y\|^2=\left(\sum_{i=1}^Kp_i^2+1-2p_y\right)\|z(x;V)\|^2$$
这个是对gradient embedding L2 范数，我们要让他最大，其实就可以最大化他的***Lower Bound***，这个是常见的做法。注意到当$p_y$最大的时候，这个数值取到最小，那么$\hat{y}=\operatorname{argmin}_{y\in[K]}\left\|g_x^y\right\|$
也是容易得出的。
所以证明了我们做法的合理性。
3. 在gradient embedding上面做*k*-MEANS++: 如果一个batch你只选gradient embedding长的，势必会引来问题，就是指导信号是一致的，比如一个是(10,5,0)，另一个是(2,1,0)，这方向是一致的，你在第一个上面做个梯度下降，这个参数被优化的差不多了，另一个也就较为确定了，相当于白选了。所以说另外一个选样本的指标，多样性(Diversity)很重要，要对这个做个*k*-MEANS++。
4. 使用*k*-MEANS++而不是*k*-DPP: 先讲一下*k*-DPP算法，就是说对每一个gradient embedding都与其他的gradient embedding算一次Gram行列式，这个行列式越大，被选中的概率也越大，但是这种方法时间复杂度很高，如图所示：
![[sample.png]]
其实这两个方法在sample能力上差不多，但是*k*-DPP时间复杂度明显高。

#### 可能遇到的问题（这个方法在哪些环境下可能不work）
这个坑回来补

#### 我想提出的方法
这个坑回来补