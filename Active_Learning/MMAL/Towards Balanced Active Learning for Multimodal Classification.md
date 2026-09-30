## 南洋理工大学、NVIDIA AI CENTER等
## ACM MM 2023

## Codebase:https://github.com/MengShen0709/bmmal

## 解决了什么问题？

多模态Active Learning中的模态不平衡问题
[[多模态学习不平衡现象]]


然后这个文章我稍微看了一下，实验结果一般，提高不是很多，然后故事讲的有点花，导致我有点没懂，也不知道要不要Follow，但是还是总结一下吧......

#### 还是简单说一下这个问题的推导：
多模态训练的时候，我们一般是这样构造loss的
$$\mathcal{L}_{final}=\frac{1}{3}[\mathcal{L}_{CE}(f_{m_1},y)+\mathcal{L}_{CE}(f_{m_2},y)+\mathcal{L}_{CE}(f_{mm},y)]$$
那么这两个模态肯定是各有一个gradien embedding的，分别面向各自的最后一层线性层的权重：
$$g_{m_1,i}=\frac{\partial\ell}{\partial(W_i)_{m_1}}=\left(p_i^{mm}-\mathbf{1}_{\hat{y}=i}\right)z_{m_1},g_{m_2,i}=\frac{\partial\ell}{\partial(W_i)_{m_2}}=\left(p_i^{mm}-\mathbf{1}_{\hat{y}=i}\right)z_{m_2}.$$
但是这个就有问题了，从这一步就有问题了，文章直接把多模态分类的loss做梯度，单模态的loss不算梯度，这个地方就有点扯了，主要还是为讲故事服务了。
### 传统方法
主要还是以[[Badge|BADGE]]为主的单模态Active Learning算法，这个工作属于是第一个做多模态AL的。

#### 解决方法和动机
