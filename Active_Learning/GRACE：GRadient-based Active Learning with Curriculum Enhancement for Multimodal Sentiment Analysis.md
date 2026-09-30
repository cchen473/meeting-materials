## ACM MM 2024
## 中科大团队
## codebase: 未开源
## 解决了什么问题？
#### 1. 多模态情感分析(Multimodal Sentiment Analysis)
这个问题没什么特殊的setting，是一个分类问题，和多模态分类基本一致
#### 2. 主动学习(Active Learning)的课程学习问题
说的就是主动学习的效果很大程度上取决于你的Task Model，也就是说你这个模型能力越强，他的选样指导就越有参考价值，所以你前期的模型性能很重要，怎么让模型在前期快速收敛是一个重要的课题，所以说如果模型前期去学一些难样本、冲突样本的话，问题就是这两个意见会指导模型往不同的方向收敛，导致model change很小，反而不利于选样任务执行。
## Method

![[GRACE.png]]

#### Training
和普通MM分类没区别，分别encode之后拼接分类
#### 梯度获取
![[grace_gradient.png]]
每个Unlabeled的的数据编码出特征之后会进入到分类层，那么其实我们有吧四个线性层去做分类，单模态一人一个，多模态融合还有一个，那么也会有四个预测的y，那么我们通过类似于[[Badge|BADGE (Diverse, Uncertain Gradient Lower Bounds)]]的手段去生成gradient embedding，每一个模态其实会有两个。即自己的预测反向传播的梯度有一个，融合的反向传播的梯度也有一个，所以我们会在这两个gradient embedding上面进行一些操作。

#### Querying
那么对于每一个模态，我们都获得了一个gradient embedding，接下来我们算三个度量值：
1. Easiness-衡量该样本的难易程度
怎么衡量，可以看framework蓝色的那部分，也就是自己单模态的gradient embedding，和多模态传过来的那个gradient embedding的cosine similarity，如果越相似，那么说明这个样本越好学习，因为优化方向很明显。
$$\hat{s}_e(x^p)=\sum_{k\in K}Sim(g_k(x^p),g_{m_k}(x^p))$$
2. Informativeness-衡量该样本的信息量
$$\hat{s}_i(x^p)=\sum_{k\in K}\|\mathbf{g}_k(x^p)+\mathbf{g}_{m_k}(x^p)\|_2$$
这个样本的信息量，就是每一个模态和融合模态的gradient embedding，矢量相加之后算一个长度，这三个模态的长度加到一起，示意可以看framework图第二个图。这个长度越长，信息量越大，在[[Badge|BADGE (Diverse, Uncertain Gradient Lower Bounds)]]中有详细推证
3. Representativeness-衡量这个样本的代表性
$$\hat{s}_r(x^p)=\sum_{k\in K}\sum_{x^q\in U\setminus x^p}Dist(g_k(x^p)+g_{m_k}(x^p),g_k(x^q)+g_{m_k}(x^q))$$
对于每一个模态，算一个他两个gradient embedding的矢量和，那么Unlabeled的样本里面去掉这个样本，其他样本与这个样本的距离之和越大，说明它越具有代表性(*这个其实略有争议)，越应该选它。
4. 最后的聚合-融合这三个分数得到一个最终的分数
$$s(x^p)=s_e(x^p)\cdot a+s_i(x^p)\cdots_r(x^p)$$
其中，$\alpha$是一个线性退火的设计：
$$\alpha_t=\alpha_{init}-(t-1)\cdot\alpha_d$$
## Experiment

![[grace_exp.png]]