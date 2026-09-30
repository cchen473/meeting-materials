## 内蒙古大学、国科大、鹏城实验室等
## TPAMI 2025
## Codebase: https://github.com/openimmc/ascmr (代码不完整)

## 解决了什么问题？
1. Supervised Cross-Model Retrieval(有监督跨模态检索)中的标注问题：
我们比较容易获取配对的图文对，但是如果人工标注样本的标签可能需要较大成本，所以使用Active Learning去降本增效。
    1. 先讲一下有监督跨模态检索任务：
有一个数据集：$\mathbb{D}=\{(\boldsymbol{x}_i^\upsilon,\boldsymbol{x}_i^t);\boldsymbol{y}_i\}_{i\boldsymbol{=}1}^N$  我们需要拿这个数据集去训练一个特征提取模型，我们输入这个文本，要能检索到和他对应的那个图像，那么很有可能不是top1检索出来的那个，可能是top5，那么这个其实也是能接受的，最后的度量主要看的是mAP。
    2. 再讲一下这个模型一般是怎么训的：
我们把图像和文本编码成两个表示：$\boldsymbol{v}_i=f_v(\boldsymbol{x}_i^v;\theta_v^f)$和$\boldsymbol{t}_i=f_t(\boldsymbol{x}_i^t;\theta_t^f)$
第一个loss：
$$\begin{aligned}\mathcal{L}_{\mathrm{con}}&=\mathcal{L}_{\mathrm{info}}\left(\boldsymbol{v}_i,\boldsymbol{v}_j\right)+\mathcal{L}_{\mathrm{info}}\left(\boldsymbol{t}_i,\boldsymbol{t}_j\right)\\&+\mathcal{L}_\mathrm{info}\left(\boldsymbol{v}_i,\boldsymbol{t}_j\right)+\mathcal{L}_\mathrm{info}\left(\boldsymbol{t}_i,\boldsymbol{v}_j\right)\end{aligned}$$
然后：$\mathcal{L}_{\mathrm{info}}=-\log\frac{\exp(\cos(\boldsymbol{x}_n,\boldsymbol{y}_n)/\tau)}{\sum_{m\neq n}\exp(\cos(\boldsymbol{x}_n,\boldsymbol{y}_m)/\tau)}$
也就是去align正样本的表征，推远负样本的表征，直观的做法
第二个loss：
$$\mathcal{L}_{\mathrm{cls}}=\frac{1}{n}\sum_{i=1}^n\|\hat{\boldsymbol{y}}_i^v-\boldsymbol{y}_i\|_2+\frac{1}{n}\sum_{i=1}^n\|\hat{\boldsymbol{y}}_i^t-\boldsymbol{y}_i\|_2$$
也就是在两个模态学出来的特征上做一个分类，然后用标注的标签去监督这个分类损失，注意是平方损失而不是CE损失。
2. 那么如果说直接把Active Learning的方法迁移过来又有什么问题呢？
![[CMRp1.png]]
可以看上面这张图，对于上面这个样本的这个图像，在图像的特征空间中，他和Teddy bear、Shoe、Dog的特征相似度都不太高，但是在共享的特征空间上，图文的相似度是近的，说明这个样本的跨模态关系我们已经很好的master到了，但是还做不好分类，那么这种uncertainty，我们的现有AL算法是能捕捉到的；
但是下面这张图（注意这个Sheets应该是Sheeps），在图像的特征空间中，这个图像和一些主要的类别都能match上，但是在共享特征空间上，他并不能和他对应的文本学到很好的跨模态匹配关系，我们现在也没有一个很好的AL算法去量化“跨模态不确定性”这个目标。
所以说我们要提出一个选样本方法去量化以上两个指标。
## Methods
![[CMRp2.png]]
这个图意思是我们除了要训一个Task Model，还要去训一个选样模型，也就是这个Unbiased Data Selection Algorithm，去满足我上面的要求，那么这个算法呢分两个module，也就是基于概率的多模态信息度量和密度感知的选样。
### Probabilistic Multi-Modal Informativeness Measurement
##### 为什么叫做Probabilistic Measurement？
也就是说这个表征是自带不确定性的，给他建模成一个多维逐维高斯分布，也就是以下形式：
$$d_{v_i}\sim\mathcal{N}(\boldsymbol{\mu}_v,\boldsymbol{\Sigma}_v)=\frac{1}{\sqrt{2\pi\boldsymbol{\Sigma}_v}}e^{-\frac{(\boldsymbol{v}_i-\boldsymbol{\mu}_v)^2}{2\boldsymbol{\Sigma}_v}}d_{t_i}\sim\mathcal{N}(\boldsymbol{\mu}_t,\boldsymbol{\Sigma}_t)=\frac{1}{\sqrt{2\pi\boldsymbol{\Sigma}_t}}e^{-\frac{(\boldsymbol{t}_i-\boldsymbol{\mu}_t)^2}{2\boldsymbol{\Sigma}_t}}$$
##### 那么我们怎么生成这样两个表示呢？
1. 对于均值，我们这样做：
$$\boldsymbol{\mu}_v^i=\mathrm{LN}(\boldsymbol{v}_i+\sigma\left(\mathrm{Attn}(\boldsymbol{v}_i)\right)\boldsymbol{\mu}_t^i=\mathrm{LN}(\boldsymbol{t}_i+\sigma\left(\mathrm{Attn}(\boldsymbol{t}_i)\right)$$
这个self- attention怎么做？
一个batch的所有表示进入到self attention，然后做attention出那个自己这个token的新的表示，这个表示做一个逐维的sigmoid，然后加上还没进attention的原始表示，也就是将这个单样本的表示，与其他各个样本的表示做一个交互和修正。
2. 对于方差，我们这样做：
$$\boldsymbol{\Sigma}_v^i=\mathrm{ReLU}\left(\boldsymbol{v}_i+\mathrm{Attn}(\boldsymbol{v}_i)\right)\boldsymbol{\Sigma}_t^i=\mathrm{ReLU}\left(\boldsymbol{t}_i+\mathrm{Attn}(\boldsymbol{t}_i)\right)$$
这个和上面很像，最后的这个方差要做一个ReLU。
##### 那么如何建模上面说的这两个指标？
对于单模态不确定性，方差大小就可以建模，那么对于跨模态对应不确定关系，我们引入一个度量方式(Bhattacharyya correlation coefficient)：
$$
p(m|d_{v_i}, d_{t_j}) = \exp\left(
-\frac{1}{8}\left(\boldsymbol{\mu}_v^i - \boldsymbol{\mu}_t^j\right)^\top \boldsymbol{\Sigma}^{-1} \left(\boldsymbol{\mu}_v^i - \boldsymbol{\mu}_t^j\right)
-\frac{1}{2}\ln\left(
\frac{|\boldsymbol{\Sigma}|}{\sqrt{|\boldsymbol{\Sigma}_v^i| \, |\boldsymbol{\Sigma}_t^j|}}
\right)
\right)$$
这个公式不需要懂，那么这个式子其实就是一个衡量两个分布之间的距离的工具，不认为他和JS、KL散度有什么区别。
##### 我们接下来要学到这个attention的一些参数对吧，比如说qkv矩阵等等
所以这样做：
首先就是对比学习，配对的这个分布相似度要高，非配对的要低：
$$\mathcal{L}_{\mathrm{CL}}=-\mathbb{I}_{ij}\log p\left(m|d_{v_i},d_{t_j}\right)-\left(1-\mathbb{I}_{ij}\right)\log\left(1-p\left(m|d_{v_i},d_{t_j}\right)\right)$$
其次就是正则化项，防止这个方差收敛为0:
$$\mathcal{L}_{\mathrm{KL}}=\mathrm{KL}(d_{v_i},\mathcal{N}(0,1))+\mathrm{KL}(d_{t_i},\mathcal{N}(0,1))$$

那么最后的loss就是二者相加：
$$\mathcal{L}_{\mathrm{overall}}=\mathcal{L}_{\mathrm{CL}}+\mathcal{L}_{\mathrm{KL}}$$
最后这个概率的选样原则就是这样：
$$\mathrm{I}_{(x_i^v,x_i^t)}=\alpha\left(\left\|\Sigma_v^i\right\|_2+\left\|\Sigma_t^i\right\|_2\right)+\left(1-p\left(m|d_{v_i},d_{t_i}\right)\right)$$
这个没什么可说的，如果单模态不确定，或者跨模态不确定，我们都有更大概率去选他。
### Density-Aware Budget Allocation
现在我们需要进行密度感知以进行无偏的挑选，不然就会出现这种类别不平衡的情况：
![[biasCMR.png]]
首先这个密度感知的这个事情是在融合空间上做的，所以先融合：$\boldsymbol{\mu}_z^i=\boldsymbol{\mu}_v^i+\boldsymbol{\mu}_t^i$ , $\boldsymbol{\Sigma}_z^i=\boldsymbol{\Sigma}_v^i+\boldsymbol{\Sigma}_t^i.$ 
之后就要做密度感知了，那么一个Batch内，密度这样定义：
$$\begin{aligned}A_{\mathcal{S}}=\frac{1}{N^s\cdot(N^s-1)}\sum_{i=1}^{N^s}\sum_{j\neq i}^{N^s}p\left(m|d_{z_i},d_{z_j}\right)\end{aligned}$$
$\boldsymbol{N}^{s}$是Batch的样本个数
然后我们选样是要这样的：
$$\begin{aligned}\max_{q\in\Delta_b}&\mathbb{E}_{\boldsymbol{S}\sim q}\left[\sum_{i\in\boldsymbol{S}}\mathbb{I}_i-A_{\boldsymbol{S}}+\lambda\right],\\&\mathrm{where~}\Delta_b=\left\{q\in[0,1]^{|N_u|}:\sum_{i\in|N_u|}q_i=b\right\}\end{aligned}$$
就是要去最大化这个文章的信息量-密度，但是这个$\lambda$其实没用，文章在胡讲，代码里面也没有这个的体现，所以这里先存疑吧，做法也比较存疑，因为第二个这个密度项应该是有了batch再算的，但是有了batch再优化这个吗，那搜索数量是不是太大了？（伪代码里面也没有写）
## Experiments
![[CMREXP.png]]
依旧SOTA，其中注意，标注对这个任务影响不大，因为不影响图文之间align的loss，只是标注loss的影响，但是这个还是效果很好。
![[CMRMAP.png]]截取的一些表，不同数据集的MAP，效果不错
![[CMRchange.png]]不管是什么CMR的方法，这个都即插即用，work的不错
![[CMRabl.png]]消融，nothing to say
## Takeaway
1. 一种量化跨模态关系的方法，我觉得可以用到分类问题中
2. 概率特征建模去量化不确定性的方法