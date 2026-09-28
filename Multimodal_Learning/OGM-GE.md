## CVPR 2022
## 人大高瓴、上海AI Lab等

## Codebase: https://github.com/GeWu-Lab/OGM-GE_CVPR2022

## 解决了什么问题？
一句话来说，解决了[多模态学习不平衡现象](../Active_Learning/MMAL/%E5%A4%9A%E6%A8%A1%E6%80%81%E5%AD%A6%E4%B9%A0%E4%B8%8D%E5%B9%B3%E8%A1%A1%E7%8E%B0%E8%B1%A1.md)，那么这个问题由理论和实验两个Observation得出：
#### 实验方面：
![OGMober.png](Figs/OGMober.png)
这张图展现了多模态学习当中的一些问题，如果我们只选用单模态数据进行学习的话，我们这个模型的效果是要比我们双模态联合训练时分类的效果好的，这说明了每一个模态的信息并没有被充分挖掘，这边也可以看出来作者的方法缓解了这一问题。

#### 理论方面：
这个也是多模态平衡学习中一个问题，就是强模态会主导优化信号，如果强模态优化过程完毕，分类结果差不多了，弱模态就得不到很好的优化，推证过程看这个链接：
[多模态学习不平衡现象](../Active_Learning/MMAL/%E5%A4%9A%E6%A8%A1%E6%80%81%E5%AD%A6%E4%B9%A0%E4%B8%8D%E5%B9%B3%E8%A1%A1%E7%8E%B0%E8%B1%A1.md)
## Method
![ogm-ge.png](Figs/ogm-ge.png)