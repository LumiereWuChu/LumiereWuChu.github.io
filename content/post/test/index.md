---
title: "Lumiere Wuchu Test blog"
description: hello world
date: 2025-05-02T05:54:05Z
image: 
math: 
license: 
hidden: false
comments: true
draft: true
---

没有忽略不重要信息的注意力机制就是分心机制
![[Pasted image 20241211171020.png]]
理解Attention模型的关键就是这里，即由**固定的中间语义表示C换成了根据当前输出单词来调整成加入注意力模型的变化的Ci**。增加了注意力模型的Encoder-Decoder框架理解起来如下图所示：有必要从encoder-decoder框架中理解attention机制的意义
![[Pasted image 20241211171207.png]]
具体的一个典型的Attention思想包括三部分：Q`query`、K`key`、V`value`。
- Q是query，是输入的信息；key和value成组出现，通常是原始文本等已有的信息；
- 通过计算Q与K之间的相关性a，得出不同的K对输出的重要程度；
- 再与对应的v进行相乘求和，就得到了Q的输出；
![[Pasted image 20241211171551.png]]
- **step1**，计算Q对每个K的相关性`相似性`，即函数F(Q,K)

- 这里计算相关性的方式有很多种，常见方法比如有：
- 求两者的【向量点积】，similarity(Q, Ki) =Q Ki  
- 求两者的向量【余弦相似度】，similarity(Q, Ki) = Q Ki /||Q|| ||Ki||
- 引入一个额外的神经网络来求值，similarity(Q, Ki) =MLP(Q,Ki)。
    
- **step2**，对step1的注意力的分进行归一化；
- softmax的好处首先可以将原始计算分值整理成所有元素权重之和为1的概率分布；
- 其次是可以通过softmax的内在机制更加突出重要元素的权重；
- 即为value_i对应的权重系数;
    
- **step3**，根据权重系数对V进行加权求和，即可求出针对Query的Attention数值。
    $$ Attention (Query,Source) = \sum_{i=1}^{Lx} a_i · Value_i$$

> 值得强调的一点是：K和V等价，它俩是一个东西。
## Self-Attention
self-attention，顾名思义它只**关注输入序列元素之间的关系，即每个输入元素都有它自己的Q、K、V**，比如在一般任务的Encoder-Decoder框架中，输入Source和输出Target内容是不一样的，比如对于英-中机器翻译来说，Source是英文句子，Target是对应的翻译出的中文句子，Attention机制发生在Target的元素Query和Source中的所有元素之间。而Self Attention指的不是Target和Source之间的Attention机制，而是**Source内部元素之间或者Target内部元素之间发生的Attention机制**，也可以理解为Target=Source这种特殊情况下的注意力计算机制。
![[Pasted image 20241211173829.png]]
在self-attention中，每个单词有3个不同的向量，即Q、K、V。它们是通过X乘以三个不同的权值矩阵、、得到的，其中三个矩阵的尺寸也是相同的，这三个矩阵也是需要学习的。

> 可以理解为：**self-Attention中的Q是对自身（self）输入的变换，而在传统的Attention中，Q来自于外部。**
那么整个self-attention的计算过程可以如下：

- 1.首先就是基本的embedding将输入单词转为词向量；
    
- 2.根据嵌入向量利用矩阵乘法得到q、k、v三个向量；
    
- 3.为每一个向量计算一个相关性score：；
    
- 4.为了梯度的稳定，除以根号dk，下面会给出推导；
    
- 5.进行softmax归一化得到权重系数；
    
- 6.与value点乘得到加权的每个输入向量的评分v；
    
- 7.相加之后得到最终的输出结果；
![[Pasted image 20241211173931.png]]
![[Pasted image 20241211182103.png]]
三个线性变化矩阵
经过缩放点积之后
![[Pasted image 20241213122223.png]]
“我”这个字对“我想吃酸菜鱼”这句话里面每个字的注意力权重，和V中“我想吃酸菜鱼”里面每个字的第一维特征进行**相乘再求和，这个过程其实就相当于用每个字的权重对每个字的特征进行加权求和**
>注意力机制是没有位置信息的，所以需要引入位置编码，下一篇transformer中会讲解。


    class BertSelfAttention(nn.Module):  
       def __init__(self, config):  
         self.w_q = nn.Linear(config.hidden_size, self.all_head_size) # 输入768， 输出768  
        self.w_k = nn.Linear(config.hidden_size, self.all_head_size) # 输入768， 输出768  
        self.w_v = nn.Linear(config.hidden_size, self.all_head_size) # 输入768， 输出768  
      def forward(self,hidden_states): # hidden_states 维度是（L, 768）  
        Q = self.query(hidden_states)  
        K = self.key(hidden_states)  
        V = self.value(hidden_states)  
          
        attention_scores = torch.matmul(Q, K.transpose(-1, -2))  
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  
  
        out = torch.matmul(attention_probs, V)  
        return out
      
   ![[Pasted image 20241213143408.png]]
 
接下来用得到的权重给备胎加权，渣男就知道该对谁付出多少的注意力了。当然也会有理想型是自己的情况，即渣男最需要关注的是自己。
复习下流程：

- 输入X和三个矩阵相乘，分别得到三个矩阵Q、K、V，Q是我们正要查询的信息，K是正在被查询的信息，V是被查询到的内容；
    
- 我们用Q和K的转置的点乘得到这两条信息的相似程度，再除以根号dk，使训练时的梯度保存稳定；
    
- 经过softmax得到权重矩阵，用这个权重矩阵和内容V进行加权，也就是相乘，这就是self-attention的原理。

## Multi-Head Attention
![[Pasted image 20241213144345.png]]
