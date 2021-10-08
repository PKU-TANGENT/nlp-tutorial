# PKU-TANGENT nlp-tutorial

本教程供新加入 TANGENT 实验室的同学入门 NLP 使用

- [PKU-TANGENT nlp-tutorial](#pku-tangent-nlp-tutorial)
  - [写在前面](#写在前面)
  - [基础知识](#基础知识)
    - [机器学习](#机器学习)
    - [深度学习](#深度学习)
    - [自然语言处理](#自然语言处理)
  - [文献阅读](#文献阅读)
    - [Basic](#basic)
    - [前沿进展](#前沿进展)
    - [工具](#工具)
  - [动手实践](#动手实践)
    - [任务一：基于深度学习的文本分类](#任务一基于深度学习的文本分类)
    - [任务二：基于 LSTM-CRF 的命名实体识别](#任务二基于-lstm-crf-的命名实体识别)
    - [任务三：新闻标题生成](#任务三新闻标题生成)
    - [任务四：Transformer](#任务四transformer)
  - [本仓库的使用说明](#本仓库的使用说明)

## 写在前面

（非劝退hhhh）在你看以下内容之前，请确保你有：

1. 绝对优秀的信息检索能力（大学生最最基本的能力）
2. 优秀的英语阅读水平（你阅读的文献基本都将是英文的）
3. 良好的编程能力（限 Python，如果会一两种深度学习框架最好）以及良好的代码规范([python](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/)、[c++](https://zh-google-styleguide.readthedocs.io/en/latest/google-cpp-styleguide/contents/))
4. 数学基础（高等数学、线性代数（主要是矩阵运算）、概率论与统计）

如需 GPU，请联系实验室服务器管理员


## 基础知识

### 机器学习

深度学习是机器学习的子集，目前 NLP 领域再用传统机器学习的方法就会非常土气，但是机器学习的基础概念仍然是相通的。如果你从未接触过机器学习，那么以下学习资料请自行选择安排学习，不必追求大而全，重在了解机器学习基本概念和传统机器学习算法的思想

网课：吴恩达 机器学习公开课；李宏毅 机器学习

书：机器学习（西瓜书），统计学习方法（李航）

### 深度学习

书：Deep Learning（GoodFellow, Bengio, Courville），神经网络与深度学习（邱锡鹏）

教程：Pytorch Tutorials（建议动手做一做与 nlp 相关的几个 tutorial，弄懂每一行代码）

### 自然语言处理

网课：stanford cs224n；cmu cs 11-747

书：统计自然语言处理（宗成庆），现代自然语言生成（黄民烈），自然语言处理：基于预训练模型的方法（车万翔）


## 文献阅读

### Basic

我们主要阅读国际会议论文，相关的会议有：

- 自然语言处理相关会议：ACL, EMNLP, NAACL
- ML 理论：ICML, NeurIPS, ICLR
- 偏应用：AAAI（读作 triple AI，不读 A A A I，不累吗？）, IJCAI

其中，ACL 系会议提供 anthology (https://aclweb.org/anthology/)

### 前沿进展

如果想了解某一个领域的前沿进展，通常会关注 arXiv（预印本），arXiv 在工作日每日更新，便于及时追踪前沿动态（https://arxiv.org/list/cs.CL/recent）

### 工具

经典论文一般都会有中文读后感，可以辅助阅读

文献分类整理是一个好习惯，建议根据个人喜好选择诸如 Zotero, Endnote, Mendeley, Papers 等文献管理软件

初学时做好论文笔记，可以使用 Markdown，也可以使用 Notion, OneNote 等笔记软件





## 动手实践

### 任务一：基于深度学习的文本分类

文本分类是入门 NLP 的一个好的开始，同时 NLU（自然语言理解）任务本质上来说都可以归类为文本分类。请使用 CNN 或 RNN 完成 Kaggle 上一个简单的文本分类任务。

任务描述 & 数据集：https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/

参考文献：
Convolutional Neural Networks for Sentence Classification (https://aclanthology.org/D14-1181/)
Recurrent Convolutional Neural Networks for Text Classification (https://www.deeplearningitalia.com/wp-content/uploads/2018/03/Recurrent-Convolutional-Neural-Networks-for-Text-Classification.pdf)


### 任务二：基于 LSTM-CRF 的命名实体识别

在 NLP 中，结构预测（Structured Prediction）是指输出空间为结构化对象的一类任务，包括命名实体识别、关系抽取、共指消解等子任务，命名实体识别又属于序列标注问题。请实现简单的基于 LSTM-CRF 的命名实体识别

任务描述 & 数据集：https://www.clips.uantwerpen.be/conll2003/ner/

参考文献：
Neural Architectures for Named Entity Recognition (https://arxiv.org/pdf/1603.01360.pdf)

建议：循序渐进，先实现 LSTM NER 模型，再在其基础上加上 CRF 层。


### 任务三：新闻标题生成

摘要和翻译是文本生成中比较主流的两大任务，在这里我们选取一个简单的新闻标题生成任务作为入门项目。

数据地址： http://www.sogou.com/labs/resource/cs.php   完整版(648M)

参考文献：Generating News Headlines with Recurrent Neural Networks （https://arxiv.org/abs/1512.01712）

可以先基于RNN实现上述模型，在算力允许的情况下再尝试预训练模型。



### 任务四：Transformer

请结合 Attention Is All You Need 原论文，读懂 The Annotated Transformer（http://nlp.seas.harvard.edu/2018/04/03/attention.html）



## 本仓库的使用说明

1. 有问题就提在issues里面，同理你也可以在issues里面检索是否已经有你遇到的问题；
2. main分支无法直接修改，所有修改均需要通过提交`Pull requests`来实现，必须选择至少一个reviewer，推荐选择大师兄`Yifan-Song793`来review；
3. git commit的规范看[这里](https://juejin.cn/post/6844903793033756680)，禁止使用意义不明的test、add等语句。
