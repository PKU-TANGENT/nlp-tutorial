# PKU-TANGENT nlp-tutorial

本教程供新加入 TANGENT 实验室的同学入门 NLP 使用

- [PKU-TANGENT nlp-tutorial](#pku-tangent-nlp-tutorial)
  - [写在前面](#写在前面)
  - [基础知识](#基础知识)
    - [机器学习](#机器学习)
    - [深度学习](#深度学习)
    - [自然语言处理](#自然语言处理)
  - [文献阅读](#文献阅读)
    - [Google Scholar](#google-scholar)
    - [会议论文](#会议论文)
    - [前沿进展](#前沿进展)
    - [工具](#工具)
    - [开源代码](#开源代码)
  - [动手实践](#动手实践)
    - [写在前面](#写在前面-1)
    - [任务一：基于深度学习的文本分类](#任务一基于深度学习的文本分类)
    - [任务二：基于 LSTM-CRF 的命名实体识别](#任务二基于-lstm-crf-的命名实体识别)
    - [任务三：Neural Machine Translation (NMT)](#任务三neural-machine-translation-nmt)
    - [任务四：Transformer & PLM](#任务四transformer--plm)
      - [基础知识](#基础知识-1)
      - [Huggingface Transformers](#huggingface-transformers)
      - [Huggingface Ecosystem](#huggingface-ecosystem)
      - [基于Huggingface Trainer的分类任务](#基于huggingface-trainer的分类任务)
  - [本仓库的使用说明](#本仓库的使用说明)

## 写在前面

相信大家经过几年的学习，已经拥有了以下的技能：
1. 优秀的信息检索能力，无论是在论文阅读、写代码、使用服务器、写论文等过程中都有可能遇到各种各样的问题，在询问他人之前，请善用搜索
2. 优秀的英文阅读能力和基本的英语写作能力
3. 良好的编程能力，在 NLP 相关研究中，我们通常会使用 Python，如果你之前只学过 C 或者 C++，那么入门 Python 对于你来说将不是一件难事。
我们一般使用 Anaconda（Miniconda）来管理个人电脑乃至 Linux 服务器上的 Python 环境，请提前安装并学习 conda 的使用。
此外在科研中我们经常会与他人合作，因此请保持良好的代码习惯，如果你不了解代码规范，请参考 [Google 的 Python 代码规范](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/)
4. 数学基础，作为一名理工科的学生，你应该已经学过高等数学（数学分析）、线性代数（高等代数）、概率论与统计等基础数学课程，在入门阶段我们涉及到的数学知识较为简单，但是扎实的数理基础会支撑你走得更深更远。
5. 最好拥有 Linux 系统使用经验，目前是深度学习的时代，对于自然语言处理领域，又是大规模预训练语言模型的时代，个人电脑无法支撑大模型的训练，我们将使用 Linux 服务器进行 Coding 和实验，提前了解工作流程会大大提高效率。
本教程[动手实践](#动手实践)部分基于 CNN 和 RNN（LSTM）的模型理论上可以在个人电脑上运行，如需 GPU 资源，请联系实验室服务器管理员。


## 基础知识

我们默认大家已经完成了计算机专业本科一年级和二年级的相关课程，拥有一定的数学和编程基础

### 机器学习

虽然目前是深度学习的时代，我们也很少使用传统机器学习的算法来解决问题，但是一方面一些基础概念仍然是相通的，另一方面经典机器学习算法的思想，如 EM、LDA 等，在深度学习时代往往能够历久弥新，以另一种方式焕发出新的光彩。
对于想要快速入门的初学者来说，建议先熟悉机器学习基础概念（什么是机器学习，机器学习用来干什么，什么是数据集，如何对机器学习算法进行评测等），了解几种具体的经典机器学习算法。

对于初学者可以学习：
* 网课：吴恩达 机器学习公开课；李宏毅 机器学习
* 书：机器学习（周志华，西瓜书），统计学习方法（李航）

如果想更深地了解：
* 网课：[机器学习白板推导](https://www.bilibili.com/video/BV1aE411o7qd)
* 书
  * [Pattern Recognition And Machine Learning](https://www.cs.uoi.gr/~arly/courses/ml/tmp/Bishop_book.pdf) (PRML)，以贝叶斯的视角介绍机器学习算法。
  * Machine Learning: A Probabilistic Prospective (MLAPP)，机器学习的百科全书，同样偏重贝叶斯视角。原书成书于2012年，该作者又相继推出了 [Probabilistic Machine Learning: An Introduction](https://github.com/probml/pml-book) 和 [Probabilistic Machine Learning: Advanced Topics](https://github.com/probml/pml2-book)
  * The Elements of Statistical Learning (ESL)，频率派

### 深度学习

深度学习的发展为我们的世界带来了巨大的改变，2018的图灵奖也颁给了对深度学习有卓越贡献的 Yoshua Bengio、Yann LeCun、Geoffrey Hinton。

书：Deep Learning（GoodFellow, Bengio, Courville），神经网络与深度学习（邱锡鹏）

对于初学者来说，仅仅了解深度学习的基本概念、基本算法是不够的，更应当到代码当中去获得更为直观和深入的认识。大家可能也听说过 TensorFlow、PyTorch 这样的深度学习框架，目前学术界通常使用 PyTorch。

PyTorch 对初学者也提供了[快速入门指南](https://pytorch.org/tutorials/beginner/basics/intro.html)和 [tutorial](https://pytorch.org/tutorials/)，对于 tutorial，建议从[简单的图像分类算法](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html#)学起，然后再进一步学习[简单的文本分类](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)、[简单的文本生成](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html)等自然语言处理相关教程。

PyTorch 提供了非常详细的[文档](https://pytorch.org/docs/stable/index.html)，遇到不明白的函数、概念都可以在文档中进行查询和学习

### 自然语言处理

我们实验室的名称为计算语言学研究所，通常意义上[计算语言学](https://zh.wikipedia.org/zh-hans/%E8%AE%A1%E7%AE%97%E8%AF%AD%E8%A8%80%E5%AD%A6)（Computational Linguistics，CL）属于语言学的一个分支，而[自然语言处理](https://zh.wikipedia.org/zh/%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86)（Natural Language Processing，NLP），在现代意义上两者往往会混为一谈。

什么是自然语言处理或者计算语言学？这里摘抄一段 The Association for Computational Linguistics (ACL) 的介绍：
"Computational linguistics is the scientific study of language from a computational perspective. Computational linguists are interested in providing computational models of various kinds of linguistic phenomena. These models may be "knowledge-based" ("hand-crafted") or "data-driven" ("statistical" or "empirical"). Work in computational linguistics is in some cases motivated from a scientific perspective in that one is trying to provide a computational explanation for a particular linguistic or psycholinguistic phenomenon; and in other cases the motivation may be more purely technological in that one wants to provide a working component of a speech or natural language system. Indeed, the work of computational linguists is incorporated into many working systems today, including speech recognition systems, text-to-speech synthesizers, automated voice response systems, web search engines, text editors, language instruction materials, to name just a few."

NLP 包含哪些 topic 呢？同样是摘抄自 60th Annual Meeting of the Association for Computational Linguistics 的 Submissions Topics：
* Computational Social Science and Cultural Analytics
* Dialogue and Interactive Systems
* Discourse and Pragmatics
* Ethics and NLP
* Generation
* Information Extraction
* Information Retrieval and Text Mining
* Interpretability and Analysis of Models for NLP
* Language Grounding to Vision, Robotics and Beyond
* Linguistic Theories, Cognitive Modeling, and Psycholinguistics
* Machine Learning for NLP
* Machine Translation and Multilinguality
* NLP Applications
* Phonology, Morphology, and Word Segmentation
* Question Answering
* Resources and Evaluation
* Semantics: Lexical
* Semantics: Sentence-level Semantics, Textual Inference, and Other Areas
* Sentiment Analysis, Stylistic Analysis, and Argument Mining
* Speech and Multimodality
* Summarization
* Syntax: Tagging, Chunking and Parsing

可以看到 NLP 这个语言学和计算机科学的交叉学科实在是包含了太多的研究方向，而其中除了机器翻译（MT）、摘要、QA 这些大家早有耳闻的应用，剩下的相信初学者大多从未听说过，即使是一位 NLP 研究者或从业人员也只能对这个列表中的某一个或几个方面有深入的研究。


想要对 NLP 是研究什么的有个大致的了解，首先我们可以快速了解深度学习时代 NLP 发展历史：A Review of the Neural History of Natural Language Processing(https://ruder.io/a-review-of-the-recent-history-of-nlp/ )，然后我们可以通过课程或书籍进行系统的学习：

* 网课：
  * [Stanford cs224n](https://web.stanford.edu/class/cs224n/)（强烈推荐，主讲人是绝对的大牛 Christopher Manning，此课程从深度学习的角度出发对 NLP 进行全面的介绍，而其中的 talk 又涉及学术最前沿的进展，可谓广度与深度俱全）
  * CMU CS 11-747
* 书：
  * 统计自然语言处理（宗成庆）成书年代较早，具体方法与当下有较大距离，可了解 NLP 基本问题
  * 现代自然语言生成（黄民烈），关注自然语言生成（Natural Language Generation，NLG）
  * 自然语言处理：基于预训练模型的方法（车万翔），当今预训练语言模型（Pretrained Language Model，PLM）俨然成为了 NLP 中的“基础设施”（Foundation Model），“预训练-微调”（Pretrain & Fine-tune）也成为了应用中的基本范式，因此我们同样需要了解基于预训练模型的方法


## 文献阅读

### Google Scholar

[Google Scholar](https://scholar.google.com/) 可以理解为学术界的 Google

### 会议论文

我们主要阅读国际会议论文，相关的会议有：

- 自然语言处理相关会议：ACL, EMNLP, NAACL, COLING（按影响力排序）
- ML 理论：ICML, NeurIPS, ICLR
- AI 应用：AAAI, IJCAI（这两个会议近年来影响力下降）

其中，ACL 系会议提供 anthology (https://aclweb.org/anthology/)，可以方便地查找历年自然语言处理领域的论文

### 前沿进展

如果想了解某一个领域的前沿进展，通常会关注 [arXiv](https://arxiv.org/)（预印本），部分作者会选择在发表前将论文上传至 arXiv。arXiv 在工作日[每日更新](https://arxiv.org/list/cs.CL/recent)，便于及时追踪前沿动态


### 工具

经典论文往往在 CSDN、知乎等平台有中文读后感，可以辅助阅读

文献分类整理是一个好习惯，建议根据个人喜好选择诸如 Zotero（界面简洁、跨平台、免费、扩展丰富）, Endnote, Mendeley, Papers 等文献管理软件

初学时做好论文笔记，可以使用 Markdown，也可以使用 Notion、Obsidian、OneNote 等笔记软件

### 开源代码

随着时代的进步，越来越多的工作会开源代码，便于其他研究者 follow 和复现实验。开源链接（GitHub 链接）一般会在论文中出现，可以在论文中 `Ctrl-F` 搜索 “github” 快速定位开源链接。

Git 和 GitHub 基本操作请自行学习。如果在复现论文的过程中出现问题，可以在开源代码的 Issues 中提问，在复现之前也应当先查看 Issues，免得踩别人踩过的坑。


## 动手实践

作为计算机科学的一个分支，NLP 同样离不开 coding，请有志加入 TANGENT 的同学完成以下练习任务。

### 写在前面

在完成这些任务之前，还是需要一些说明。

一个深度学习项目的流程通常是这样的：
1. 数据读取和预处理，得到 Dataset 和 DataLoader
2. 构建 Model、Optimizer
3. 使用随机梯度下降迭代优化模型参数
4. 设置 Metric，对模型进行评测

通常我们也会按照上述流程和流程中出现的各个模块组织项目文件，一个项目往往会包含这些文件：主函数（入口，负责以上流程的控制），数据读取和预处理，模型，Metric。

我们针对任务二，给出了一个 ChineseNER 完整项目的[源代码](https://github.com/PKU-TANGENT/nlp-tutorial/tree/main/ChineseNER)。

需注意，下面部分任务参考代码是以 Notebook 的形式组织的，在完成任务时，请参考 ChineseNER 重新组织代码。

此外还有科研工作者的老大难问题：环境配置，在进行以下任务之前，请在个人电脑或服务器配置好 Anaconda 或 miniconda，新建环境，并安装最新版 PyTorch

### 任务一：基于深度学习的文本分类

文本分类是入门 NLP 的一个好的开始，同时 NLU（自然语言理解）任务本质上来说都可以归类为文本分类。请使用 CNN 或 RNN（LSTM） 完成 Kaggle 上一个简单的文本分类任务。

任务描述 & 数据集：https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/

Kaggle 里也有一些[代码](https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/code)可以参考，如：[LSTM 实现](https://www.kaggle.com/code/hanjoonchoe/movie-sentimental-analysis-lstm-pytorch)

参考文献：
Convolutional Neural Networks for Sentence Classification (https://aclanthology.org/D14-1181/)
Recurrent Convolutional Neural Networks for Text Classification (https://www.deeplearningitalia.com/wp-content/uploads/2018/03/Recurrent-Convolutional-Neural-Networks-for-Text-Classification.pdf)


### 任务二：基于 LSTM-CRF 的命名实体识别

在 NLP 中，结构预测（Structured Prediction）是指输出空间为结构化对象的一类任务，包括命名实体识别、关系抽取、共指消解等子任务，命名实体识别又属于序列标注问题。请实现简单的基于 LSTM-CRF 的命名实体识别

任务描述：https://www.clips.uantwerpen.be/conll2003/ner/

数据集：本仓库 [CoNLL03](https://github.com/PKU-TANGENT/nlp-tutorial/tree/main/CoNLL03) 文件夹下

参考文献：
Neural Architectures for Named Entity Recognition (https://arxiv.org/pdf/1603.01360.pdf)

为了简化任务难度，我们给出了基于 LSTM 的中文命名实体识别的[代码](https://github.com/PKU-TANGENT/nlp-tutorial/tree/main/ChineseNER)，配置好 PyTorch 环境后，从 `ChineseNER/train.py` 即可直接运行。可参考该代码将其迁移至 CoNLL03 英文数据集上，进行实验观察初步结果，后续再增加 CRF 层。


### 任务三：Neural Machine Translation (NMT)

摘要和翻译是文本生成中比较主流的两大任务，在这里我们选取 PyTorch tutorial 中的文本翻译作为入门项目。

请按照 [PyTorch 文本翻译教程](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)，一步步实现一个简单的文本翻译模型，注意请参考 ChineseNER 的组织方式重构代码。

生成任务涉及到的细节较多，如 encoder-decoder，teacher forcing，beam search 等，tutorial 中给出了深入浅出的介绍，请仔细阅读并理解。



### 任务四：Transformer & PLM
#### 基础知识
以 BERT、GPT 为代表的预训练语言模型（Pretrain Language Model，PLM）的出现使 NLP 翻开了新的一页，目前的预训练语言模型大多基于 Transformer，因此想要追踪前沿 NLP 技术，我们不得不对 Transformer 有深入的理解。

请结合 Attention Is All You Need 原论文，读懂 [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

建议继续阅读：
[encoder-decoder 结构](https://huggingface.co/blog/encoder-decoder#encoder-decoder)
[可视化 Transformer](http://jalammar.github.io/illustrated-transformer/)
[关于 decode](https://huggingface.co/blog/how-to-generate)

关于预训练语言模型，请阅读 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding 并做阅读笔记，重点关注 BERT 是如何训练出来的，以及如何将 BERT 应用于下游任务。
#### Huggingface Transformers
我们在实践中通常会使用 HuggingFace🤗 的 Transformers 库，该库提供了包括 BERT 和 GPT 在内的常见预训练语言模型，代码风格较好，[文档](https://huggingface.co/docs/transformers/main/index)详细。我们可以通过 [Transformers 教程](https://huggingface.co/course/)进行学习。重点读懂:
- [transformers.models.modeling_bert](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py)
#### Huggingface Ecosystem
在编写机器学习代码时，我们往往会套用成熟的模板作为训练流程的框架。目前，Huggingface所提供的Trainer框架是很好的选择。另外，Huggingface的开源生态较为完善 (包括但不限于datasets, evaluate, tokenizers (一般与Transformers绑定), diffusers, huggingface-hub, accelerate)。事实上，选择Huggingface Ecosystem可以很大程度统一代码风格，从而促进合作开发。
#### 基于Huggingface Trainer的分类任务
Huggingface Transformers仓库中给出了许多示例代码，其中一个非常泛用的模板可以在这里找到:

https://github.com/huggingface/transformers/tree/28a0811652c680078503a56703327f267b9bdb9a/examples/pytorch/text-classification

不同版本的Transformers可能会对代码有删改，推荐学习v4.22.0之后的版本。具体的目标包括:
- 跑通GLUE训练代码
- Debug代码，尤其关注
  - 如何继承`PretrainedModel` 并定制Model, 参考 [BertForSequenceClassification](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L1506)
  - [`Trainer._inner_training_loop`](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L1507-L1891) 中的主要逻辑
  - [`TrainingArguments`](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L121) 中已经实现好，并且涉及到的主要参数
- 尝试使用`transformers`集成的`tensorboard`或者`wandb`功能可视化训练过程
- 尝试魔改`evaluate`及其调用的子方法，实现训练过程中更多指标的可视化
<!-- 完成本小节任务后，如果学有余力，可尝试基于 Transformers 库，实现基于 BERT 的文本分类和 NER。 -->



## 本仓库的使用说明

1. 有问题就提在issues里面，同理你也可以在issues里面检索是否已经有你遇到的问题；
2. main分支无法直接修改，所有修改均需要通过提交`Pull requests`来实现，必须选择至少一个reviewer，推荐选择大师兄`Yifan-Song793`来review；
3. git commit的规范看[这里](https://juejin.cn/post/6844903793033756680)。
