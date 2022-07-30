# BiLSTM Baseline



简单的 BiLSTM NER baseline，作为我们 Project 的一个起点

代码已经做好注释，如有不明白之处请及时提问，如发现 bug 请及时报告hhh



## 项目结构

```shell
├── data						# The directory of datasets
│   └── renMinRiBao
│       ├── tags.txt
│       ├── test_data.txt
│       ├── train_data.txt
│       └── val_data.txt
├── model						# The directory to save models
├── BiLSTM.py					# BiLSTM model
├── preprocess_data.py			# Preprocess the datasets
├── train.py					# The entry of this project
└── utils.py					# Some useful functions

```



## 运行环境

```shell
Python 3.7
PyTorch 1.7.1
NumPy 1.19.2
```

请使用 pip install 命令安装以上运行环境（版本不一定非要一样，正常安装最新版一般不会有问题），建议使用 anaconda 或 miniconda 等虚拟环境（自行百度），做好环境管理

执行 `python train.py` 即可按照默认设置训练模型

执行 `python train.py --test` 即可在测试集上进行测试



## 运行结果

训练10个 epoch，准确率：88.58，召回率：83.45，F1：85.94（实际测试结果略有出入为正常现象）



## 学习资料

机器学习基本概念：周志华《机器学习》前几章

Word embedding：https://www.zhihu.com/question/32275069

RNN：https://zhuanlan.zhihu.com/p/123211148，https://zhuanlan.zhihu.com/p/28054589，https://zhuanlan.zhihu.com/p/30844905，https://www.bilibili.com/video/BV1JE411g7XF?p=20

LSTM：http://colah.github.io/posts/2015-08-Understanding-LSTMs/

交叉熵：https://zhuanlan.zhihu.com/p/149186719，https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

梯度下降：https://www.bilibili.com/video/BV1JE411g7XF?p=5

PyTorch document：https://pytorch.org/docs/stable/index.html

PyTorch 教程：https://pytorch.org/tutorials/index.html

课程：李宏毅深度学习，https://www.bilibili.com/video/BV1JE411g7XF