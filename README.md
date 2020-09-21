# 找寻近义词的三种方法

比如我手里有两个词语：`孩子`、`知道`，我想获得他们的近义词；或者我想判断它们和另外几个词如：`孩童`、`父母`、`清楚`、`失信`、`家长`、`小子` 等是不是近义词；

这类找寻近义词问题的基本解决思路都是差不多的，可以分为两步：

- 先将词语转换成向量
- 然后计算向量间的相似度

其中，生成词向量有很多种方法，如：使用Word2Vec训练生成、使用Bert生成、使用开放的训练好的词向量等；计算相似度也有多种方式。

我此次尝试，试了下谷歌预训练过的中文 **Bert** 模型 [https://github.com/google-research/bert#pre-trained-models](https://github.com/google-research/bert#pre-trained-models) 加上封装好的 **Bert as Service** 模块[https://github.com/hanxiao/bert-as-service](https://github.com/hanxiao/bert-as-service) 来获得词向量，也试了试**腾讯AI实验室**开源的词向量数据集[https://ai.tencent.com/ailab/nlp/embedding.html](https://ai.tencent.com/ailab/nlp/embedding.html)来获得词向量。

至于相似度，我是用的基本的余弦相似度计算方法，没有去了解其他优化方法。

在了解上面两个方法的时候，我还发现了一个中文近义词工具包 [Synonyms](https://github.com/huyingxi/Synonyms) 

所以，下面将分别对这三种方式做简单记录。

### Synonyms 工具包

这是一个python的模块，使用前需要安装一下：

```shell
pip install synonyms -i https://mirrors.aliyun.com/pypi/simple
```

使用起来十分简单：

```python
>>> import synonyms  # 先导入模块
# 这里会有一些提示，是在加载字典和模型
>>> synonyms.nearby("小孩")  # 获取“小孩”的近义词
(['小孩', '孩子', '孩童', '小孩子', '男孩', '孩子们', '女孩', '妈妈', '小女孩', '男孩子'], [1.0, 0.8427243, 0.8069258, 0.7965132, 0.76880276, 0.75509995, 0.74481255, 0.7379649, 0.7226252, 0.71569645])
```

上面这个例子的结果显示，和“小孩”这个词最相近的词是“小孩”，相似度1.0；其次是“孩子”，相似度0.8427243；以此类推。这个工具包已经能够满足很多需求了，它还自带了分词、获取词向量、计算两个句子的相似度等功能。

### Bert中文预训练模型 + Bert as Service

其实有了Bert中文预训练模型就可以自己写代码生成词向量了，但是，过程稍微繁琐一点。所以，就发现了一个偷懒的方法……使用 Bert as Service 这个python模块。这个模块分为服务端和客户端，服务端和客户端可以安装在两台不同的机器上，服务端负责读取预训练模型和运算，客户端负责发送和接收词语。

使用这个方法首先需要下载Bert中文预训练模型（网址上面有，大小300多兆），然后安装Bert as Service的服务端模块和客户端模块。

```shell
pip install bert-serving-server -i https://mirrors.aliyun.com/pypi/simple  # 服务端
pip install bert-serving-client -i https://mirrors.aliyun.com/pypi/simple  # 客户端
```

本人并没有两台机器，所以我将两个模块都安装到了同一台机器上。

然后我开了一个终端执行命令启动服务端程序：

```shell
bert-serving-start -model_dir ./chinese_L-12_H-768_A-12/ -num_worker=1
```

其中 `./chinese_L-12_H-768_A-12/` 是下载的预训练模型的路径，我的就在当前目录下。`-num_worker=1`是指使用1个CPU或GPU，这里的官方示例是写的4，视机器情况改改。

经过一段时间的等待，服务端启动了完成了，此时会看到 ready 之类的提示输出，提示我们服务端已经就绪。

这时候，就可以通过客户端来获取词向量了：

```python
from bert_serving.client import BertClient  # 引入相应模块
bc = BertClient()  # 创建一个客户端实例
vectors = bc.encode(["小孩", "清楚"])  # 传入需要转换成向量的词语列表，返回结果是对应的向量列表
```

拿到了vectors，那就可以向量之间计算余弦相似度，排序寻找近义词了。

### 使用腾讯AI实验室开放的词向量数据集

首先是下载这个数据集，一共6个G

下载完成之后解压，会得到一个 `README.txt` 说明文件和数据 `Tencent_AILab_ChineseEmbedding.txt`

使用这个数据集可以借助 `gensim` 这个模块

```shell
pip install gensim -i https://mirrors.aliyun.com/pypi/simple  # 安装这个模块
```

然后理论上就可以导入数据使用了，也就是：

```python
from gensim.models import KeyedVectors  # 导入gensim模块下的KeyedVectors类，用于导入数据
wv = KeyedVectors.load_word2vec_format('Tencent_AILab_ChineseEmbedding.txt', binary=False)  # 导入词向量数据
```

但是，这里有但是了，6G大小的数据在解压后是15.5G大小，据说，完整载入内存需要18个G的内存空间，这显然不是手中的笔记本电脑能hold的住的，所以我的做法是从数据集中筛选出我需要的那些词语和向量，得到一份缩减版的数据，然后再导入使用。

```python
wv.most_similar("小孩", topn=3)  # 获取数据中和“小孩”最相近的前三个词语
```

上面这行代码就可以获取近义词了，这里不需要自己去计算相似度和排序了。

### 部分结果

以上就是三种获取近义词方法的记录

我用这三种方法试着对哈工大同义词词林[https://github.com/BiLiangLtd/WordSimilarity/tree/master/data](https://github.com/BiLiangLtd/WordSimilarity/tree/master/data)里的部分数据做了下计算

下面是部分结果：

- 这是用Bert做的

```
知道,失信,0.9092331528663635
知道,闻名,0.9076200127601624
知道,霸道,0.9073370695114136

孩子,孩儿,0.9516780376434326
孩子,孩童,0.9505244493484497
孩子,小子,0.9290173053741455
```

- 这是用腾讯AI实验室公开的词向量数据做的

```
知道,明白,0.7958440184593201
知道,可是,0.7454869151115417
知道,清楚,0.7410669326782227

孩子,家长,0.8679977655410767
孩子,父母,0.8400565385818481
孩子,小孩,0.8278622031211853
```

- 这是用synonyms工具包做的

```
'知道'近义词：
  1. 知道:1.0
  2. 晓得:0.838299
  3. 明白:0.81247455
  4. 认得:0.7662381
  5. 知晓:0.7469959
  6. 想到:0.71753424
  7. 发觉:0.6753804
  8. 弄清楚:0.67261803
  9. 搞清楚:0.6656825
  10. 相信:0.6608304

'孩子'近义词：
  1. 孩子:1.0
  2. 小孩:0.8830553
  3. 小孩子:0.81332326
  4. 爸爸妈妈:0.79903644
  5. 父母:0.7785208
  6. 宝宝:0.77398
  7. 小朋友:0.7715908
  8. 孩童:0.7444508
  9. 妈妈:0.7437662
  10. 女孩:0.74150336
```

### 结尾

希望以后有时间能对上面涉及到的Word2Vec和Bert有更深入的了解，而不仅仅是调几个包看看效果，以上。
