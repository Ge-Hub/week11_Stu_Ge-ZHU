1. 什么是Graph Embedding，都有哪些算法模型
简要说名Graph Embedding，以及相关算法模型（10points）

    Graph Embedding是一种Embedding降维技术，可以有效的挖掘图网络中的节点特征表示
           常用的算法模型有Deep Walk、Node2Vec、LINE、SDNE、Struc2Vec

    在数据结构中，图是一种基础且常用的结构。现实世界中许多场景可以抽象为一种图结构，如社交网络，交通网络，电商网站中用户与物品的关系等。以躺平APP社区为例，它是“躺平”这个大生态中生活方式分享社区，分享生活分享家，努力打造社区交流、好物推荐与居家指南。用户在社区的所有行为：发布、点击、点赞、评论收藏等都可以抽象为网络关系图。因此Graph Embedding技术非常自然地成为学习社区中用户与内容的embedding的一项关键技术。


    目前落地的模型大致两类：
         1. 直接优化节点的浅层网络模型包括基于用户行为理解内容，学习内容向量表征的item2vec,用于扩充i2i召回；同时学习用户与内容向量表征的异构网络表示学习模型metapath2vec，用于提高内容召回的多样性；以群体行为代替个体行为的userCluster2vec，缓解新用户冷启动问题。
        2. 和基于GNN的深层网络模型包括采用邻域聚合的思想，同时融入节点属性信息，通过多层节点聚合使得网络中的拓扑结构能够有效捕捕获的graphSage，以及将attention机制运用邻域信息聚合阶段的GAT，对用户与内容向量表征进行更加细致化学习。

    常用的算法模型有Deep Walk、Node2Vec、LINE、SDNE、Struc2Vec。可以参考https://developer.aliyun.com/article/770625 


2. 如何使用Graph Embedding在推荐系统，比如NetFlix 电影推荐，请说明简要的思路
简要说明Graph Embedding在NetFlix电影推荐中的作用，有自己的见解（10points）

    Graph Embedding——引入更多的结构信息的图嵌入。word2vec 和 item2vec都是建立再序列的样本上的，但是在互联网的场景下，数据对象之间更多呈现的是图结构，在面对图结构的数据时，传统的序列Embedding方法就显得力不从心了，这时候Graph Embedding成为了新的研究方向。

    Graph Embedding 是一种对图结构中的节点进行Embedding编码的方法，最终生成的节点Embedding向量一般包含图的结构信息及附近节点的局部相似性信息。不同的Graph Embedding方法原理不尽相同，对于图信息的保留方式也有所区别。

    DeepWalk-基础graph Embedding的方法
    DeepWalk于2014年提出，其主要思想时在由物品组成的图结构上进行随机游走，产生大量的物品序列，然后将这些物品序列作为训练样本输入Word2Vec进行训练，继而得到物品的Embedding信息。DeepWalk可以被看作时序列Embedding和Graph Embedding的过渡方法。

    2018年，阿里巴巴公布了其在淘宝应用的Embedding方法EGES（Enhanced Graph Embedding with side Infomation），其基本思想是在DeepWalk生成的Graph Embedding基础上引入补充信息。单纯的使用用户行为构建的Graph Embedding可以生成Embedding信息，但是对于新物品或者没有过多「互动」信息的「长尾物品」，推荐系统会表现出很严重的冷启动问题。为了解决这个问题，阿里巴巴引入更多的补充信息来丰富Embedding的来源。在构建物品关系图时，不仅依赖用户的交互行为，也可以利用用户的属性信息建立联系，从而生成基于内容的知识图谱，基于知识图谱生成的向量可以称为补充信息Embedding向量。

    而在NetFlix电影推荐中，需要预测用户对电影的评分并对用户进行推荐。这样做需要过Graph Embedding提取用户和电影的表示向量。也就是说，需要把用户与电影，用户与用户，电影与电影直接搭建知识图谱。



3. 数据探索EDA都有哪些常用的方法和工具
简要说明常用的EDA方法和工具（10points）

    EDA (Exploratory Data Analysis)，即对数据进行探索性的分析。充分了解数据，为之后的数据清洗和特征工程等提供想法和结论。在探索分析时，也可进行数据清洗的工作，两者是相辅相成的，没有说必须按照顺序去执行。
    
    EDA主要包括以下三个方面：
    分布分析：定量定性分析。
    统计量分析：集中，离散趋势和分布形状。
    相关分析：单个图，图矩阵，相关系数。

    项目中，通常需要查看：
    1. 整体情况： 
    df.head() df.info() df.describe() df.value_counts() pandas_profiling 
    np.min() np.max() np.mean() np.std() np.var() np.ptp()
    2. 缺失值: df.isnull().any() df.isnull().sum msno.matrix(sample)
    3. 数据分布：技术偏度和峰度，常用数据分布 norm, lognorm

    参考：https://cloud.tencent.com/developer/article/1625968  在Python中进行探索式数据分析（EDA）