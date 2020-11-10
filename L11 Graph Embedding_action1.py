"""
seealsology是个针对Wikipidea页面的语义分析工具，可以找到与指定页面相关的Wikipidea
seealsology-data.tsv 文件存储了Wikipidea页面的关系（Source, Target, Depth）
使用Graph Embedding对节点（Wikipidea）进行Embedding（DeepWalk或Node2Vec模型）
对Embedding进行可视化（使用PCA呈现在二维平面上)找到和critical illness insurance相关的页面
"""
# 导入库
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.decomposition import PCA
from node2vec import Node2Vec
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import networkx as nx

# 加载数据，并构造图
    # TSV文件和CSV的文件的区别是：前者使用\t作为分隔符，后者使用,作为分隔符。
    # 如果已有表头，则可使用header参数：train=pd.read_csv('test.tsv', sep='\t', header=0)
    # 如果已有主键列：train=pd.read_csv('test.tsv', sep='\t', header=0, index_col='id')

df = pd.read_csv('seealsology-data.tsv',sep='\t') 
G = nx.from_pandas_edgelist(df,'source','target',edge_attr=True,create_using=nx.Graph())
create_using = nx.Graph()
print(len(G))

# Node2Vec模型
# node2vec和deepwalk非常类似，主要区别在于顶点序列的采样策略不同，所以这里我们主要关注node2vecWalk的实现。
model = Node2Vec(G, walk_length = 10, num_walks = 5, p = 0.25, q = 4, workers = 1)

# 模型训练
result = model.fit(window=4, iter=20)
print(result.wv.most_similar('critical illness insurance'))
embeddings = result.wv # 得到节点的embedding
print(embeddings)

# 在二维空间中绘制所选节点的向量
def plot_nodes(word_list):
    # 每个节点的embedding为100维
    X = []
    for item in word_list:
        X.append(embeddings[item])
    #print(X.shape)
    # 将100维向量减少到2维
    pca = PCA(n_components=2)
    result = pca.fit_transform(X) 
    #print(result)
    # 绘制节点向量
    plt.figure(figsize=(12,9))
    # 创建一个散点图的投影
    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(list(word_list)):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))        
    plt.show()

plot_nodes(result.wv.vocab) # 将所有的节点embedding进行绘制