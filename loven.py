# import statsmodels.api as sm
import time
# import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
# from keras import backend as K
from math import log
from sklearn import metrics
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import train_test_split
# from keras.datasets import mnist
# from keras.models import Model  #采用通用模型
# from keras.layers import Dense, Input  #只用到全连接层
# from sklearn.metrics.cluster.supervised import contingency_matrix
from sklearn.metrics import accuracy_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import VarianceThreshold
from math import log
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
np.random.seed(1337)  # for reproducibility

names = [
    'Cao',
    'Deng',
    'Goolam',
    'Grun',
    'Haber',
    'Han',
    'Klein',
    'Kumar',
    'KumarTCC',
    'Macosko',
    'Patel',
    'Petropoulos',
    'Ramskold',
    'Sala',
    'Shekhar',
    'Spallanzani',
    'Tasic',
    'Trapnell',
    'TrapnellTCC',
    'Wallrapp',
    'Wang',
    'Zemmour',
]


#读取数据
path_x = r'C:\Users\86166\Desktop\数据集\{}.csv'.format('Goolam')
path_y = r'C:\Users\86166\Desktop\数据集\{}.csv'.format('Goolamlabel')
data = pd.read_csv(path_x).iloc[:,1:]
label = pd.read_csv(path_y)['label']

#判断是否为x轴为样本，y轴为标签
if label.shape[0] == data.shape[1]:
    data = data.T
x_shape = data.shape[0]
y_shape = data.shape[1]

#分箱去噪
def getEntropy(s):
    prt_ary = 0.0
    data_vc = s.value_counts()
    data_vc = data_vc[data_vc > 5]
    Sum = data_vc.values.sum()
    for key, value in zip(data_vc.index, data_vc.values):
        prt_ary += np.log2(value / Sum) * (value / Sum)
    return -prt_ary

arr_list = []
for i in range(data.shape[1]):   #data是DataFrame格式的数据，行是细胞样本，列是特征
    data_i = pd.cut(data.iloc[:,i],bins = 20,labels = False)
    t = getEntropy(data_i)
    arr_list.append(t > 0.3)
data = data.loc[:, arr_list]
sel = VarianceThreshold(threshold=(0.8 * (1 - 0.8)))
data = sel.fit_transform(data)

# 去掉唯一属性
data = pd.DataFrame(data)
zeros_list = []
for i in range(data.shape[1]):
    if data.iloc[:, i].unique().shape[0] > 1:
        zeros_list.append(i)
data_zeros = data.iloc[:, zeros_list]

# 归一化
scaler = MinMaxScaler()
scaler.fit(data_zeros)
scaler.data_max_
data_zeros_normorlize = scaler.transform(data_zeros)

# 数据异常平滑
data_ol = pd.DataFrame()
for i in range(data_zeros_normorlize.shape[1]):
    ol = np.array(data_zeros_normorlize[:, i])
    ol_mean = ol.mean()
    ol_std = ol.std()
    # 大于最大值等于最大值
    ol[ol > ol_mean + 2 * ol_std] = ol_mean + 2 * ol_std
    # 小于最小值等于最小值
    ol[ol < ol_mean - 2 * ol_std] = ol_mean - 2 * ol_std
    data_ol['x_{}'.format(i)] = ol

df = data_ol.var(axis=0)  # 求每一列的方差
data_ol.loc['var'] = df  # 把方差加到最后一行
data_ol = data_ol.sort_values(by=['var'], axis=1, ascending=False)  # 按照方差的值进行降序排序
data = data_ol.iloc[:, :2000]  # 取方差大的前2000个基因
data = data.drop('var', axis=0)  # 把var这一行删去
data_ol = data

import hnswlib
import time
import os
import psutil


def get_ann(length, dimen):
    # 向量维度
    dim = dimen
    num_elements = length
    data = X
    data_labels = np.arange(num_elements)
    k = int(0.1 * int(X.shape[0]))
    # print(data_labels)
    # 声明索引
    p = hnswlib.Index(space='cosine', dim=dim)  # hnswlib支持的距离有L2距离，向量内积以及cosine相似度
    # 初始化index
    p.init_index(max_elements=num_elements, ef_construction=100, M=16)
    # ef: 动态检索链表的大小。ef必须设置的比检索最近邻的个数K大。ef取值范围为k到集合大小之间的任意值。
    p.set_ef(int(k))
    p.set_num_threads(4)  # cpu多线程并行计算时所占用的线程数

    # 构建items
    p.add_items(X, data_labels)
    index_path = 'data.bin'
    p.save_index(index_path)
    global labels, distances
    labels, distances = p.knn_query(data, k=k)  # 分别包括k个最近邻结果的标签和与这k个标签的距离。


X = DataFrame.as_matrix(data_ol)
# X = data_ol
get_ann(X.shape[0], X.shape[1])
new_labels = labels

num = []
corr_list = []
for j in new_labels:
    for i in j:
        num.append(X[i])
    corr = pd.DataFrame(num).T.corr('spearman')
    corr_list.append(corr)
    num.clear()  # 清空列表所有元素

import matplotlib.pyplot as plt
import networkx as nx

G = nx.Graph()
for i in range(len(new_labels)):
    for j in range(1, len(new_labels[0])):
        G.add_edge(new_labels[i][0], new_labels[i][j], weight=corr_list[i].iloc[j, 0])
A = nx.adjacency_matrix(G)
print(A)
# elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.3]
# esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.3]
pos = nx.spring_layout(G)
# pos = nx.spectral_layout(G)#根据图的拉普拉斯特征向量排列节
# pos = nx.random_layout(G)#节点随机分布
nx.draw_networkx_nodes(G, pos, node_color='red', node_size=10)
nx.draw_networkx_edges(G, pos, width=0.2, alpha=0.9, edge_color='black', style='solid')
plt.axis('off')  # 关闭坐标轴
plt.show()


Adjacency_Matrix_raw = nx.to_numpy_array(G)
from communities.algorithms import louvain_method
from communities.visualization import draw_communities

adj_matrix = Adjacency_Matrix_raw
communities,_= louvain_method(adj_matrix)

draw_communities(adj_matrix, communities)


from communities.algorithms import louvain_method
from communities.visualization import louvain_animation


communities, frames = louvain_method(adj_matrix)

louvain_animation(adj_matrix, frames)
