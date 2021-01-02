#https://zhuanlan.zhihu.com/p/111180811
import time
import matplotlib.pyplot as plt
import matplotlib
from  sklearn.cluster import KMeans
from sklearn.datasets import load_iris 
matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 获取鸢尾花数据集，特征分别是sepal length、sepal width、petal length、petal width
iris = load_iris() 
X = iris.data[:,2:]  # 通过花瓣的两个特征来聚类
k=3  # 假设聚类为3类
# 构建模型
s=time.time()
km = KMeans(n_clusters=k) 
km.fit(X)
print("用sklearn内置的K-Means算法聚类耗时：",time.time()-s)

label_pred = km.labels_   # 获取聚类后的样本所属簇对应值
centroids = km.cluster_centers_  # 获取簇心

#绘制K-Means结果
# 未聚类前的数据分布
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title("未聚类前的数据分布")
plt.subplots_adjust(wspace=0.5)

plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=label_pred, s=50, cmap='viridis')
plt.scatter(centroids[:,0],centroids[:,1],c='red',marker='o',s=100)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title("用sklearn内置的K-Means算法聚类结果")
plt.show()
