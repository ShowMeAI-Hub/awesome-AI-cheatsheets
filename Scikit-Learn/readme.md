# Scikit-Learn
<table align="left">
  <td>
    <a target="_blank" href="http://nbviewer.ipython.org/github/ShowMeAI-Hub/awesome-AI-cheatsheets/blob/main/Scikit-Learn/Scikit-Learn-cheatsheet-code.ipynb"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" /><br>在nbviewer上查看notebook</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/ShowMeAI-Hub/awesome-AI-cheatsheets/blob/main/Scikit-Learn/Scikit-Learn-cheatsheet-code.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" /><br>在Google Colab运行</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets/Scikit-Learn/Scikit-Learn-cheatsheet-code.ipynb"><img src="https://badgen.net/badge/open/github/color=cyan?icon=github" /><br>在Github上查看源代码</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets/Scikit-Learn/Scikit-Learn速查表.pdf"><img src="https://badgen.net/badge/download/pdf/color=white?icon=github"/><br>下载速查表</a>
  </td>
</table>
<br></br>
<br>

## 说明
**notebook by [韩信子](https://github.com/HanXinzi-AI)@[ShowMeAI](https://github.com/ShowMeAI-Hub)**

更多AI速查表资料请查看[速查表大全](https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets)

Scikit-learn是开源的Python库，通过统一的界面实现机器学习、预处理、交叉验证及可视化算法。

## 简例


```python
# 导入工具库
from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据
iris = datasets.load_iris()
X, y = iris.data[:, :2], iris.target

# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)

# 数据预处理
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# 训练与预测
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# 评估
accuracy_score(y_test, y_pred)
```

## 加载数据

Scikit-learn处理的数据是存储为NumPy数组或SciPy稀疏矩阵的数字，还支持Pandas数据框等可转换为数字数组的其它数据类型。


```python
import numpy as np
```


```python
X = np.random.random((10,5))
```


```python
y = np.array(['M','M','F','F','M','F','M','M','F','F'])
```


```python
X[X < 0.7] = 0
```

## 训练/测试集切分


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

## 数据预处理

### 标准化


```python
from sklearn.preprocessing import StandardScaler
```


```python
scaler = StandardScaler().fit(X_train)     #拟合
```


```python
standardized_X = scaler.transform(X_train)   #训练集变换
```


```python
standardized_X_test = scaler.transform(X_test)   #测试集变换
```

### 归一化


```python
from sklearn.preprocessing import Normalizer
```


```python
scaler = Normalizer().fit(X_train)     #拟合
```


```python
normalized_X = scaler.transform(X_train)   #训练集变换
```


```python
normalized_X_test = scaler.transform(X_test)   #测试集变换
```

### 二值化


```python
from sklearn.preprocessing import Binarizer
```


```python
binarizer = Binarizer(threshold=0.0).fit(X)     #拟合
```


```python
binary_X = binarizer.transform(X)     #变换
```

### 编码分类特征


```python
from sklearn.preprocessing import LabelEncoder
```


```python
enc = LabelEncoder()
```


```python
y = enc.fit_transform(y)
```

### 缺失值处理


```python
from sklearn.impute import SimpleImputer
```


```python
imp = SimpleImputer(missing_values=0, strategy='mean')    #均值填充器
```


```python
imp.fit_transform(X_train)   #对数据进行缺失值均值填充变换
```

### 生成多项式特征


```python
from sklearn.preprocessing import PolynomialFeatures
```


```python
poly = PolynomialFeatures(5)
```


```python
poly.fit_transform(X)
```

## 创建模型

### 有监督学习评估器

**线性回归**


```python
from sklearn.linear_model import LinearRegression
```


```python
lr = LinearRegression(normalize=True)
```

**支持向量机(SVM)**


```python
from sklearn.svm import SVC
```


```python
svc = SVC(kernel='linear')
```

**朴素贝叶斯**


```python
from sklearn.naive_bayes import GaussianNB
```


```python
gnb = GaussianNB()
```

**KNN**


```python
from sklearn import neighbors
```


```python
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
```

### 无监督学习评估器

**主成分分析(PCA)**


```python
from sklearn.decomposition import PCA
```


```python
pca = PCA(n_components=0.95)
```

**K-Means聚类**


```python
from sklearn.cluster import KMeans
```


```python
k_means = KMeans(n_clusters=3, random_state=0)
```


```python
1. ## 模型拟合
```

### 有监督学习


```python
lr.fit(X, y)   #拟合数据与模型
```


```python
knn.fit(X_train, y_train)
```


```python
svc.fit(X_train, y_train)
```

### 无监督学习


```python
k_means.fit(X_train)   #拟合数据与模型
```


```python
pca_model = pca.fit_transform(X_train)   #拟合并转换数据
```

## 预测

### 有监督评估器


```python
y_pred = svc.predict(np.random.random((2,5)))   #预测标签
```


```python
y_pred = lr.predict(X_test)   #预测标签
```


```python
y_pred= knn.predict_proba(X_test)   #评估标签概率
```

### 无监督评估器


```python
y_pred = k_means.predict(X_test)   #预测聚类算法里的标签
```

## 评估模型性能

### 分类评价指标

**准确率**


```python
svc.fit(X_train, y_train)
svc.score(X_test, y_test)   #评估器评分法
```


```python
from sklearn.metrics import accuracy_score   #指标评分函数
```


```python
y_pred = svc.predict(X_test)
accuracy_score(y_test, y_pred)  #评估accuracy
```

**分类预估评价函数**


```python
from sklearn.metrics import classification_report   #精确度、召回率、F1分数及支持率
```


```python
print(classification_report(y_test, y_pred))
```

**混淆矩阵**


```python
from sklearn.metrics import confusion_matrix
```


```python
print(confusion_matrix(y_test, y_pred))
```

### 回归评价指标

**平均绝对误差**


```python
from sklearn.metrics import mean_absolute_error
```


```python
house_price = datasets.load_boston()
X, y = house_price.data, house_price.target
house_X_train, house_X_test, house_y_train, house_y_test = train_test_split(X, y, random_state=0)
```


```python
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor().fit(house_X_train, house_y_train)
house_y_pred = dt.predict(house_X_test)
mean_absolute_error(house_y_test, house_y_pred)
```

**均方误差**


```python
from sklearn.metrics import mean_squared_error
```


```python
mean_squared_error(house_y_test, house_y_pred)
```

**R^2评分**


```python
from sklearn.metrics import r2_score
```


```python
r2_score(house_y_test, house_y_pred)
```

### 聚类评价指标

**调整兰德系数**


```python
from sklearn.metrics import adjusted_rand_score
```


```python
adjusted_rand_score(y_true, y_pred)
```

**同质性**


```python
from sklearn.metrics import homogeneity_score
```


```python
homogeneity_score(y_true, y_pred)
```

**V-measure**


```python
from sklearn.metrics import v_measure_score
```


```python
metrics.v_measure_score(y_true, y_pred)
```

### 交叉验证


```python
from sklearn.model_selection import cross_val_score
```


```python
print(cross_val_score(knn, X_train, y_train, cv=4))
```


```python
print(cross_val_score(lr, X, y, cv=2))
```

## 模型调参与优化

### 网格搜索超参优化


```python
from sklearn.model_selection import GridSearchCV

params = {"n_neighbors": np.arange(1,3), "metric": ["euclidean", "cityblock"]}
grid = GridSearchCV(estimator=knn, param_grid=params)

grid.fit(X_train, y_train)

print(grid.best_score_)
print(grid.best_estimator_.n_neighbors)
```

### 随机搜索超参优化


```python
from sklearn.model_selection import RandomizedSearchCV

params = {"n_neighbors": range(1,5),
            "weights": ["uniform", "distance"]}

rsearch = RandomizedSearchCV(estimator=knn,
                             param_distributions=params,
                             cv=4,
                             n_iter=8,
                             random_state=5)

rsearch.fit(X_train, y_train)
print(rsearch.best_score_)
```
