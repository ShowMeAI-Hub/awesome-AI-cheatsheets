# SciPy
<table align="left">
  <td>
    <a target="_blank" href="http://nbviewer.ipython.org/github/ShowMeAI-Hub/awesome-AI-cheatsheets/blob/main/Scipy/Scipy-cheatsheet-code.ipynb"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" /><br>在nbviewer上查看notebook</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/ShowMeAI-Hub/awesome-AI-cheatsheets/blob/main/Scipy/Scipy-cheatsheet-code.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" /><br>在Google Colab运行</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets/Scipy/Scipy-cheatsheet-code.ipynb"><img src="https://badgen.net/badge/open/github/color=cyan?icon=github" /><br>在Github上查看源代码</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets/Scipy/Scipy速查表.pdf"><img src="https://badgen.net/badge/download/pdf/color=white?icon=github"/><br>下载速查表</a>
  </td>
</table>
<br></br>
<br>

## 说明
**notebook by [韩信子](https://github.com/HanXinzi-AI)@[ShowMeAI](https://github.com/ShowMeAI-Hub)**

更多AI速查表资料请查看[速查表大全](https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets)

SciPy是基于NumPy创建的Python科学计算核心库，提供了众多数学算法与函数。

## 与NumPy交互

### 初始化


```python
import numpy as np
```


```python
a = np.array([1, 2, 3])
```


```python
b = np.array([(1+5j, 2j, 3j), (4j, 5j, 6j)])
```


```python
c = np.array([[(1.5, 2, 3), (4, 5, 6)], [(3, 2, 1), (4, 5, 6)]])
```

### 索引技巧


```python
np.mgrid[0:5, 0:5]   #创建稠密栅格
```


```python
np.ogrid[0:2, 0:2]   #创建开放栅格
```


```python
np.r_[3, [0]*5, -1:1:10j]   #按行纵向堆叠数组
```


```python
np.c_[a, a]   #按列横向堆叠数组
```

### 操控形状


```python
np.transpose(b)   #转置矩阵
```


```python
b.flatten()   #拉平数组
```


```python
np.hstack((c, c))   #按列横向堆叠数组
```


```python
np.vstack((a, b))   #按行纵向堆叠数组
```


```python
np.hsplit(c, 2)   #在索引2横向分割数组
```


```python
np.vsplit(c, 2)   #在索引2纵向分割数组
```

### 多项式


```python
from numpy import poly1d
```


```python
p = poly1d([3, 4, 5])   #创建多项式对象
```

### 矢量函数


```python
def myfunc(a):
    if a < 0:
        return a*2
    else:
        return a/2
```


```python
np.vectorize(myfunc)   #矢量函数
```

### 类型控制


```python
np.real(c)   #返回数组元素的实部
```


```python
np.imag(c)   #返回数组元素的虚部
```


```python
np.real_if_close(c, tol=1000)   #如果复数接近0，返回实部
```


```python
np.cast['f'](np.pi)   #将对象转化为数据类型
```

### 常用函数


```python
np.angle(b, deg=True)   #返回复数的辐角
```


```python
g = np.linspace(0, np.pi, num=5)   #创建等差数组(样本数)
```


```python
g [3:] += np.pi
```


```python
np.unwrap(g)   #解包
```


```python
np.logspace(0, 10, 3)   #创建等差数组(对数刻度)
```


```python
np.select([c<4], [c*2])   #根据条件返回数组列表的值
```


```python
from scipy import special
special.factorial(a)   #因子
```


```python
import scipy
scipy.special.comb(10, 3, exact=True)   #计算排列组合$$C_{10}^3$$
```


```python
from scipy import misc
misc.central_diff_weights(3)   #NP点中心导数的权重
```


```python
misc.derivative(myfunc, 1.0)   #查找函数在某点的第n个导数
```

## 线性代数

使用linalg和sparse模块。注意scipy.linalg包含了numpy.linalg，并扩展了其功能。


```python
 from scipy import linalg, sparse
```

### 创建矩阵


```python
A = np.matrix(np.random.random((2, 2)))
```


```python
B = np.asmatrix(b)
```


```python
C = np.mat(np.random.random((10, 5)))
```


```python
D = np.mat([[3, 4], [5, 6]])
```

### 基础矩阵操作

**逆矩阵**


```python
A.I   #求逆矩阵
```


```python
linalg.inv(A)   #求逆矩阵
```


```python
A.T   #矩阵转置
```


```python
A.H   #共轭转置
```


```python
np.trace(A)   #计算对角线元素的和
```

**范数**


```python
linalg.norm(A)   #Frobeniu范数
```


```python
linalg.norm(A, 1)   #L1范数 (最大列汇总)
```


```python
linalg.norm(A, np.inf)   #L范数 (最大列汇总)
```

**排名**


```python
np.linalg.matrix_rank(C)   #矩阵排名
```

**行列式**


```python
linalg.det(A)   #行列式
```

**求解线性问题**


```python
linalg.solve(A, b)   #求解稠密矩阵
```


```python
E = np.mat(a).T   #求解稠密矩阵
```


```python
linalg.lstsq(D, D)   #用最小二乘法求解线性代数方程
```

**广义逆**


```python
linalg.pinv(C)   #计算矩阵的伪逆(最小二乘法求解器)
```


```python
linalg.pinv2(C)   #计算矩阵的伪逆(SVD)
```

### 创建稀疏矩阵


```python
F = np.eye(3, k=1)   #创建2X2单位矩阵
```


```python
G = np.mat(np.identity(2))   #创建2X2单位矩阵
```


```python
C[C > 0.5] = 0
```


```python
H = sparse.csr_matrix(C)   #压缩稀疏行矩阵
```


```python
I = sparse.csc_matrix(D)   #压缩稀疏列矩阵
```


```python
J = sparse.dok_matrix(A)   #DOK矩阵
```


```python
I.todense()   #将稀疏矩阵转为全矩阵
```


```python
sparse.isspmatrix_csc(A)   #单位稀疏矩阵
```

### 稀疏矩阵操作

**逆矩阵**


```python
import scipy.sparse.linalg as linalg
linalg.inv(I)   #求逆矩阵
```

**范数**


```python
linalg.norm(I)   #范数
```

**解决线性问题**


```python
linalg.spsolve(I, I)   #稀求解疏矩阵
```

### 稀疏矩阵函数


```python
sparse.linalg.expm(I)   #稀疏矩阵指数
```

### 矩阵函数

**加法**


```python
np.add(A, D)   #加法
```

**减法**


```python
np.subtract(A, D)   #减法
```

**除法**


```python
np.divide(A, D)   #除法
```

**乘法**


```python
np.multiply(D, A)   #乘法
```


```python
np.dot(A, D)   #点积
```


```python
np.vdot(A, D)   #向量点积
```


```python
np.inner(A, D)   #内积
```


```python
np.outer(A, D)   #外积
```


```python
np.tensordot(A, D)   #张量点积
```


```python
np.kron(A, D)   #Kronecker积
```


```python
help(linalg.expm_multiply)
```

**指数函数**


```python
linalg.expm(A)   #矩阵指数
```

**对数函数**


```python
scipy.linalg.logm(A)   #矩阵对数
```

**三角函数**


```python
scipy.linalg.sinm(D)   #矩阵正弦
```


```python
scipy.linalg.cosm(D)   #矩阵余弦
```


```python
scipy.linalg.tanm(A)   #矩阵切线
```

**双曲三角函数**


```python
scipy.linalg.sinhm(D)   #双曲矩阵正弦
```


```python
scipy.linalg.coshm(D)   #双曲矩阵余弦
```


```python
scipy.linalg.tanhm(A)   #双曲矩阵切线
```

**矩阵符号函数**


```python
np.sign(A)   #矩阵符号函数
```

**矩阵平方根**


```python
scipy.linalg.sqrtm(A)   #矩阵平方根
```

**任意函数**


```python
scipy.linalg.funm(A, lambda x: x*x)   #评估矩阵函数
```

### 矩阵分解

**特征值与特征向量**


```python
la, v = scipy.linalg.eig(A)   #求解方阵的普通或广义特征值问题
```


```python
l1, l2 = la   #解包特征值
```


```python
v[:, 0]   #第一个特征值
```


```python
v[:, 1]   #第二个特征值
```


```python
scipy.linalg.eigvals(A)   #解包特征值
```

**奇异值分解**


```python
U, s, Vh = scipy.linalg.svd(B)   #奇异值分解(SVD)
```


```python
M, N = B.shape
```


```python
Sig = scipy.linalg.diagsvd(s, M, N)   #在SVD中构建Sigma矩阵
```

**LU 分解**


```python
P, L, U = scipy.linalg.lu(C)   #LU分解
```

### 解构稀疏矩阵


```python
la, v = sparse.linalg.eigs(F, 1)   #特征值与特征向量
```


```python
sparse.linalg.svds(H, 2)   #奇异值分解(SVD)
```

## 调用帮助

**help函数**


```python
help(scipy.linalg.diagsvd)
```


```python
np.info(np.matrix)
```
