# NumPy

<table align="left">
  <td>
    <a target="_blank" href="http://nbviewer.ipython.org/github/ShowMeAI-Hub/awesome-AI-cheatsheets/blob/main/Numpy/numpy-cheatsheet-code.ipynb"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" /><br>在nbviewer上查看notebook</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/ShowMeAI-Hub/awesome-AI-cheatsheets/blob/main/Numpy/numpy-cheatsheet-code.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" /><br>在Google Colab运行</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets/tree/main/Numpy/numpy-cheatsheet-code.ipynb"><img src="https://badgen.net/badge/open/github/color=cyan?icon=github" /><br>在Github上查看源代码</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets/tree/main/Numpy/Numpy-Cheatsheet.pdf"><img src="https://badgen.net/badge/download/pdf/color=white?icon=github"/><br>下载速查表</a>
  </td>
</table>
<br></br>
<br>

## 说明
**notebook by [韩信子](https://github.com/HanXinzi-AI)@[ShowMeAI](https://github.com/ShowMeAI-Hub)**

更多AI速查表资料请查看[速查表大全](https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets)

欢迎访问[ShowMeAI官网](http://www.showmeai.tech/)获取更多资源。

## 导入工具库

Numpy是Python数据科学计算的核心库，提供了高性能的多维数组对象及处理数组的工具。

使用以下语句导入Numpy库：


```python
import numpy as np
```

### Numpy数组

1维数组

2维数组【axis 1  axis 0】

3维数组【axis 2  axis 1  axis 0】

## 创建数组

### 初始化


```python
a = np.array([1, 2, 3])
```


```python
b = np.array([(1.5, 2, 3), (4, 5, 6)], dtype = float)
```


```python
c = np.array([[(1.5, 2, 3), (4, 5, 6)], [(3, 2, 1), (4, 5, 6)]], dtype = float)
```

### 特殊数组


```python
np.zeros((3, 4))   #创建值为0数组
```


```python
np.ones((2, 3, 4),dtype=np.int16)   #创建值为1数组
```


```python
d = np.arange(10, 25, 5)   #创建均匀间隔的数组（步进值）
```


```python
np.linspace(0, 2, 9)   #创建均匀间隔的数组（样本数）
```


```python
e = np.full((2, 2), 7)   #创建常数数组
```


```python
f = np.eye(2)   #创建2x2单位矩阵
```


```python
np.random.random((2, 2))   #创建随机值的数组
```


```python
np.empty((3,2))   #创建空数组
```

## 输入/输出

### 保存与载入磁盘上的文件


```python
np.save('my_array', a)
```


```python
np.savez('array.npz', a, b)
```


```python
np.load('my_array.npy')
```


```python
np.savetxt("my_array.txt", a, delimiter=" ")
```

### 保存与载入文本文件


```python
np.loadtxt("my_array.txt")
```


```python
np.genfromtxt("my_array.txt", delimiter=' ')
```

## 数据类型

**有以下的一些数据类型**


```python
np.int64   #带符号的64位整数
```


```python
np.float32   #标准双精度浮点数
```


```python
np.complex   #显示为128位浮点数的复数
```


```python
np.bool   #布尔值：True值和False值
```


```python
np.object   #Python对象
```


```python
np.string_   #固定长度字符串
```


```python
np.unicode_   #固定长度Unicode
```

## 数组信息

**查看数组的基本信息**


```python
a.shape   #数组形状，几行几列
```


```python
len(a)   #数组长度
```


```python
b.ndim   #几维数组
```


```python
e.size   #数组有多少元素
```


```python
b.dtype   #数据类型
```


```python
b.dtype.name   #数据类型的名字
```


```python
b.astype(int)   #数据类型转换
```

## 数组计算

### 算数运算


```python
g = a - b   #减法
```


```python
g
```


```python
np.subtract(a,b)   #减法
```


```python
b + a   #加法
```


```python
np.add(b,a)   #加法
```


```python
a / b   #除法
```


```python
np.divide(a,b)   #除法
```


```python
a * b   #乘法
```


```python
np.multiply(a,b)   #乘法
```


```python
np.exp(b)   #幂
```


```python
np.sqrt(b)   #平方根
```


```python
np.sin(a)   #正弦
```


```python
np.cos(b)   #余弦
```


```python
np.log(a)   #自然对数
```


```python
e.dot(f)   #点积
```

### 比较


```python
a == b   #对比值
```


```python
a < 2   #对比值
```


```python
np.array_equal(a, b)   #对比数组
```

### 聚合函数


```python
a.sum()   #数组汇总
```


```python
a.min()   #数组最小值
```


```python
b.max(axis=0)   #数组最大值，按行
```


```python
b.cumsum(axis=1)   #数组元素的累加值
```


```python
a.mean()   #平均数
```


```python
np.median(b)   #中位数
```


```python
np.corrcoef(a, b)   #相关系数
```


```python
np.std(b)   #标准差
```

## 数组复制

**可以通过copy复制数组**


```python
h = a.view()   #使用同一数据创建数组视图
```


```python
np.copy(a)   #创建数组的副本
```


```python
h = a.copy()   #创建数组的深度拷贝
```

## 数组排序

**通过sort进行数组排序**


```python
a.sort()   #数组排序
```


```python
c.sort(axis=0)   #以轴为依据对数组排序
```

## 子集、切片、索引

### 子集


```python
a[2]   #选择索引2对应的值
```


```python
b[1, 2]   #选择行列index为1和2位置对应的值（等同于b[1][2] ）
```

### 切片


```python
a[0:2]   #选择索引为0与1对应的值
```


```python
b[0:2, 1]   #选择第1列中第0行、第1行的值
```


```python
b[:1]   #选择第0行的所有值（等同于b[0:1,:1] ）
```


```python
c[1,...]   #等同于 [1,:,:]
```


```python
a[ : :-1]   #反转数组a
```


```python
a[a<2]   #选择数组a中所有小于2的值
```


```python
b[[1, 0, 1, 0],[0, 1, 2, 0]]   #选择(1,0),(0,1),(1,2)和(0,0)所对应的值
```


```python
b[[1, 0, 1, 0]][:,[0,1,2,0]]   #选择矩阵的行列子集
```

## 数组操作

### 转置数组


```python
i = np.transpose(b)   #转置数组
```


```python
i.T   #转置数组
```

### 改变数组形状


```python
b.ravel()   #拉平数组
```


```python
g.reshape(3, -2)   #改变数组形状，但不改变数据
```

### 添加或删除值


```python
h.resize((2, 6))   #返回形状为(2,6)的新数组
```


```python
np.append(h, g)   #追加数据
```


```python
np.insert(a, 1, 5)   #插入数据
```


```python
np.delete(a, [1])   #删除数据
```

### 合并数组


```python
np.concatenate((a, d), axis=0)   #拼接数组
```


```python
np.vstack((a, b))   #纵向以行的维度堆叠数组
```


```python
np.r_[e, f]   #纵向以行的维度堆叠数组
```


```python
np.hstack((e, f))   #横向以列的维度堆叠数组
```


```python
np.column_stack((a,d))   #以列的维度创建堆叠数组
```


```python
np.c_[a, d]   #以列的维度创建堆叠数组
```

### 分割数组


```python
np.hsplit(a, 3)   #纵向分割数组为3等份
```


```python
np.vsplit(c,2)   #横向分割数组为2等份
```

## 调用帮助

**通过info函数调用帮助信息**


```python
np.info(np.ndarray.dtype)
```
