# Python基础
<table align="left">
  <td>
    <a target="_blank" href="http://nbviewer.ipython.org/github/ShowMeAI-Hub/awesome-AI-cheatsheets/blob/main/Python/Python基础-cheatsheet-code.ipynb"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" /><br>在nbviewer上查看notebook</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/ShowMeAI-Hub/awesome-AI-cheatsheets/blob/main/Python/Python基础-cheatsheet-code.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" /><br>在Google Colab运行</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets/Python/Python基础-cheatsheet-code.ipynb"><img src="https://badgen.net/badge/open/github/color=cyan?icon=github" /><br>在Github上查看源代码</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets/Python/Python基础.pdf"><img src="https://badgen.net/badge/download/pdf/color=white?icon=github"/><br>下载速查表</a>
  </td>
</table>
<br></br>
<br>

## 说明
**notebook by [韩信子](https://github.com/HanXinzi-AI)@[ShowMeAI](https://github.com/ShowMeAI-Hub)**

更多AI速查表资料请查看[速查表大全](https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets)

## 环境配置


```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all" 
```

## Python介绍
Python已经成为最受欢迎的程序设计语言之一，具有简单、易学、易读、易维护、速度快、免费、开源等优点。

## 调用帮助


```python
help(print)
```

## 变量与数据类型

### 变量赋值


```python
x=5
x
```

### 变量计算


```python
x+2    #加
x-2    #减
x*2    #乘
x**2    #幂
x%2    #取余
x/float(2)    #除
```

### 类型与类型转换


```python
str(3.1415)
int('10')
float(3)
bool(4)
```

## 字符串
### 初始化字符串


```python
my_string = 'ShowMeAI-Is-Awesome'  #单引号/双引号/三引号都可以
my_string
```

### 字符串运算


```python
my_string * 2

my_string + 'Innit'

'm' in my_string
```

### 字符串操作
**注意字符串的索引index从0开始**


```python
my_string[3]    #根据索引取字符
my_string[4:9]  #根据索引切片取子串
```

### 字符串方法


```python
my_string.upper()    #字符串字母全部大写
my_string.lower()    #字符串字母全部小写
my_string.count('w')    #统计某字符出现的次数
my_string.replace('e', 'i')   #替换字符
my_string.strip()    #清除左右空格
```

## 列表
### 初始化


```python
a = 'is'
b = 'nice'
my_list = ['my', 'list', a, b]
my_list2 = [[4, 5, 6, 7], [3, 4, 5, 6]]
```

### 选择列表元素
#### 取元素


```python
my_list[1]    #选择索引1对应的值
my_list[-3]    #选择倒数第3个索引对应的值
```

#### 切片


```python
my_list[1:3]     #选取索引1和2对应的值
my_list[1:]    #选取索引1及之后对应的值
my_list[:3]    #选取索引3之前对应的值
my_list[:]    #复制列表
```

#### 子集列表的列表


```python
my_list2[1][0]    #my_list[list][itemOfList]
my_list2[1][:2]
```

### 列表操作


```python
my_list + my_list


my_list * 2
```

### 列表方法


```python
my_list.index(a)    #获取某值的索引
my_list.count(a)    #统计某值出现的次数
my_list.append('!')    #追加某值
my_list.remove('!')    #移除某值
del(my_list[0:1])    #移除某个数据切片
my_list.reverse()    #反转列表
my_list.extend('!')    #添加某值
my_list.pop(-1)    #移除并返回某值
my_list.insert(0, '!')    #插入某值
my_list.sort()    #列表排序
```

## Python库
### 导入库


```python
import numpy
import numpy as np
```

### 导入指定功能


```python
from math import pi
```

## Numpy数组
### Numpy数组创建与操作


```python
my_list = [1, 2, 3, 4]
my_array = np.array(my_list)
my_2darray = np.array([[1, 2, 3], [4, 5, 6]])
```

**选取 Numpy 数组的值**


```python
my_array[1]    #选择索引1对应的值
```


```python
my_array[0:2]    #选择索引0和1对应的值
```


```python
my_2darray[:, 0]    #my_2darray[rows, columns]
```

### Numpy 数组运算


```python
my_array > 3

my_array * 2

my_array + np.array([5, 6, 7, 8])
```

### Numpy 数组函数


```python
my_array.shape    #获取数组形状
np.append(my_array, np.array([9,10,11,12]))    #追加数据
np.insert(my_array, 1, 5)    #插入数据
np.delete(my_array, [1])    #删除数据
np.mean(my_array)    #平均值
np.median(my_array)    #中位数
np.corrcoef(my_array, my_array)    #相关系数
np.std(my_array)    #标准差
```

