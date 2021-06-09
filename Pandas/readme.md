# Pandas基础
<table align="left">
  <td>
    <a target="_blank" href="http://nbviewer.ipython.org/github/ShowMeAI-Hub/awesome-AI-cheatsheets/blob/main/Pandas/pandas基础-cheatsheet-code.ipynb"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" /><br>在nbviewer上查看notebook</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/ShowMeAI-Hub/awesome-AI-cheatsheets/blob/main/Pandas/pandas基础-cheatsheet-code.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" /><br>在Google Colab运行</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets/Pandas/pandas基础-cheatsheet-code.ipynb"><img src="https://badgen.net/badge/open/github/color=cyan?icon=github" /><br>在Github上查看源代码</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets/Pandas/pandas基础-速查表.pdf"><img src="https://badgen.net/badge/download/pdf/color=white?icon=github"/><br>下载速查表</a>
  </td>
</table>
<br></br>
<br>

## 说明
**notebook by [韩信子](https://github.com/HanXinzi-AI)@[ShowMeAI](https://github.com/ShowMeAI-Hub)**

更多AI速查表资料请查看[速查表大全](https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets)

Pandas是基于Numpy创建的Python库，为Python提供了易于使用的数据结构和数据分析工具。

使用以下语句导入Pandas库：


```python
import pandas as pd
```

## Pandas数据结构

### Series - 序列

存储任意类型数据的一维数组


```python
s = pd.Series([3, -5, 7, 4], index=['a', 'b', 'c', 'd'])
```

### DataFrame - 数据帧


```python
data = {'Country': ['Belgium', 'India', 'Brazil'],
        'Capital': ['Brussels', 'New Delhi', 'Brasília'],
        'Population': [11190846, 1303171035, 207847528]}
```


```python
df = pd.DataFrame(data, columns=['Country', 'Capital', 'Population'])
```


```python
df
```

### 调用帮助


```python
help(pd.Series.loc)
```

## 输入/输出

### 读取/写入CSV


```python
df.to_csv('myDataFrame.csv', index=False)
```


```python
pd.read_csv('myDataFrame.csv', nrows=5)
```

### 读取/写入Excel


```python
df.to_excel('myDataFrame.xlsx', index=False, sheet_name='Sheet1')
```


```python
pd.read_excel('myDataFrame.xlsx')
```


```python
xlsx = pd.ExcelFile('myDataFrame.xlsx')   #读取内含多个表的Excel
```


```python
df = pd.read_excel(xlsx, 'Sheet1')   #读取多表Excel中的Sheet1表
```


```python
df
```

### 读取和写入 SQL 查询及数据库表


```python
from sqlalchemy import create_engine
```


```python
engine = create_engine('sqlite:///:memory:')
```


```python
pd.read_sql("SELECT * FROM my_table;", engine)
```


```python
pd.read_sql_table('my_table', engine)
```


```python
pd.read_sql_query("SELECT * FROM my_table;", engine)
```

**read_sql()是 read_sql_table() 与 read_sql_query()的便捷打包器**


```python
pd.to_sql('myDf', engine)
```

## 筛选数据

### 取值


```python
s['b']   #取序列的值
```


```python
df[1:]   #取数据帧的子集
```

### 选取、布尔索引及设置值

**按位置**


```python
df.iloc[[0], [0]]   #按行与列的位置选择某值
```


```python
df.iat[0, 0]
```

**按标签**


```python
df.loc[[0], ['Country']]   #按行与列的名称选择某值
```


```python
df.at[0, 'Country']   #按行与列的名称选择某值
```

**按标签/位置**


```python
df.loc[2]   #选择某行
```


```python
df.loc[:, 'Capital']   #选择某列
```


```python
df.loc[1, 'Capital'] #按行列取值
```

**布尔索引**


```python
s[~(s > 1)]   #序列S中没有大于1的值
```


```python
s[(s < -1) | (s > 2)]   #序列S中小于-1或大于2的值
```


```python
df[df['Population']>1200000000]   #选择数据帧中Population大于12亿的数据
```


```python
df.loc[df['Population']>1200000000, ['Country', 'Capital']]   #选择数据帧中人口大于12亿的数据'Country'和'Capital'字段
```

**设置值**


```python
s['a'] = 6   #将序列s中索引为a的值设为6
```

## 删除数据

**通过drop函数删除数据**


```python
s.drop(['a', 'c'])   #按索引删除序列的值(axis=0)
```


```python
df.drop('Country', axis=1)   #按列名删除数据帧的列(axis=1)
```

## 排序和排名

**根据索引或者值进行排序**


```python
df.sort_index()   #按索引排序
```


```python
df.sort_values(by='Country')   #按某列的值排序
```


```python
df.rank()   #数据帧排名
```

## 查询信息与计算

### 基本信息


```python
df.shape   #(行,列)
```


```python
df.index   #获取索引
```


```python
df.columns   #获取列名
```


```python
df.info()   #获取数据帧基本信息
```


```python
df.count()   #非Na值的数量
```

### 汇总


```python
df.sum()   #合计
```


```python
df.cumsum()   #累计
```


```python
df['Population'].min()/df['Population'].max()   #最小值除以最大值
```


```python
df['Population'].idxmin()/df['Population'].idxmax()   #索引最小值除以索引最大值
```


```python
df.describe()   #基础统计数据
```


```python
df.mean()   #平均值
```


```python
df.median()   #中位数
```

## 应用函数

**通过apply函数应用变换**


```python
f = lambda x: x*2   #应用匿名函数lambda
```


```python
df.apply(f)   # 应用函数
```


```python
df.applymap(f)   #对每个单元格应用函数
```

## 数据对齐

### 内部数据对齐

**如有不一致的索引，则使用NA值：**


```python
s3 = pd.Series([7, -2, 3], index=['a', 'c', 'd'])
```


```python
s + s3
```

### 使用 Fill 方法运算

**还可以使用 Fill 方法****补齐缺失后再****运算：**


```python
s.add(s3, fill_value=0)
```


```python
s.sub(s3, fill_value=2)
```


```python
s.div(s3, fill_value=4)
```


```python
s.mul(s3, fill_value=3)
```
