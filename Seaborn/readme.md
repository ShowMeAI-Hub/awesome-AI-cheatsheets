# Seaborn
<table align="left">
  <td>
    <a target="_blank" href="http://nbviewer.ipython.org/github/ShowMeAI-Hub/awesome-AI-cheatsheets/blob/main/Seaborn/Seaborn-cheatsheet-code.ipynb"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" /><br>在nbviewer上查看notebook</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/ShowMeAI-Hub/awesome-AI-cheatsheets/blob/main/Seaborn/Seaborn-cheatsheet-code.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" /><br>在Google Colab运行</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets/Seaborn/Seaborn-cheatsheet-code.ipynb"><img src="https://badgen.net/badge/open/github/color=cyan?icon=github" /><br>在Github上查看源代码</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets/Seaborn/Seaborn速查表.pdf"><img src="https://badgen.net/badge/download/pdf/color=white?icon=github"/><br>下载速查表</a>
  </td>
</table>
<br></br>
<br>

## 说明
**notebook by [韩信子](https://github.com/HanXinzi-AI)@[ShowMeAI](https://github.com/ShowMeAI-Hub)**

更多AI速查表资料请查看[速查表大全](https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets)

## 导入工具库

用Seaborn绘制统计型数据可视图

Seaborn是基于matplotlib开发的高阶Python数据可视图库，用于绘制优雅、美观的统计图形。

使用下列别名导入该库：


```python
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
```

使用 Seaborn 创建图形的基本步骤：
- Step 1 准备数据
- Step 2 设定画布外观
- Step 3 使用 Seaborn 绘图
- Step 4 自定义图形
- Step 5 展示结果图


```python
tips = sns.load_dataset("tips")   #Step 1
sns.set_style("whitegrid")   #Step 2
g = sns.lmplot(x="tip", y="total_bill", data=tips, aspect=2)    #Step 3
g = (g.set_axis_labels("Tip", "Total bill(USD)"). set(xlim=(0, 10), ylim=(0, 100)))
plt.title("title")   #Step 4
plt.show(g)   #Step 5
```

## 数据准备

**可以是numpy数组和Dataframe等数据格式**


```python
import pandas as pd
import numpy as np
```


```python
uniform_data = np.random.rand(10, 12)
```


```python
data = pd.DataFrame({'x':np.arange(1, 101), 'y':np.random.normal(0, 4, 100)})
```

**Seaborn 提供了内置数据集：**


```python
titanic = sns.load_dataset("titanic")
```


```python
iris = sns.load_dataset("iris")
```

## 画布外观


```python
f, ax = plt.subplots(figsize=(5, 6))   #创建画布与子图
```

### Seaborn 样式


```python
sns.set()    #设置或重置Seaborn默认值
```


```python
sns.set_style("whitegrid")    #设置 matplotlib 参数
```


```python
sns.set_style("ticks", {"xtick.major.size":8, "ytick.major.size":8})    #设置matplotlib参数
```


```python
sns.axes_style("whitegrid")    #返回参数字典或用with设置临时样式
```

### 上下文函数


```python
sns.set_context("talk")   #将上下文设置为"talk"
```


```python
sns.set_context("notebook",
                font_scale=1.5,
                rc={"lines.linewidth":2.5})   #上下文设为"notebook"，缩放字体，覆盖参数映射
```

### 调色板


```python
sns.set_palette("husl", 3)   #定义调色板
```


```python
sns.color_palette("husl")   #使用with临时设置调色板
```


```python
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
```


```python
sns.set_palette(flatui)   #设置调色板
```

## 使用Seaborn绘图

### 坐标轴栅格


```python
g = sns.FacetGrid(titanic, col="survived", row="sex")   #绘制条件关系的子图栅格
g = g.map(plt.hist, "age")
```


```python
sns.factorplot(x="pclass", y="survived", hue="sex", data=titanic)   #在分面栅格上绘制分类图
```


```python
sns.lmplot(x="sepal_width", y="sepal_length", hue="species", data=iris)   #绘制适配分面栅格的数据与回归模型
```


```python
h = sns.PairGrid(iris)   #绘制配对关系的子图栅格
h = h.map(plt.scatter)    #绘制配对的双变量分布
```


```python
sns.pairplot(iris)   #绘制双变量图的边际单变量图栅格
```


```python
i = sns.JointGrid(x="x", y="y", data=data)
i = i.plot(sns.regplot, sns.jointplot)
```


```python
sns.jointplot("sepal_length",  "sepal_width", data=iris, kind='kde')    #绘制双变量分布
```

### 各类图形

**散点图**


```python
sns.stripplot(x="species", y="petal_length", data=iris)   #含分类变量的抖动图
```


```python
sns.swarmplot(x="species", y="petal_length", data=iris)   #不重叠分类蜂群图
```

**条形图**


```python
sns.barplot(x="sex", y="survived", hue="class", data=titanic)   #用散点图示符显示点估计值和置信区间
```

**计数图**


```python
sns.countplot(x="deck", data=titanic, palette="Greens_d")   #显示观测数量
```

**点图**


```python
sns.pointplot(x="class",
                y="survived",
                hue="sex",
                data=titanic,
                palette={"male":"g",
                "female":"m"},
                markers=["^", "o"],
                linestyles=["-", "--"])   #显示点估计和置信区间
```

**箱形图**


```python
sns.boxplot(x="alive", y="age", hue="adult_male", data=titanic)   #箱形图
```


```python
sns.boxplot(data=iris, orient="h")   #使用宽表数据的箱形图
```

**小提琴图**


```python
sns.violinplot(x="age", y="sex", hue="survived", data=titanic)   #小提琴图
```

### 回归图


```python
sns.regplot(x="sepal_width", y="sepal_length", data=iris, ax=ax)   #绘制与线性回归模型拟合的数据
```

### 分布图


```python
plot = sns.distplot(data.y, kde=False, color="b")   #绘制单变量分布
```

### 矩阵图


```python
sns.heatmap(uniform_data, vmin=0, vmax=1)   #热力图
```

## 深度自定义

### Axisgrid对象


```python
g.despine(left=True)   #移除左框
```


```python
g.set_ylabels("Survived")   #设置Y轴标签
```


```python
g.set_xticklabels(rotation=45)   #设置X轴刻度标签
```


```python
g.set_axis_labels("Survived", "Sex")   #设置坐标轴标签
```


```python
h.set(xlim=(0, 5), ylim=(0, 5), xticks=[0, 2.5, 5], yticks=[0, 2.5, 5])   #设置X与Y轴的幅度区间和刻度
```

### 图形


```python
plt.title("A Title")   #添加图形标题
```


```python
plt.ylabel("Survived")   #调整Y轴标签
```


```python
plt.xlabel("Sex")   #调整X轴标签
```


```python
plt.ylim(0, 100)   #调整Y轴幅度区间
```


```python
plt.xlim(0, 10)   #调整X轴幅度区间
```


```python
plt.setp(ax, yticks=[0, 5])   #调整图形属性
```


```python
plt.tight_layout()   #调整子图参数
```

## 显示或保存图形

**show与savefig函数**


```python
plt.show()   #显示图形
```


```python
plt.savefig("foo.png")   #将画布保存为图形
```


```python
plt.savefig("foo.png", transparent=True)   #保存透明画布
```

## 关闭与清除

**绘图关闭与清除操作**


```python
plt.cla()   #清除坐标轴
```


```python
plt.clf()   #清除画布
```


```python
plt.close()   #关闭窗口
```
