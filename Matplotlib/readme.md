# Matplotlib
<table align="left">
  <td>
    <a target="_blank" href="http://nbviewer.ipython.org/github/ShowMeAI-Hub/awesome-AI-cheatsheets/blob/main/Matplotlib/Matplotlib-cheatsheet-code.ipynb"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" />在nbviewer上查看notebook</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/ShowMeAI-Hub/awesome-AI-cheatsheets/blob/main/Matplotlib/Matplotlib-cheatsheet-code.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" />在Google Colab运行</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets/Matplotlib/Matplotlib-cheatsheet-code.ipynb"><img src="https://badgen.net/badge/open/github/color=cyan?icon=github" />在Github上查看源代码</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets/Matplotlib/Matplotlib.pdf"><img src="https://badgen.net/badge/download/pdf/color=white?icon=github"/>下载速查表</a>
  </td>
</table>
<br></br>
<br>

## 说明
**notebook by [韩信子](https://github.com/HanXinzi-AI)@[ShowMeAI](https://github.com/ShowMeAI-Hub)**

更多AI速查表资料请查看[速查表大全](https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets)

Matplotlib是Python的二维绘图库，用于生成符合出版质量或跨平台交互环境的各类图形。

## 准备数据

### 一维数据


```python
import numpy as np
```


```python
x = np.linspace(0, 10, 100)
```


```python
y = np.cos(x)
```


```python
z = np.sin(x)
```

### 二维数据或图片


```python
data = 2 * np.random.random((10, 10))
```


```python
data2 = 3 * np.random.random((10, 10))
```


```python
Y, X = np.mgrid[-3:3:100j, -3:3:100j]
```


```python
U = -1 - X**2 + Y
```


```python
V = 1 + X - Y**2
```

## 绘制图形

### 导入库


```python
import matplotlib.pyplot as plt
%matplotlib inline
```

### 画布


```python
fig = plt.figure()
```


```python
fig2 = plt.figure(figsize=plt.figaspect(2.0))
```

### 坐标轴

**图形是以坐标轴为核心绘制的，大多数情况下，子图就可以满足需求。子图是栅格系统的坐标轴。**


```python
fig.add_axes()
```


```python
ax1 = fig.add_subplot(221)   #row-col-num
```


```python
ax3 = fig.add_subplot(212)
```


```python
fig3, axes = plt.subplots(nrows=2, ncols=2)
```


```python
fig4, axes2 = plt.subplots(ncols=3)
```

## 绘图例程

### 一维数据


```python
fig, ax = plt.subplots()
```


```python
lines = ax.plot(x, y)   #用线或标记连接点
```


```python
ax.scatter(x, y)   #缩放或着色未连接的点
```


```python
axes[0, 0].bar([1, 2, 3], [3, 4, 5])   #绘制柱状图
```


```python
axes[1, 0].barh([0.5, 1, 2.5], [0, 1, 2])   #绘制水平柱状图
```


```python
axes[1, 1].axhline(0.45)   #绘制与轴平行的横线
```


```python
axes[0, 1].axvline(0.65)   #绘制与轴垂直的竖线
```


```python
ax.fill(x, y, color='blue')   #绘制填充多边形
```


```python
ax.fill_between(x, y, color='yellow')   #填充y值和0之间
```

### 向量场


```python
axes[0, 1].arrow(0, 0, 0.5, 0.5)   #为坐标轴添加箭头
```


```python
axes[1, 1].quiver(y, z)   #二维箭头
```


```python
axes[0, 1].streamplot(X, Y, U, V)   #二维箭头
```

### 数据分布


```python
ax1.hist(y)   #直方图
```


```python
ax3.boxplot(y)   #箱形图
```


```python
ax3.violinplot(z)   #小提琴图
```

### 二维数据或图片


```python
fig, ax = plt.subplots()
```


```python
axes2[0].pcolor(data2)   #二维数组伪彩色图
```


```python
axes2[0].pcolormesh(data)   #二维数组等高线伪彩色图
```


```python
CS = plt.contour(Y, X, U)   #等高线图
```


```python
axes2[2]= ax.clabel(CS)   #等高线图标签
```

## 图形解析与工作流

### 图形解析

### 工作流

Matplotlib 绘图的基本步骤：
- Step 1 准备数据
- Step 2 创建图形
- Step 3 绘图
- Step 4 自定义设置
- Step 5 保存图形
- Step 6 显示图形


```python
import matplotlib.pyplot as plt
```


```python
x = [1, 2, 3, 4]   #Step 1
```


```python
y = [10, 20, 25, 30]
```


```python
fig = plt.figure()   #Step 2
```


```python
ax = fig.add_subplot(111)   #Step 3
```


```python
ax.plot(x, y, color='lightblue', linewidth=3)   #Step 3, 4
```


```python
ax.scatter([2, 4, 6], [5, 15, 25], color='darkgreen', marker='^')
```


```python
ax.set_xlim(1, 6.5)
```


```python
plt.savefig('foo.png')   #Step 5
```


```python
plt.show()   #Step 6
```

## 自定义图形

### 颜色、色条与色彩表


```python
plt.plot(x, x, x, [i**2 for i in x], x, [i**3 for i in x])
```


```python
ax.plot(x, y, alpha = 0.4)
```


```python
ax.plot(x, y, c='k')
```

### 标记


```python
fig, ax = plt.subplots()
```


```python
ax.scatter(x, y, marker=".")
```


```python
ax.plot(x, y, marker="o")
```

### 线型


```python
plt.plot(x, y, linewidth=4.0)
```


```python
plt.plot(x, y, ls='solid')
```


```python
plt.plot(x, y, ls='--')
```


```python
plt.plot(x, y, '--', [i**2 for i in x], [i**2 for i in y], '-.')
```


```python
plt.setp(lines, color='r', linewidth=4.0)
```

### 文本与标注


```python
ax.text(1, -2.1, 'Example Graph', style='italic')
```


```python
ax.annotate("Sine",
            xy=(8, 0),
            xycoords='data',
            xytext=(10.5, 0),
            textcoords='data',
            arrowprops=dict(arrowstyle="->",
            connectionstyle="arc3"), )
```

### 数学符号


```python
plt.title(r'$sigma_i=15$', fontsize=20)
```

### 尺寸限制、图例和布局

**尺寸限制与自动调整**


```python
ax.margins(x=0.0, y=0.1)   #添加内边距
```


```python
ax.axis('equal')    #将图形纵横比设置为1
```


```python
ax.set(xlim=[0, 10.5], ylim=[-1.5, 1.5])    #设置x轴与y轴的限制
```


```python
ax.set_xlim(0, 10.5)    #设置x轴的限制
```

**图例**


```python
ax.set(title='An Example Axes', ylabel='Y-Axis',  xlabel='X-Axis')    #设置标题与x、y轴的标签
```


```python
ax.legend(loc='best')    #自动选择最佳的图例位置
```

**标记**


```python
ax.xaxis.set(ticks=range(1, 5), ticklabels=[3, 100, -12, "foo"])    #手动设置X轴刻度
```


```python
ax.tick_params(axis='y', direction='inout', length=10)    #设置Y轴长度与方向
```

**子图间距**


```python
fig3.subplots_adjust(wspace=0.5,
                    hspace=0.3,
                    left=0.125,
                    right=0.9,
                    top=0.9,
                    bottom=0.1)    #调整子图间距
```


```python
fig.tight_layout()   #设置画布的子图布局
```

**坐标轴边线**


```python
ax1.spines['top'].set_visible(False)   #隐藏顶部坐标轴线
```


```python
ax1.spines['bottom'].set_position(('outward', 10))   #设置底部边线的位置为outward
```

## 保存

**savefig函数**


```python
plt.savefig('foo.png')   #保存画布
```


```python
plt.savefig('foo.png', transparent=True)   #保存透明画布
```

##  显示图形

**show函数**


```python
plt.show()
```

## 关闭与清除

**绘图清除与关闭**


```python
plt.cla()   #清除坐标轴
```


```python
plt.clf()   #清除画布
```


```python
plt.close()   #关闭窗口
```
