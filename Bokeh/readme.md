# Bokeh
<table align="left">
  <td>
    <a target="_blank" href="http://nbviewer.ipython.org/github/ShowMeAI-Hub/awesome-AI-cheatsheets/blob/main/Bokeh/Bokeh-cheatsheet-code.ipynb"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" /><br>在nbviewer上查看notebook</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/ShowMeAI-Hub/awesome-AI-cheatsheets/blob/main/Bokeh/Bokeh-cheatsheet-code.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" /><br>在Google Colab运行</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets/tree/main/Bokeh/Bokeh-cheatsheet-code.ipynb"><img src="https://badgen.net/badge/open/github/color=cyan?icon=github" /><br>在Github上查看源代码</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets/tree/main/Bokeh/Bokeh-Cheatsheet.pdf"><img src="https://badgen.net/badge/download/pdf/color=white?icon=github"/><br>下载速查表</a>
  </td>
</table>
<br></br>
<br>

## 说明
**notebook by [韩信子](https://github.com/HanXinzi-AI)@[ShowMeAI](https://github.com/ShowMeAI-Hub)**

更多AI速查表资料请查看[速查表大全](https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets)

欢迎访问[ShowMeAI官网](http://www.showmeai.tech/)获取更多资源。

## Bokeh工具库

Bokeh 是 Python 的交互式可视图库，用于生成在浏览器里显示的大规模数据集高性能可视图。

Bokeh 的中间层通用 bokeh.plotting 界面主要为两个组件：数据与图示符。

使用 bokeh.plotting 界面绘图的基本步骤为：

- Step 1：准备数据（Python列表、Numpy数组、Pandas数据框或其它序列值）
- Step 2：创建图形
- Step 3：为数据添加渲染器，自定义可视化图
- Step 4：指定生成的输出类型
- Step 5：显示视图或保存结果


```python
from bokeh.plotting import figure
from bokeh.io import output_file, show
import warnings
warnings.filterwarnings('ignore')
```


```python
x = [1, 2, 3, 4, 5]   #Step 1
```


```python
y = [6, 7, 2, 4, 5]
```


```python
p = figure(title="simple line example", x_axis_label='x', y_axis_label='y')   #Step 2
```


```python
p.line(x, y, legend_label="Temp.", line_width=2)   #Step 3
```


```python
output_file("lines.html")   #Step 4
```


```python
show(p)   #Step 5
```

## 数据准备

**通常，Bokeh在后台把数据转换为列数据源，不过也可手动转换：**


```python
import numpy as np
```


```python
import pandas as pd
```


```python
df = pd.DataFrame(np.array([[33.9,4,65, 'US'], [32.4,4,66, 'Asia'], [21.4,4,109, 'Europe']]),
                    columns=['mpg','cyl', 'hp', 'origin'],
                    index=['Toyota', 'Fiat', 'Volvo'])
```


```python
from bokeh.models import ColumnDataSource
```


```python
cds_df = ColumnDataSource(df)
```

## 绘图

**figure函数**


```python
from bokeh.plotting import figure
```


```python
p1 = figure(plot_width=300, tools='pan,box_zoom')
```


```python
p2 = figure(plot_width=300, plot_height=300, x_range=(0, 8), y_range=(0, 8))
```


```python
p3 = figure()
```

## 渲染器与自定义可视化

### 图示符

**散点标记**


```python
p1.circle(np.array([1,2,3]), np.array([3,2,1]), fill_color='white')
```


```python
p2.square(np.array([1.5,3.5,5.5]), [1,4,3],  color= 'blue' , size=1)
```

**线型图示符**


```python
p1.line([1,2,3,4], [3,4,5,6], line_width=2)
```


```python
p2.multi_line(pd.DataFrame([[1,2,3],[5,6,7]]), pd.DataFrame([[3,4,5],[3,2,1]]), color="blue")
```

### 自定义图示符

**图示符选择与反选**


```python
p = figure(tools='box_select')
```


```python
p.circle('mpg', 'cyl', source=cds_df, selection_color='red', nonselection_alpha=0.1)
```

**绘图区内部**


```python
from bokeh.models import CategoricalColorMapper
```


```python
color_mapper = CategoricalColorMapper(factors=['US', 'Asia', 'Europe'], palette=['blue', 'red', 'green'])
```


```python
p3.circle('mpg', 'cyl', source=cds_df,color=dict(field='origin', transform=color_mapper), legend_label='Origin')
```

### 图例位置

**绘图区内部**


```python
p.legend.location = 'bottom_left'
```

**绘图区外部**


```python
from bokeh.models import Legend
```


```python
r1 = p2.asterisk(np.array([1,2,3]), np.array([3,2,1]))
```


```python
r2 = p2.line([1,2,3,4], [3,4,5,6])
```


```python
legend = Legend(items=[("One" ,[r1]),("Two",[r2])], location=(0, -30))
```


```python
p.add_layout(legend, 'right')
```

### 图例方向


```python
p.legend.orientation = "horizontal"
```


```python
p.legend.orientation = "vertical"
```

### 图例背景与边框


```python
p.legend.border_line_color = "navy"
```


```python
p.legend.background_fill_color = "white"
```

### 行列布局

**行**


```python
from bokeh.layouts import row
```


```python
layout = row(p1,p2,p3)
```

**列**


```python
from bokeh.layouts import column
```


```python
layout = column(p1,p2,p3)
```

**行列嵌套**


```python
layout = row(column(p1,p2), p3)
```

### 栅格布局


```python
from bokeh.layouts import gridplot
```


```python
row1 = [p1,p2]
```


```python
row2 = [p3]
```


```python
layout = gridplot([[p1,p2],[p3]])
```

### 标签布局


```python
from bokeh.models.widgets import Panel, Tabs
```


```python
tab1 = Panel(child=p1, title="tab1")
```


```python
tab2 = Panel(child=p2, title="tab2")
```


```python
layout = Tabs(tabs=[tab1, tab2])
```

### 链接图

**链接坐标轴**


```python
p2.x_range = p1.x_range
```


```python
p2.y_range = p1.y_range
```

**链接刷**


```python
p4 = figure(plot_width = 100, tools='box_select,lasso_select')
```


```python
p4.circle('mpg', 'cyl', source=cds_df)
```


```python
p5 = figure(plot_width = 200, tools='box_select,lasso_select')
```


```python
p5.circle('mpg', 'hp', source=cds_df)
```


```python
layout = row(p4,p5)
```

## 输出与导出

### Notebook


```python
from bokeh.io import output_notebook, show
```


```python
output_notebook()
```

### HTML

**本地****HTML**


```python
from bokeh.embed import file_html
```


```python
from bokeh.resources import CDN
```


```python
html = file_html(p, CDN, "my_plot")
```


```python
from bokeh.io import output_file, show
```


```python
output_file('my_bar_chart.html', mode='cdn')
```

**组件**


```python
from bokeh.embed import components
```


```python
script, div = components(p)
```

**PNG**


```python
from bokeh.io import export_png
```


```python
# !pip install selenium
export_png(p, filename="plot.png")
```

**SVG**


```python
from bokeh.io import export_svgs
```


```python
p.output_backend = "svg"
```


```python
# !pip install selenium
export_svgs(p, filename="plot.svg")
```

##  显示或保存图形

**show与save函数**


```python
show(p1)
```


```python
save(p1)
```


```python
show(layout)
```


```python
save(layout)
```

 
