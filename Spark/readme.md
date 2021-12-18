# Spark RDD 基础
<table align="left">
  <td>
    <a target="_blank" href="http://nbviewer.ipython.org/github/ShowMeAI-Hub/awesome-AI-cheatsheets/blob/main/Spark/Spark_RDD_cheatsheet_code.ipynb"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" /><br>在nbviewer上查看notebook</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/ShowMeAI-Hub/awesome-AI-cheatsheets/blob/main/Spark/Spark_RDD_cheatsheet_code.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" /><br>在Google Colab运行</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets/tree/main/Spark/Spark_RDD_cheatsheet_code.ipynb"><img src="https://badgen.net/badge/open/github/color=cyan?icon=github" /><br>在Github上查看源代码</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets/tree/main/Spark/Spark-RDD-Cheatsheet.pdf"><img src="https://badgen.net/badge/download/pdf/color=white?icon=github"/><br>下载速查表</a>
  </td>
</table>
<br></br>
<br>

## 说明
**notebook by [韩信子](https://github.com/HanXinzi-AI)@[ShowMeAI](https://github.com/ShowMeAI-Hub)**

更多AI速查表资料请查看[速查表大全](https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets)

欢迎访问[ShowMeAI官网](http://www.showmeai.tech/)获取更多资源。

PySpark是Spark的PythonAPI，允许Python调用Spark编程模型。

## 配置spark环境


```python
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q www-us.apache.org/dist/spark/spark-2.4.8/spark-2.4.8-bin-hadoop2.7.tgz  
!tar xf spark-2.4.8-bin-hadoop2.7.tgz
!pip install -q findspark
```


```python
import os
os.environ["JAVA_HOME"]="/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"]="/content/spark-2.4.8-bin-hadoop2.7"
```


```python
import findspark
findspark.init()
```

## 初始化Spark

### SparkContext


```python
from pyspark import SparkContext
```


```python
sc = SparkContext(master = 'local[2]')
```

### SparkContext信息获取


```python
sc.version   #获取SparkContext版本
```


```python
sc.pythonVer   #获取Python版本
```


```python
sc.master   #要连接的MasterURL
```


```python
str(sc.sparkHome)   #Spark在工作节点的安装路径
```


```python
str(sc.sparkUser())   #获取SparkContext的Spark用户名
```


```python
sc.appName   #返回应用名称
```


```python
sc.applicationId   #获取应用程序ID
```


```python
sc.defaultParallelism   #返回默认并行级别
```


```python
sc.defaultMinPartitions   #RDD默认最小分区数
```

### 配置


```python
from pyspark import SparkConf, SparkContext

conf = (SparkConf() \
        .setMaster("local") \
        .setAppName("My app") \
        .set("spark.executor.memory", "1g"))

sc = SparkContext.getOrCreate(conf = conf)
```

### 使用Shell

PySpark Shell已经为SparkContext创建了名为 sc 的变量。


```python
$./bin/spark-shell --master local[2]        #命令行启动spark
```


```python
$./bin/pyspark --master local[4] --py-files code.py     #命令行提交spark脚本任务
```

用 --master 参数设定 Context 连接到哪个Master 务器，通过传递逗号分隔列表至 --py-files 添加 Python.zip、.egg 或 .py文件到 Runtime 路径。

## 加载数据

### 并行集合


```python
rdd = sc.parallelize([('a',7),('a',2),('b',2)])
```


```python
rdd2 = sc.parallelize([('a',2),('d',1),('b',1)])
```


```python
rdd3 = sc.parallelize(range(100))
```


```python
rdd4 = sc.parallelize([("a",["x","y","z"]), ("b",["p", "r"])])
```

### 外部数据

使用textFile()函数从HDFS、本地文件或其它支持Hadoop的文件系统里读取文本文件，或使用wholeTextFiles()函数读取目录里文本文件。


```python
textFile = sc.textFile("/my/directory/*.txt")
# 如果是在google colab中可以运行下方代码
# textFile = sc.textFile("sample_data/california_housing_train.csv")
```


```python
textFile2 = sc.wholeTextFiles("/my/directory/")
# 如果是在google colab中可以运行下方代码
# textFile2 = sc.wholeTextFiles("sample_data/")
```

## 提取RDD信息

### 基础信息


```python
rdd.getNumPartitions()   #列出分区数
```


```python
rdd.count()   #计算RDD实例数量
```


```python
rdd.countByKey()   #按键计算RDD实例数量
```


```python
rdd.countByValue()   #按值计算RDD实例数量
```


```python
rdd.collectAsMap()   #以字典形式返回键值
```


```python
rdd3.sum()   #RDD元素求和
```


```python
sc.parallelize([]).isEmpty()   #检查RDD是否为空
```

### 汇总


```python
rdd3.max()   #RDD元素的最大值
```


```python
rdd3.min()   #RDD元素的最小值
```


```python
rdd3.mean()   #RDD元素的平均值
```


```python
rdd3.stdev()   #RDD元素的标准差
```


```python
rdd3.variance()   #RDD元素的方差
```


```python
rdd3.histogram(3)   #分箱(Bin)生成直方图
```


```python
rdd3.stats()   #综合统计包括：计数、平均值、标准差、最大值和最小值
```

## 应用函数

**map与flatmap函数**


```python
rdd.map(lambda x: x+(x[1],x[0])).collect()   #对每个RDD元素执行函数
```


```python
rdd5=rdd.flatMap(lambda x: x+(x[1],x[0]))   #对每个RDD元素执行函数，并拉平结果
```


```python
rdd5.collect()
```


```python
rdd4.flatMapValues(lambda x: x).collect()   #不改变键，对rdd4的每个键值对执行flatMap函数
```

## 选择数据

### **获取**


```python
rdd.collect()   #返回包含所有RDD元素的列表
```


```python
rdd.filter(lambda x: "a" in x) .collect()   #提取前两个RDD元素
```


```python
rdd.first()   #提取第一个RDD元素
```


```python
rdd5.distinct().collect()   #提取前两个RDD元素
```

### 抽样


```python
rdd3.sample(False, 0.15, 81).collect()   #返回rdd3的采样子集
```

### 筛选


```python
 rdd.filter(lambda x: "a" in x) .collect()   #筛选RDD
```


```python
rdd5.distinct().collect()   #返回RDD里的唯一值
```


```python
rdd.keys().collect()   #返回RDD键值对里的键
```

## 迭代

**foreach函数迭代**


```python
def g(x):
    print(x)
```


```python
rdd.foreach(g)   #为所有RDD应用函数
```

## 改变数据形状

### Reduce操作


```python
rdd.reduceByKey(lambda x,y : x+y).collect()   #合并每个键的RDD值
```


```python
rdd.reduce(lambda a, b: a + b)   #合并RDD的值
```

### 分组


```python
rdd3.groupBy(lambda x: x % 2).mapValues(list).collect()   #返回RDD的分组值
```


```python
rdd.groupByKey().mapValues(list).collect()   #按键分组RDD
```

### 聚合


```python
seqOp = (lambda x,y: (x[0]+y,x[1]+1))
```


```python
combOp = (lambda x,y:(x[0]+y[0],x[1]+y[1]))
```


```python
add = (lambda x,y:x+y)
```


```python
rdd3.aggregate((0,0), seqOp, combOp)   #汇总每个分区里的RDD元素，并输出结果
```


```python
rdd.aggregateByKey((0,0), seqOp, combOp).collect()   #汇总每个RDD的键的值
```


```python
rdd3.fold(0, add)   #汇总每个分区里的RDD元素，并输出结果
```


```python
rdd.foldByKey(0, add).collect()   #合并每个键的值
```


```python
rdd3.keyBy(lambda x: x+x).collect()   #通过执行函数，创建RDD元素的元组
```

## 数学运算

**RDD运算**


```python
rdd.subtract(rdd2).collect()   #返回在rdd2里没有匹配键的rdd键值对
```


```python
rdd2.subtractByKey(rdd).collect()   #返回rdd2里的每个(键，值)对，rdd中没有匹配的键
```


```python
rdd.cartesian(rdd2).collect()   #返回rdd和rdd2的笛卡尔积
```

## 排序

**RDD排序**


```python
rdd2.sortBy(lambda x: x[1]).collect()   #按给定函数排序
```


```python
rdd2.sortByKey() .collect()   #RDD按键排序RDD的键值对
```

## 重分区

**repartition函数**


```python
rdd.repartition(4)   #新建一个含4个分区的RDD
```


```python
rdd.coalesce(1)   #将RDD中的分区数缩减为1个
```

## 保存

**存储RDD到本地或HDFS**


```python
rdd.saveAsTextFile("rdd.txt")
```


```python
rdd.saveAsHadoopFile("hdfs://namenodehost/parent/child", 'org.apache.hadoop.mapred.TextOutputFormat')
```

## 终止SparkContext

**停止SparkContext**


```python
sc.stop()
```

## 执行脚本程序

**提交脚本执行**


```python
$./bin/spark-submit examples/src/main/python/pi.py
```
