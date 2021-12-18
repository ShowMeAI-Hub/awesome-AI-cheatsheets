# Keras
<table align="left">
  <td>
    <a target="_blank" href="http://nbviewer.ipython.org/github/ShowMeAI-Hub/awesome-AI-cheatsheets/blob/main/Keras/Keras-cheatsheet-code.ipynb"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" /><br>在nbviewer上查看notebook</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/ShowMeAI-Hub/awesome-AI-cheatsheets/blob/main/Keras/Keras-cheatsheet-code.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" /><br>在Google Colab运行</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets/tree/main/Keras/Keras-cheatsheet-code.ipynb"><img src="https://badgen.net/badge/open/github/color=cyan?icon=github" /><br>在Github上查看源代码</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets/tree/main/Keras/Keras-Cheatsheet.pdf"><img src="https://badgen.net/badge/download/pdf/color=white?icon=github"/><br>下载速查表</a>
  </td>
</table>
<br></br>
<br>

## 说明
**notebook by [韩信子](https://github.com/HanXinzi-AI)@[ShowMeAI](https://github.com/ShowMeAI-Hub)**

更多AI速查表资料请查看[速查表大全](https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets)

欢迎访问[ShowMeAI官网](http://www.showmeai.tech/)获取更多资源。

Keras是强大、易用的深度学习库，基于Theano和TensorFlow提供了高阶神经网络API，用于开发和评估深度学习模型。

## 典型示例


```python
import numpy as np
from keras.models import Sequential     #顺序模型
from keras.layers import Dense    #全连接层
import warnings
warnings.filterwarnings('ignore')

data = np.random.random((1000,100))   #数据
labels = np.random.randint(2,size=(1000,1))  #标签

model = Sequential()   #初始化顺序模型
model.add(Dense(32, activation='relu', input_dim=100))   #添加全连接层
model.add(Dense(1, activation='sigmoid'))   #添加二分类全连接层
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])   #编译模型
model.fit(data,labels,epochs=10,batch_size=32)  #拟合数据
predictions = model.predict(data)   #预估数据
```

## 数据加载

数据要存为 NumPy 数组或数组列表，使用 sklearn.cross_validation 的 train_test_split 模块进行分割将数据分割为训练集与测试集。

### Keras 数据集


```python
from keras.datasets import boston_housing, mnist, cifar10, imdb
```


```python
(x_train,y_train),(x_test,y_test) = mnist.load_data()          #手写数字数据集
```


```python
(x_train2,y_train2),(x_test2,y_test2) = boston_housing.load_data()     #波士顿房价数据集
```


```python
(x_train3,y_train3),(x_test3,y_test3) = cifar10.load_data()   #cifar图像分类数据集
```


```python
(x_train4,y_train4),(x_test4,y_test4) = imdb.load_data(num_words=20000)     #imdb评论数据集
```


```python
num_classes = 10
```

### 其它


```python
import pandas as pd
```


```python
data = pd.read_csv("https://github.com/npradaschnor/Pima-Indians-Diabetes-Dataset/raw/master/diabetes.csv")
```


```python
data.head()
```


```python
X = data.loc[:,'Pregnancies':'Age']
```


```python
y = data.loc[:,'Outcome']
```

## 数据预处理

### 序列填充


```python
from keras.preprocessing import sequence

x_train4 = sequence.pad_sequences(x_train4, maxlen=80)   #填充为固定长度80的序列
x_test4 = sequence.pad_sequences(x_test4, maxlen=80) #填充为固定长度80的序列
```

### 训练与测试集


```python
from sklearn.model_selection import train_test_split

X_train5,X_test5,y_train5,y_test5 = train_test_split(X, y, test_size=0.33, random_state=42)
```

### 独热编码


```python
from keras.utils import to_categorical

Y_train = to_categorical(y_train, num_classes)   #类别标签独热编码转换
Y_test = to_categorical(y_test, num_classes)  #类别标签独热编码转换

Y_train3 = to_categorical(y_train3, num_classes)  #类别标签独热编码转换
Y_test3 = to_categorical(y_test3, num_classes) #类别标签独热编码转换
```

### 标准化/归一化


```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(x_train2)
standardized_X = scaler.transform(x_train2)
standardized_X_test = scaler.transform(x_test2)
```

## 模型架构

### 顺序模型


```python
from keras.models import Sequential

model = Sequential()
model2 = Sequential()
model3 = Sequential()
```

### 多层感知器（MLP）

**二进制分类**


```python
from keras.layers import Dense

model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))  #添加12个神经元的全连接层
model.add(Dense(8,kernel_initializer='uniform',activation='relu')) #添加8个神经元的全连接层
model.add(Dense(1,kernel_initializer='uniform',activation='sigmoid')) #二分类
```

**多级分类**


```python
from keras.layers import Dropout

model.add(Dense(512,activation='relu',input_shape=(784,))) #添加512个神经元的全连接层
model.add(Dropout(0.2))  #添加随机失活层
model.add(Dense(512,activation='relu'))  #添加512个神经元的全连接层
model.add(Dropout(0.2))  #添加随机失活层
model.add(Dense(10,activation='softmax'))  #10分类的全连接层
```

**回归**


```python
model.add(Dense(64,activation='relu',input_dim=x_train2.shape[1]))  #添加64个神经元的全连接层
model.add(Dense(1))
```

### 卷积神经网络（CNN）


```python
from keras.layers import Activation,Conv2D,MaxPooling2D,Flatten

model2.add(Conv2D(32,(3,3),padding='same',input_shape=(28,28,1,)))  #2D卷积层
model2.add(Activation('relu'))  #ReLU激活函数
model2.add(Conv2D(32,(3,3)))  #2D卷积层
model2.add(Activation('relu'))  #ReLU激活函数
model2.add(MaxPooling2D(pool_size=(2,2)))  #2D池化层
model2.add(Dropout(0.25))  #添加随机失活层
model2.add(Conv2D(64,(3,3), padding='same'))  #2D卷积层
model2.add(Activation('relu'))  #ReLU激活函数
model2.add(Conv2D(64,(3, 3)))  #2D卷积层
model2.add(Activation('relu'))  #ReLU激活函数
model2.add(MaxPooling2D(pool_size=(2,2)))  #2D池化层
model2.add(Dropout(0.25))  #添加随机失活层
model2.add(Flatten())  #展平成vector
model2.add(Dense(512)) #全连接层
model2.add(Activation('relu')) #ReLU激活函数
model2.add(Dropout(0.5)) #添加随机失活层
model2.add(Dense(num_classes))  #类别数个神经元的全连接层
model2.add(Activation('softmax'))  #softmax多分类
```

### 递归神经网络（RNN）


```python
from keras.layers import Embedding,LSTM

model3.add(Embedding(20000,128))  #嵌入层
model3.add(LSTM(128,dropout=0.2,recurrent_dropout=0.2))  #LSTM层
model3.add(Dense(1,activation='sigmoid')) #二分类全连接
```

## 审视模型

**获取模型信息**


```python
model.output_shape   #模型输出形状
```


```python
model.summary()   #模型摘要展示
```


```python
model.get_config()   #模型配置
```


```python
model.get_weights()   #列出模型的所有权重张量
```

## 编译模型

**多层感知器：二进制分类**


```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

**多层感知器：多级分类**


```python
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```

**多层感知器：回归**


```python
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
```

**递归神经网络**


```python
model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

## 模型训练

**在数据上拟合**


```python
model3.fit(x_train4, y_train4, batch_size=32, epochs=10, verbose=1, validation_data=(x_test4,y_test4))
```

## 评估模型性能

**在测试集评估**


```python
score = model3.evaluate(x_test4, y_test4, batch_size=32)
```

## 预测

**预测标签与概率**


```python
model3.predict(x_test4, batch_size=32)
```


```python
model3.predict_classes(x_test4,batch_size=32)
```

## 保存/加载模型

**存储与加载模型**


```python
from keras.models import load_model
```


```python
model3.save('model_file.h5')
```


```python
my_model = load_model('model_file.h5')
```

## 模型微调

### 参数优化


```python
from keras.optimizers import RMSprop
```


```python
opt = RMSprop(lr=0.0001, decay=1e-6)
```


```python
model2.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
```

### 早停法


```python
from keras.callbacks import EarlyStopping

early_stopping_monitor = EarlyStopping(patience=2)  #最多等待2轮，如果效果不提升，就停止

model3.fit(x_train4, y_train4, batch_size=32,
            epochs=15,
            validation_data=(x_test4,y_test4),
            callbacks=[early_stopping_monitor])
```
