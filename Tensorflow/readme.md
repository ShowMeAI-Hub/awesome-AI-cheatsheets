# TensorFlow2

<table align="left">
  <td>
    <a target="_blank" href="http://nbviewer.ipython.org/github/ShowMeAI-Hub/awesome-AI-cheatsheets/blob/main/Tensorflow/Tensorflow-cheatsheet1-code.ipynb"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" /><br>在nbviewer上查看notebook</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/ShowMeAI-Hub/awesome-AI-cheatsheets/blob/main/Tensorflow/Tensorflow-cheatsheet1-code.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" /><br>在Google Colab运行</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets/tree/main/Tensorflow/Tensorflow-cheatsheet1-code.ipynb"><img src="https://badgen.net/badge/open/github/color=cyan?icon=github" /><br>在Github上查看源代码</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets/tree/main/Tensorflow/Tensorflow-cheatsheet1.pdf"><img src="https://badgen.net/badge/download/pdf/color=white?icon=github"/><br>下载速查表</a>
  </td>
</table>
<br></br>
<br>

## 说明
**notebook by [韩信子](https://github.com/HanXinzi-AI)@[ShowMeAI](https://github.com/ShowMeAI-Hub)**

更多AI速查表资料请查看[速查表大全](https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets)

欢迎访问[ShowMeAI官网](http://www.showmeai.tech/)获取更多资源。

TensorFlow2建模应用速查表

## Tensorflow建模手册
TensorFlow是一个由谷歌开发和维护的开源机器学习平台，用于科学计算、神经网络、图像分类、聚类、回归、强化学习、自然语言处理等。它有两个主要组件：graph（计算图）和 session（会话）。它定义操作的方式是为计算构建计算图（graph）；并在会话（session）中执行部分计算图。

### 1.1 安装 installation
要在本地计算机上安装TensorFlow，可以使用pip
```shell
pip install tensorflow
```

安装TensorFlow的GPU版本，需要安装一些其他软件（https://www.tensorflow.org/install/gpu）
```shell
pip install tensorflow-gpu
```

```python
# For google collab users to use 2.x version, add this line
% tensorflow_version 2.x
```
 
### 1.2 导入  importing tensorflow
```python
import tensorflow as tf

# Make sure the version is 2.x 
print(tf.version)
```

### 1.3 张量 tensors
Tensors 是向量和矩阵向更高维度的推广，每个张量都有数据类型和维度。

```python
# 创建一个张量
tf.Variable("sample string", tf.string) 
tf.Variable(32, tf.int16)


# 张量中涉及的维数
tf.rank(tf.Variable([[1, 2], [3, 4]], tf.int16)) 
# Rank - 2


# 更改张量的形状（shape）
t1 = tf.ones(original_shape) 
t2 = tf.reshape(t1, new_shape)
```

张量类型
- 变量
- 常量
- 占位符
- SparseTensor

### 1.4 TF基本的机器学习算法
#### ① 线性回归 Linear Regression
线性回归是机器学习的最基本形式之一，用于预测数值。

```python
# 创建模型
lr = tf.estimator.LinearClassifier(feature_columns)

# 训练模型
lr.train(train_function)

#获取测试集的评估准则计算结果
lr.evaluate(test_function)

# 获取数据集的预测结果
lr.predict(test_function) 
```
 
#### ② 用于分类的深度神经网络 Deep Neural Network for classification
在一些分类任务中，我们的数据和标签并不存在线性关系，使用DNN这种非线性结构可以更好地拟合和建模。

```python
# 创建模型
c = tf.estimator.DNNClassifier(feature_columns, hidden_units, n_classes)

#隐藏单元 hidden_units -[#第一隐藏层的神经元，#第二隐藏层的神经元，…]

# 训练模型
c.train(train_function, steps)
```
  
#### ③ 隐马尔可夫模型 Hidden Markov Model - HMM
隐马尔可夫模型（Hidden Markov Model，HMM）是典型的的概率模型，它使用概率来预测未来事件或状态。
- 了解 HMM 更多应用（https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/HiddenMarkovMode）

```python
# 所以输入 tensorflow_probability 记为 tfp（HMM为统计模型）
tfd = tfp.distributions 

# 创建模型
model = tfd.HiddenMarkovModel(initial_distribution, transition_distribution, observation_distribution, num_steps)
```

在tensorflow的新版本中，如果需要使用session，我们需要使用 tf.compat.v1.Session() 而不是 tf.Session()

## 二、tensorflow.keras
Keras 是一种高级神经网络API，用Python编写，并合并在tensorflow的高版本中。tensorflow.keras对用户友好、可模块化、可扩展性，能够轻松快速地创建原型，支持各种类型的网络（CNN、RNN、transformer等），可以在CPU和GPU上无缝运行。

### 2.1 导入 importing
```python
# 导入Keras
from tensorflow import keras
```
 
### 2.2 Keras 模型的使用
keras模型的基本工作流程：
- 定义模型 Define model
- 编译  Compile
- 拟合 Fit
- 评估 Evaluate
- 预测 Predict

创建模型后，我们使用 model.compile() 编译模型（计算损失函数值和评估准则值），使用 model.fit() 训练模型，使用 model.evaluate() 对测试集的loss和评估准则进行计算，使用 model.predict() 应用模型进行预测。

#### ① 定义模型 define a model

```python
model = keras.Model(inputs, outputs, ...) 
model.summary()

# Groups a linear stack of layers into a model 
keras.Sequential(layers, ...)

# For multi - GPU data parallelism 
keras.utils.multi_gpu_model(model, gpus, ...)
```
 
#### ② 编译模型 compile a model

```python
# 参数设置&编译
model.compile(optimizer, loss, metrics, loss_weights, weighted_metrics, ...)
```
 
#### ③ 训练模型 fit a model

```python
# 固定迭代次数的训练模型
model.fit(x, y, batch_size, epochs, verbose, callbacks, ...)

# 在生成器逐批生成的数据上拟合模型
model.fit_generator()

# 在一批特定训练数据上更新梯度
model.train_on_batch()
```

#### ④ 评估模型 evaluate a model

```python
# 返回测试模式下模型的损失值和度量值
model.evaluate(x, y, batch_size, steps, ...) 

# 在数据生成器上计算模型
model.evaluate_generator(generator, ...)
```
 
#### ⑤ 预测 make predictions

```python
# 从模型生成预测
model.predict()

# 返回单个批次样本的预测
model.predict_on_batch(x)

# 从数据生成器生成输入样本的预测
model.predict_generator(generator, steps, ...) 

# 一个推理步骤的逻辑
model.predict_step(data)
```
 
## 三、可选高级功能

```python
# 输出模型信息
model.summary()

# 在图层堆栈顶部添加图层
model.add(layer)

# 使用名称/索引
model.get_layer()

# 保存模型并在将来随时重新加载
model.save()
keras.models.load_model()

# 要使用的各种数据集
# 例如，keras.datasets.cifar10.load data(), 此模型包含60000个32x32彩色图像，以及10个不同日常对象的6000个图像
keras.datasets

# 从分类分布中提取样本
tf.random.categorical()

# 通过指定要加载的确切文件来加载任何检查点
tf.train.load_checkpoint()
```

### 3.1 keras.layers可用的不同模型层
#### ① 卷积层 convolutional layers 
尺寸（1D、2D、3D）的选择取决于输入的尺寸

```python
# Conv1D通常用于与语音类似的输入信号，Conv2D用于图像，Conv3D通常用于每个时间跨度都有一帧的视频。
layers.Conv1D()
layers.Conv2D()
layers.Conv3D()

# 转置卷积，即与正常卷积方向相反
keras.layers.Conv1DTranspose()

# 零填充层
keras.layers.ZeroPadding1D()

# 裁剪层
keras.layers.Cropping1D()

# 上采样层
keras.layers.UpSampling1D()
```

#### ② 池化层 pooling layers

```python
# 最大池化层
keras.layers.MaxPool1D()
keras.layers.MaxPool2D()
keras.layers.MaxPool3D()

# 平均池化层
keras.layers.AveragePooling1D()

# 全局平均池化操作和全局最大池化操作
keras.layers.GlobalAveragePooling1D()
keras.layers.GlobalMaxPool2D()
```

#### ③ 激活层 activation layers

```python
# Applies an activation function to an output / 激活函数 
keras.layers.Activation('relu')

# Different versions of a Rectified Linear Unit / 不同版本的ReLU激活函数
keras.layers.ReLU()
keras.layers.LeakyReLU()
keras.layers.PReLU()
```

#### ④ dropout 层

```python
# Applies Dropout to the input / Dropout随机失活
keras.layers.Dropout()

# Spatial 1D, 2D, 3D version of Dropout / Dropout的1D、2D、3D版本
Spatial 1D version of Dropout
```

#### ⑤ 嵌入层 embedding layers

```python
# 将索引转换为固定大小的embedding密集向量 
keras.layers.Embedding()
```

#### ⑥ RNN层 recurrent layers

```python
# 长短时记忆层(Long Short Term Memory，LSTM)
keras.layers.LSTM()

# RNN层
keras.layers.RNN()

# GRU层(Gated Recurrent Unit)
keras.layers.GRU()
```

#### ⑦ 展平层 flatten layers

```python
# 展平输入，不影响批次大小
keras.layers.Flatten()
```

#### ⑧ 稠密层/全连接层 dense layers

```python
# 全连接层
keras.layers.Dense(32, activation='relu')
```

#### ⑨ 局部连接层 locally connected layers

```python
# 局部连接层的工作方式与Conv层类似，不同之处在于权重是非共享的，即在输入的每个不同面片上应用一组不同的过滤器
keras.layers.LocallyConnected1D
keras.layers.LocallyConnected2D
```

### 3.2 回调  callbacks

```python
# 回调是在训练过程中的一组函数，可以在训练期间使用回调获取模型内部状态和统计信息的视图 
keras.callbacks

# 回调以某种频率保存Keras模型或模型权重
keras.callbacks.ModelCheckpoint()

# 当监控指标停止改善时停止训练
keras.callbacks.EarlyStopping()
```

### 3.3 预处理 pre-processing
#### ① 图像预处理 image preprocessing

```python
#图像数据实时数据扩充工具集
keras.preprocessing.image

# 加载图像，将图像加载到数组，或将数组保存为图像
keras.preprocessing.image.load_img()
keras.preprocessing.image.img_to_array()
keras.preprocessing.image.array_to_img()

# 通过实时数据增强生成批量张量图像数据
keras.preprocessing.image.ImageGenerator
```

#### ② 文本预处理 text preprocessing

```python
# Text tokenization
keras.preprocessing.text.Tokenizer()

# One-hot将文本编码到单词索引列表中
keras.preprocessing.text.one_hot()
```

#### ③ 序列预处理 sequence preprocessing

```python
# 将序列填充到相同长度
keras.preprocessing.sequence.pad_sequences

# 生成skipgram字对
keras.preprocessing.sequence.skipgrams()
```

### 3.4 预训练模型 pre-trained models
如果数据集与原始数据集分布没有显著差异，则预训练模型的 Transfer learning 和 fine-tuning 可节省时间。tensorflow.keras应用程序是深度学习模型，可与预先训练的权重一起使用。这些模型可用于预测、特征提取和微调。

```python
keras.applications

# 举例：该模型在140万张图像上进行训练，有1000个不同的类
tensorflow.keras.applications.MobileNetV2()
```

## 四、各种超参数 hyperparameters
### 4.1 内置优化器 built-in optimizers

```python
tf.keras.optimizers
```

- SGD（随机梯度下降）
- Adagrad
- Adam
- RMSProp

### 4.2 内置损失函数 built-in loss functions

```python
tf.keras.losses
```
- BinaryCrossentropy（二元交叉熵）
- CategoricalCrossentropy（分类交叉熵）
- MeanAbsoluteError（平均绝对误差）
- MeanSquaredError（均方差） 

### 4.3 内置指标 built-in metrics

```python
tf.keras.metrics
```
- Accuracy（准确度）
- AUC
- False Positive（假阳）
- Precision（精度）
