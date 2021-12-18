# Pytorch

<table align="left">
  <td>
    <a target="_blank" href="http://nbviewer.ipython.org/github/ShowMeAI-Hub/awesome-AI-cheatsheets/blob/main/Pytorch/Pytorch-cheatsheet-code.ipynb"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" /><br>在nbviewer上查看notebook</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/ShowMeAI-Hub/awesome-AI-cheatsheets/blob/main/Pytorch/Pytorch-cheatsheet-code.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" /><br>在Google Colab运行</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets/tree/main/Pytorch/Pytorch-cheatsheet-code.ipynb"><img src="https://badgen.net/badge/open/github/color=cyan?icon=github" /><br>在Github上查看源代码</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets/tree/main/Pytorch/Pytorch-cheatsheet.pdf"><img src="https://badgen.net/badge/download/pdf/color=white?icon=github"/><br>下载速查表</a>
  </td>
</table>
<br></br>
<br>

## 说明
**notebook by [韩信子](https://github.com/HanXinzi-AI)@[ShowMeAI](https://github.com/ShowMeAI-Hub)**

更多AI速查表资料请查看[速查表大全](https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets)

欢迎访问[ShowMeAI官网](http://www.showmeai.tech/)获取更多资源。

## 一、Pytorch概述
PyTorch 是一个开源的机器学习框架，用于数据的表示和处理。Tensor（张量）是一种多维矩阵。PyTorch 中所有神经网络的核心元素是Autograd软件包，该软件包可为所有有关 Tensor 的操作提供自动区分。

```python
# 导入基础工具库
import torch
import torch.nn as nn

# pytorch中用于视觉的数据集、模型与变换
from torchvision import datasets, models, transforms 
import torch.nn.functional as F # Function Collection

torch.randn(*size) # 随机张量
torch.Tensor(L) # 由L初始化的张量
tnsr.view(a, b, ...) # Tensor in size (a, b,...) transform
requires_grad=True # 计算梯度
```

## 二、层 layer

```python
nn.Linear(m, n)
从m到n神经元的全连接层（或稠密层）

nn.Flatten()
将 tensor 的维度减少到1（展平矩阵成向量）

nn.Dropout(p=0.5)
在训练期间，将输入随机设置为0（随机失活）；有助于避免过拟合

nn.Embedding(m, n)
显示n维向量上大小为m的目录中的索引

nn.ConvXd(m, n, s)
输入通道数m，输出通道数n，卷积核大小为s的X维卷积层；X∈{1，2，3}

nn.MaxPoolXd(s)
池化核大小为s的X维池化层；X∈{1，2，3}

nn.BatchNormXd(n)
将具有 n 个特征的X维输入批次标准化；X∈{1，2，3}

nn.RNN/LSTM/GRU
递归网络将一层的神经元连接到同一层或前一层的神经元
```

## 三、加载数据
一条样本可以用从Dataset（(Features, Label)元组构建的序列）继承而来的类来初始化表示。
使用 DataLoader，可以分批完成加载。

数据集通常分为训练数据（通常80%）和测试数据（通常20%）。

```python
from torch. utils. data2
    import Dataset, Tensordataset, Dataloader, random_splits

train_data, test_data = random_split(TensorDataset(inps, tgts), 
                                     [train_size, test_size])
                                     
train_loader = DataLoader(dataset=train_data,
                          batch_size=16,
                          shuffle=True)
```

## 四、激活函数 activation
最常见的激活函数包括ReLU、Sigmoid和Tanh。当然，也还有许多其他激活函数。

```python
nn.ReLU() 是一个新的模块和序列模型，是ReLU函数的新版本。

nn.ReLU() oder F.relu()
输出介于0和∞之间，最常见的激活函数

nn.Sigmoid() oder F.sigmoid()
输出介于0和1之间，通常用于概率

nn.Tanh() oder F.tanh()
介于-1和1之间的问题，通常用于两个类
```

## 五、定义模型
在 Pytorch 中定义神经网络有几种方法，例如使用 nn.Sequential 或 Class 或两者的组合。

```python
# nn.Sequential
model = nn.Sequential(
    nn.Conv2D( , , )
    nn.ReLU()
    nn.MaxPool2D( )
    nn.Flatten()
    nn.Linear( ,  )
)


# class
class Net(nn.Module):
    def  init ():
        super(Net, self). init ()        
            self.conv = nn.Conv2D( ,  ,  )            
            self.pool = nn.MaxPool2D( )
            self.fc = nn.Linear( ,  )

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1,  )
        x = self.fc(x)
        return x

model = Net()
```

## 六、保存或加载模型

```python
model = torch.load('PATH') # Load Model
torch.save(model, 'PATH') # Save Model


model.state_dict()：通常只保存模型参数，而不保存整个模型
torch.save(model.state_dict(), 'params.ckpt') 
model.load_state_dict(torch.load('params.ckpt'))
```


## 七、GPU训练

```python
# 在有CUDA支持的GPU可用的情况下，计算任务将通过 modeled.to(device) 分别发送到各设备。
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# 已传送到 ID 为0的GPU
inputs, labels = data[0].to(device), data[1].to(device)
```

## 八、训练
### 8.1 各种计算误差的方法

```python
# 平均绝对误差（Mean Absolute Error）
nn.L1Loss

# 均方差（Mean Squared Error / L2Loss)
nn.MSELoss

# 交叉熵（Cross-Entropy），包括单标签、不平衡训练数据
nn.CrossEntropyLoss 

# 二元交叉熵（Binary Cross-Entropy），包括多标签或自动编码器
nn.BCELoss 
```

### 8.2 优化算法

torch.optim 优化算法是一种有效的梯度增强算法，它可以有效地提高动态性能。

```python
# Stochastic Gradient Descent
optim.SGD

# Adaptive Moment Estimation 
optim.Adam

# Adaptive Gradient 
optim.Adagrad

# Root Mean Square Prop 
optim.RMSProp



import torch.optim as optim

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 随机梯度下降用于优化
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

# 在训练集上遍历多次
for epoch in range(2):
    model.train() # 启用训练模式
    for i, data in enumerate(train_loader, 0) : 
    
    # 数据以batch个[输入、标签]的形式组织
    inputs, labels = data
    
    # 梯度初始化为0
    optimizer.zero_grad()
    
    # 计算模型输出
    outputs = model(inputs)
    
    # 计算损失并传回
    loss = loss_fn(outputs, labels)
    loss.backward()
    
    # 更新权重/学习率
    optimizer.step()
```

## 九、评估
根据数据指标评估模型的训练结果。评估目标不同，使用的数据指标也随之变化。常用的指标包括：准确度（acurracy）、精确度（precision）、召回率（recall）、F1或BLEU。

```python
# 启用评估模式，此处某些层的行为不同
model.eval() 

# 禁用自动差分/求导，减少内存需求
torch.no_grad() 


correct = 0 # 正确分类的样本数
total = 0 # 总分类样本数

model.eval()
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size 0) # Batch
        correct += (predicted == labels)
                    .sum().item ()
                                                        
print ('Accuracy: %s' % (correct/total))
```
