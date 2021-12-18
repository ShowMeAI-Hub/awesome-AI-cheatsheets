# OpenCV 4.x(Python版)

<table align="left">
  <td>
    <a target="_blank" href="http://nbviewer.ipython.org/github/ShowMeAI-Hub/awesome-AI-cheatsheets/blob/main/OpenCV/OpenCV-cheatsheet-code.ipynb"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" /><br>在nbviewer上查看notebook</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/ShowMeAI-Hub/awesome-AI-cheatsheets/blob/main/OpenCV/OpenCV-cheatsheet-code.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" /><br>在Google Colab运行</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets/tree/main/OpenCV/OpenCV-cheatsheet-code.ipynb"><img src="https://badgen.net/badge/open/github/color=cyan?icon=github" /><br>在Github上查看源代码</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets/tree/main/OpenCV/OpenCV-cheatsheet.pdf"><img src="https://badgen.net/badge/download/pdf/color=white?icon=github"/><br>下载速查表</a>
  </td>
</table>
<br></br>
<br>

## 说明
**notebook by [韩信子](https://github.com/HanXinzi-AI)@[ShowMeAI](https://github.com/ShowMeAI-Hub)**

更多AI速查表资料请查看[速查表大全](https://github.com/ShowMeAI-Hub/awesome-AI-cheatsheets)

欢迎访问[ShowMeAI官网](http://www.showmeai.tech/)获取更多资源。

## 速查知识手册
OpenCV是一个基于Apache2.0许可（开源）发行的跨平台计算机视觉和机器学习软件库，可以运行在Linux、Windows、Android和Mac OS操作系统上。 OpenCV轻量且高效——由一系列 C 函数和少量 C++ 类构成，同时提供了Python、Ruby、MATLAB等语言的接口，实现了图像处理和计算机视觉方面的很多通用算法。本篇为Python版本的OpenCV使用速查手册。

### 一、输入/输出 I/O

```python
# 将图像加载为BGR（如果为灰度图，则为B=G=R）
i = imread("name.png")

# 按原样加载图像（包括透明度，如果可用）
i = imread("name.png", IMREAD_UNCHANGED)

# 将图像加载为灰度图
i = imread("name.png", IMREAD_GRAYSCALE)

# 显示图像I
imshow("Title", i)

# 保存图像I
imwrite("name.png", i)

# 等待0.5秒按键（0为永远等待）
waitKey(500)

# 释放并关闭所有窗口
destroyAllWindows()
```

### 二、颜色/亮度 color/intensity
#### 2.1 颜色 color

```python
# BGR到灰度图转换 
i_gray = cvtColor(i, COLOR_BGR2GRAY)

# BGR到RGB（对matplotlib有用）
i_rgb = cvtColor(i, COLOR_BGR2RGB)

# 将gray转换为RGB（R=G=B）
i = cvtColor(i, COLOR_GRAY2RGB)
```

#### 2.2 亮度 intensity

```python
# 直方图均衡化
i = equalizeHist(i)

# 将I在0到255之间标准化
i = normalize(i, None, 0, 255, NORM_MINMAX, CV_8U)

# 将I在0到1之间标准化
i = normalize(i, None, 0, 1, NORM_MINMAX, CV_32F)
```

#### 2.3 其他颜色空间 Other useful color spaces

```python
# BGR to HSV （色调、饱和度、值）
COLOR_BGR2HSV

# BGR to Lab（亮度、绿色/洋红色、蓝色/黄色）
COLOR_BGR2LAB

# BGR to Luv（≈Lab，但不同的标准化）
COLOR_BGR2LUV

# BGR to YCrCb（亮度、蓝色亮度、红色亮度）
COLOR_BGR2YCrCb
```

### 三、通道操作 channel manipulation

```python
# 将图像I分割为多个通道
b, g, r = split(i)

# 和上面一样，但I有a个通道
b, g, r, a = split(i)

# 将通道合并到图像中
i = merge((b, g, r))
```

### 四、算术运算 arithmetic operations

```python
# min(I1+I2, 255), 例如饱和添加量
i = add(i1, i2) 

# min(α*I1+β*I2+γ, 255),例如图像混合
i = addWeighted(i1, alpha, i2, beta, gamma)

# max(I1−I2, 0),例如饱和减法
i = subtract(i1, i2)

# |I1−I2|, 例如绝对差
i = absdiff(i1, i2)
注意：其中一个图像可以替换为标量。
```

### 五、逻辑运算 logical operations

```python
# 反转I中的每一位（例如掩码反转）
i = bitwise_not(i)

# I1和I2之间的逻辑and（例如遮罩图像）
i = bitwise_and(i1, i2)

# I1和I2之间的逻辑or（例如，合并2个掩码）
i = bitwise_or(i1, i2)

# I1和I2之间的异或
i = bitwise_xor(i1, i2)
```

### 六、统计数字 statistics

```python
# 每个通道的平均值（即BGRA）
mB, mG, mR, mA = mean(i)

#平均值和SDev p/通道（各3或4行）
ms, sds = meanStdDev(i) 

# 通道c直方图，无掩码，256个分桶（0-255）
h = calcHist([i], [c], None, [256], [0,256])

# 使用通道0和通道1的2D直方图，每个维度的“分辨率”为256
h = calcHist([i], [0,1], None, [256,256], [0,256,0,256])
```

### 七、过滤 filtering

```python
# 带5*5箱式过滤器的过滤器I（即平均过滤器）
i = blur(i, (5, 5))

# 具有5×5高斯分布的滤波器I；自动σs；
i = GaussianBlur(i, (5,5), sigmaX=0, sigmaY=0)

# 高斯模糊
i = GaussianBlur(i, None, sigmaX=2, sigmaY=2)

# 基于互相关的二维核滤波器
i = filter2D(i, -1, k)

# 长度为5的一维高斯核（自动StDev）
kx = getGaussianKernel(5, -1)

# 使用可分离内核的过滤器（相同的输出类型）
i = sepFilter2D(i, -1, kx, ky)

# 尺寸为3的中值滤波器（尺寸≥3)
i = medianBlur(i, 3)

# σr=10，σs=50的双边滤波器，自动大小
i = bilateralFilter(i, -1, 10, 50)
```

### 八、边界 borders
All filtering operations have parameter borderType which can be set to: 
所有过滤操作都有参数borderType，可设置为：

```python
# 具有恒定边界的Pads（需要附加参数值）
BORDER_CONSTANT

# 将第一行/最后一行和列复制到padding上
BORDER_REPLICATE

# 将图像边框反射到padding上
BORDER_REFLECT

# 与前面相同，但不包括边界处的像素（默认值）
BORDER_REFLECT_101

# 环绕图像边框以构建填充
BORDER_WRAP
```

也可以使用自定义宽度添加边框：

```python
# Widths: top, bottom, left, right
i = copyMakeBorder(i, 2, 2, 3, 1, borderType=BORDER_WRAP)
```
### 九、微分算子 differential operators

```python
# x方向的Sobel算子: Ix =(∂/∂x)I
i_x = Sobel(i, CV_32F, 1, 0)

# y方向的Sobel算子：Iy =(∂/∂y)I
i_y = Sobel(i, CV_32F, 0, 1)

# 梯度: ∇I (使用 3*3 SobelSobel): 需要是uint8图片
i_x, i_y = spatialGradient(i, 3)

# Ix, Iy 需要是浮点数类型
m = magnitude(i_x, i_y)

# ||∇I||; θ∈[0, 2π]; angleInDegrees=False; needs float32 Ix, Iy
m, d = cartToPolar(i_x, i_y)

# ∆I, 核大小为5的拉普拉斯算子
l = Laplacian(i, CV_32F, ksize=5)
```

### 十、几何变换 geometric transforms

```python
# 将图像大小调整为宽度*高度
i = resize(i, (width, height))

# 将图像缩放为20%宽度和10%高度
i = resize(i, None, fx=0.2, fy=0.1)

# 返回2*3旋转矩阵M，任意（xc，yc）
M = getRotationMatrix2D((xc, yc), deg, scale) 

# 由3个对应关系得到的一个Affine变换矩阵M
M = getAffineTransform(pts1, pts2)

# 将Affine变换M应用于I，输出大小=（列，行）
i = warpAffine(i, M, (cols, rows))

# 由4个对应关系得到的透视变换矩阵M
M = getPerspectiveTransform(pts1, pts2)

# 透视变换矩阵M≥4对应（最小二乘法）
M, s = findHomography(pts1, pts2)

# 透视变换矩阵M
M, s = findHomography(pts1, pts2, RANSAC)

# 将透视变换M应用于图像I
i = warpPerspective(i, M, (cols, rows))
```

#### 10.1 插值方法 interpolation methods
resize, warpAffine and warpPerspective use bilinear interpolation by default. It can be changed by parameter interpolation for resize, and flags for the others:
默认情况下，调整大小、扭曲仿射和扭曲透视使用双线性插值。可以通过调整大小的参数插值和其他参数的标志进行更改：


```python
#最简单、最快
flags = INTER_NEAREST

# 双线性插值：默认值
flags = INTER_LINEAR

#双立方插值
flags = INTER_CUBIC
```

### 十一、分割 segmentation

```python
# 给定阈值级别t的手动阈值图像I
_, i_t = threshold(i, t, 255, THRESH_BINARY)

#使用Otsu返回阈值级别和阈值图像
t, i_t = threshold(i, 0, 255, THRESH_OTSU)

#具有块大小b和常数c的自适应mean-c
i_t = adaptiveThreshold(i, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, b, c)

#仅使用色调和饱和度将直方图h反向投影到图像i_hsv上；无缩放（即1）
bp = calcBackProject([i_hsv], [0,1], h, [0,180,0,256], 1) 

#返回K簇的标签la和中心ct，10个簇中的最佳紧性cp；1专长/纵队
cp, la, ct = kmeans(feats, K, None, crit, 10, KMEANS_RANDOM_CENTERS)
```

### 十二、特征 features

```python
# 返回Canny边（e是二进制的）
e = Canny(i, tl, th)

# 返回全部（ρ，θ）≥150票，票面分辨率：ρ=1像素，θ=1度
l = HoughLines(e, 1, pi/180, 150)

# 概率霍夫，最小长度=100，最大间隙=20
l = HoughLinesP(e, 1, pi/180, 150, None, 100, 20)

# 返回所有（xc，yc，r），至少有18个投票，bin分辨率=1，param1是Canny的第th个，中心必须至少相距50像素
c = HoughCircles(i, HOUGH_GRADIENT, 1, minDist=50, param1=200, param2=18, minRadius=20, maxRadius=60)

# Harris角点的每像素Rs，窗口=3，Sobel=5，α=0.04
r = cornerHarris(i, 3, 5, 0.04)

# 实例化星形特征检测器
f = FastFeatureDetector_create()

# 检测灰度图像I上的关键点
k = f.detect(i, None)

# 在彩色图像I上绘制关键点k
i_k = drawKeypoints(i, k, None) 

# 实例化一个简短的描述符
d = xfeatures2d.BriefDescriptorExtractor_create()

# 计算I上关键点k的描述符
k, ds = d.compute(i, k)

# 实例化AKAZE检测器/描述符
dd = AKAZE_create()

# 使用x检查和汉明距离实例化brute-force matcher
m = BFMatcher.create(NORM_HAMMING, crossCheck=True) 

# 匹配左右描述符
ms = m.match(ds_l, ds_r)

# 使用matches ms从左图像Il上的左关键点k_l到右Ir绘制匹配
i_m = drawMatches(i_l, k_l, i_r, k_r, ms, None)
```

### 十三、检测 detection

```python
# 将模板T与图像I匹配（标准化X-correl）
ccs = matchTemplate(i, t, TM_CCORR_NORMED)

# ccs中的最小值、最大值和相应坐标
m, M, m_l, M_l = minMaxLoc(ccs)

# 创建“空”级联分类器的实例
c = CascadeClassifier() 

# 从文件加载预先训练的模型；r是真/假
r = c.load("file.xml")

# 每个检测到的对象返回1个元组（x、y、w、h）
objs = c.detectMultiScale(i)
```

### 十四、运动与追踪 motion and tracking

```python
# 返回100个Shi-Tomasi角点，质量(quality)至少为0.5，彼此相距10像素 
pts = goodFeaturesToTrack(i, 100, 0.5, 10)

# 根据I0和I1之间的估计光流确定pts的新位置；如果找到点i的流量，则st[i]为1，否则为0
pts1, st, e = calcOpticalFlowPyrLK(i0, i1, pts0, None)

# 实例化CSRT跟踪器
t = TrackerCSRT_create()

# 使用框架和边界框初始化跟踪器
r = t.init(f, bbox)

# 返回给定下一帧的新边界框
r, bbox = t.update(f)
```

### 十五、图像绘制 drawing on the image

```python
# Line 线
line(i,(x0, y0),(x1, y1), (b, g, r), t)

# Rectangle 长方形
rectangle(i, (x0, y0), (x1, y1), (b, g, r), t)

# Circle 圈
circle(i,(x0, y0), radius, (b, g, r), t)

# 闭合（真）多边形（pts是点数组）
polylines(i,[pts], True, (b, g, r), t)

# 在（x，y）处写入“Hi”，字体大小=1，粗细=2
putText(i, "Hi", (x,y), FONT_HERSHEY_SIMPLEX, 1, (r,g,b), 2, LINE_AA)
```

#### 15.1 参数 parameters

```python
# 原点/起点/左上角（注意它不是（行、列））
(x0, y0)

# 右下角
(x1, y1)

# 线条颜色（uint8）
(b, g, r)

# 线粗度（填充，如果为负值）
t
```
 
### 十六、立体相机标定 calibration and stereo

```python
# 检测角点的二维坐标；i是灰度；r是状态；（n_x，n_y）是校准目标的大小
r, crns = findChessboardCorners(i, (n_x, n_y))

# 应用亚像素精度提高坐标
crnrs = cornerSubPix(i, crns, (5,5), (-1,-1), crit)

# 计算内部（包括畸变系数）和外部（即每个目标视图1 R+T），crns_3D包含1个3D角坐标阵列p/目标视图，crns_2D包含相应的2D角坐标阵列（即1个crns p/目标视图）
r, K, D, ExRs, ExTs = calibrateCamera(crns_3D, crns_2D, i.shape[:2], None, None)

# 在I上绘制corners角点；r是角点检测的状态
drawChessboardCorners(i, (n_x, n_y), crns, r)

# 使用intrinsics取消I的变形
u = undistort(i, K, D)

# 实例化半全局块匹配方法
s = StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=11)

# 实例化一个更简单的块匹配方法
s = StereoBM_create(32, 11) 

# 计算视差图(∝−1个深度图）
d = s.compute(i_L, i_R)
```

### 十七、终止标准 termination criteria
用于K-Means、相机校准（Camera calibration）等


```python
# 20次迭代后停止
crit = (TERM_CRITERIA_MAX_ITER, 20, 0)

# 如果“movement”小于1.0，则停止
crit = (TERM_CRITERIA_EPS, 0, 1.0)

# 上两项任何一项发生就停止
crit = (TERM_CRITERIA_MAX_ITER | TERM_CRITERIA_EPS, 20, 1.0)
```

### 十八、常用知识点总结 useful stuff
#### 18.1 Numpy (np.)

```python
# 阵列I的平均值
m = mean(i) 

# 数组I的加权平均值
m = average(i, weights)

# 阵列/图像I的方差
v = var(i)

# 阵列/图像I的标准偏差
s = std(i)

# numpy柱状图也会返回分桶/bins b
h, b = histogram(i.ravel(), 256, [0,256])

# numpy的饱和/固定功能
i = clip(i, 0, 255)

# 将图像类型转换为float32（与uint8、float64相比）
i = i.astype(np.float32)

# 解决最小二乘问题 1/2*||Ax−b||^2
x, _, _, _ = linalg.lstsq(A, b)

# 并排合并I1和I2
i = hstack((i1, i2))

# 将I1堆叠到I2之上
i = vstack((i1, i2))

# 左右翻转图像
i = fliplr(i)

# 上下翻转图像
i = flipud(i)

# copyMakeBorder的另一种写法（也包括顶部、底部、左侧、右侧）
i = pad(i, ((1, 1), (3, 3)), 'reflect')

# I中最大值的线性指数（即平坦I的指数）
idx = argmax(i)

# 索引相对于i形状的2D坐标
r, c = unravel_index(idx, i.shape)

# 如果数组M中的任何元素大于5，则返回True
b = any(M>5)R

# 如果数组M中的所有元素都大于5，则返回True
b = all(M>5)

# 返回M中的元素大于5的行和列的索引
rows, cols = where(M>5)

# 创建一个包含成对行和列元素的列表
coords = list(zip(rows, cols))

# M的逆
M_inv = linalg.inv(M)

# 将角度转换为弧度
rad = deg2rad(deg)
```

#### 18.2 Matplotlib.pyplot (plt.)

```python
# matplotlib的imshow阻止自动归一化
imshow(i, cmap="gray", vmin=0, vmax=255)

# 在xx、yy位置绘制渐变方向
quiver(xx, yy, i_x, i_y, color="green")

# 将绘图另存为图像
savefig("name.png")
```