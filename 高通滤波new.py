
import cv2
import numpy as np
import time
from skimage import morphology
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from skimage import data ,filters ,segmentation ,measure ,morphology ,color
start1=time.perf_counter()#标记开始的时间
img = cv2.imread('101.jpg', cv2.IMREAD_GRAYSCALE)#读取图像，并转化为灰度图
#对图像进行灰度计算
def calcGrayHist(image):
    '''
    统计像素值
    :param image:
    :return:
    '''
    # 灰度图像的高，宽
    rows, cols = img.shape
    # 存储灰度直方图
    grayHist = np.zeros([256], np.uint64)
    for r in range(rows):
        for c in range(cols):
            grayHist[image[r][c]] += 1
    return grayHist
#利用直方图法进行阈值选取
def threshTwoPeaks(image):
    # 计算灰度直方图
    histogram = calcGrayHist(image)
    # 找到灰度直方图的最大峰值对应的灰度值
    maxLoc = np.where(histogram == np.max(histogram))
    firstPeak = maxLoc[0][0]
    # 寻找灰度直方图的第二个峰值对应的灰度值
    measureDists = np.zeros([256], np.float32)
    for k in range(256):
        measureDists[k] = pow(k - firstPeak, 2) * histogram[k]
    maxLoc2 = np.where(measureDists == np.max(measureDists))
    secondPeak = maxLoc2[0][0]
    # 找两个峰值之间的最小值对应的灰度值，作为阈值
    thresh = 0
    if firstPeak > secondPeak:
        temp = histogram[int(secondPeak): int(firstPeak)]
        minLoc = np.where(temp == np.min(temp))
        thresh = secondPeak + minLoc[0][0] + 1
    else:
        temp = histogram[int(firstPeak): int(secondPeak)]
        minLoc = np.where(temp == np.min(temp))
        thresh = firstPeak + minLoc[0][0] + 1
    return thresh
    # 找到阈值，我们进行处理

tt=threshTwoPeaks(img)#tt为通过直方图法选取的阈值
print(tt)
ret1, th1 = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)#ret1、th1为利用OTSU法选取的阈值

#进行高通滤波
def HighPassFilter(image, d):
    #将图像进行filer变换
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        def make_transform_matrix(d):
            transfor_matrix = np.zeros(image.shape)
            center_point = tuple(map(lambda x: (x - 1) / 2, s1.shape))
            for i in range(transfor_matrix.shape[0]):#对每一行每一列进行处理
                for j in range(transfor_matrix.shape[1]):
                    def cal_distance(pa, pb):
                        from math import sqrt
                        dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)#计算每个像素点对于的幅值
                        return dis
                    dis = cal_distance(center_point, (i, j))
                    if dis <= d:
                        transfor_matrix[i, j] = 0
                    else:
                        transfor_matrix[i, j] = 1
            return transfor_matrix
        d_matrix = make_transform_matrix(d)
    #将图像进行filer逆变换
        new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * d_matrix)))
        return new_img

plt.figure(figsize=(50,50))#定位画图的尺寸
#输出灰度化后图像
plt.subplot(331)
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.title('gray')

#输出filer变换后图像
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
s1 = np.log(np.abs(fshift))
plt.subplot(332)
plt.imshow(s1, 'gray')
plt.axis("off")
plt.title('Frequency Domain')

#输出高通滤波后图像
plt.subplot(333)
plt.imshow(HighPassFilter(img, 10), cmap="gray")
plt.axis("off")
plt.title('HighPassFilter')
cv2.imwrite('HighPassFilter.jpg',img)

#输出Canny变换后的图像
Old_image = HighPassFilter(img,10)#高通变换对于阈值的选取
plt.subplot(334)
New_image = Old_image.astype(np.uint8)
#Sobel算子
x = cv2.Sobel(New_image, cv2.CV_16S, 1, 0)
y = cv2.Sobel(New_image, cv2.CV_16S, 0, 1)
absX = cv2.convertScaleAbs(x)  # 转回uint8
absY = cv2.convertScaleAbs(y)
sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
#Canny算子
canny = cv2.Canny(New_image,50, 200)#主要的阈值处理部分（对于Canny算子的检测）
plt.imshow(sobel)
plt.axis("off")
plt.title('Canny')
cv2.imwrite('Canny.jpg',canny)

#输出修改后的连通域信息
kernel = np.ones((3,3),np.uint8)
dilation = cv2.dilate(canny,kernel,iterations = 1)
plt.subplot(335)
plt.imshow(dilation)
plt.axis("off")
plt.title('output')
cv2.imwrite('output1.jpg',dilation)

result = cv2.morphologyEx(dilation,cv2.MORPH_CLOSE,kernel=(3,3),iterations=3)
# 原图取补得到MASK图像
mask = 255 - result

# 构造Marker图像
marker = np.zeros_like(result)
marker[0, :] = 255
marker[-1, :] = 255
marker[:, 0] = 255
marker[:, -1] = 255
marker_0 = marker.copy()

# 形态学重建
SE = cv2.getStructuringElement(shape=cv2.MORPH_CROSS, ksize=(3, 3))
while True:
    marker_pre = marker
    dilation = cv2.dilate(marker, kernel=SE)
    marker = np.min((dilation, mask), axis=0)
    if (marker_pre == marker).all():
        break
dst = 255 - marker
cv2.imwrite('picture2.jpg', dst)
# 显示


num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dst, connectivity=8)

#删除小的连通域
# 查看各个返回值
# 连通域数量
#print('num_labels = ',num_labels)
# 连通域的信息：对应各个轮廓的x、y、width、height和面积
#print('stats = ',stats)
# 连通域的中心点
#print('centroids = ',centroids)
# 每一个像素的标签1、2、3.。。，同一个连通域的标签是一致的
#print('labels = ',labels)




# 不同的连通域赋予不同的颜色
output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
for i in range(1, num_labels):

    mask = labels == i
    output[:, :, 0][mask] = 255
    output[:, :, 1][mask] = 255
    output[:, :, 2][mask] = 255
plt.subplot(336)  # width * height
img1=output
plt.imshow(img1)
plt.axis("off")
plt.title('output')
cv2.imwrite('picture2.jpg',img1)

h, w, _ = img1.shape

GrayImage=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY) #图片灰度化处理

ret,binary = cv2.threshold(GrayImage,40,255,cv2.THRESH_BINARY) #图片二值化,灰度值大于40赋值255，反之0

threshold = h/30 * w/30   #设定阈值

#cv2.fingContours寻找图片轮廓信息
"""提取二值化后图片中的轮廓信息 ，返回值contours存储的即是图片中的轮廓信息，是一个向量，内每个元素保存
了一组由连续的Point点构成的点的集合的向量，每一组Point点集就是一个轮廓，有多少轮廓，向量contours就有
多少元素"""
contours,hierarch=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

for i in range(len(contours)):
    area = cv2.contourArea(contours[i]) #计算轮廓所占面积
    if area < 100:                         #将area小于阈值区域填充背景色，由于OpenCV读出的是BGR值
        cv2.drawContours(img1,[contours[i]],-1, (0,0,0), thickness=-1)     #原始图片背景BGR值(84,1,68)
        continue



plt.subplot(337)  # width * height
plt.imshow(img1)
plt.axis("off")
plt.title('output')
plt.show()
cv2.imwrite('picture3.jpg',img1)

img2=cv2.imread('picture3.jpg')
GrayImage=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(GrayImage, connectivity=8)

#删除小的连通域
# 查看各个返回值
# 连通域数量
#print('num_labels = ',num_labels)
# 连通域的信息：对应各个轮廓的x、y、width、height和面积
#print('stats = ',stats)
# 连通域的中心点
#print('centroids = ',centroids)
# 每一个像素的标签1、2、3.。。，同一个连通域的标签是一致的
#print('labels = ',labels)





#image = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#thresh =filters.threshold_otsu(image)  # 阈值分割
#bw = morphology.closing(image > thresh, morphology.square(3))  # 闭运算
#cleared = image.copy()  # 复制
#label_image = measure.label(cleared)  # 连通区域标记
#borders = np.logical_xor(bw, cleared)  # 异或
#label_image[borders] = -1
#image_label_overlay = color.label2rgb(label_image, image=image)  # 不同标记用不同颜色显示

#fig, (ax1)=plt.subplot(1)
#ax1.imshow(image_label_overlay)
f#or region in measure.regionprops(label_image):  # 循环得到每一个连通区域属性集

    # 忽略小区域
    #if region.area < 1000:
       # continue

    # 绘制外包矩形
   # minr, minc, maxr, maxc = region.bbox
    #rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              #fill=False, edgecolor='white', linewidth=1)
   # ax1.add_patch(rect)
#fig.tight_layout()
#plt.show()
''''''
end1=time.process_time()
print("final is in ",end1-start1)