
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import data ,filters ,segmentation ,measure ,morphology ,color

# 加载并裁剪硬币图片
img = cv2.imread("picture3.jpg")

# 二值化
image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#thresh =filters.threshold_otsu(image)  # 阈值分割
#bw = morphology.closing(image > thresh, morphology.square(10))  # 闭运算

cleared = image.copy()  # 复制

label_image = measure.label(cleared)  # 连通区域标记
#borders = np.logical_xor(bw, cleared)  # 异或
#label_image[borders] = -1
image_label_overlay = color.label2rgb(label_image, image=image)  # 不同标记用不同颜色显示

fig, (ax1) = plt.subplots(1, figsize=(5, 5))
ax1.imshow(image)
for region in measure.regionprops(label_image):  # 循环得到每一个连通区域属性集

    # 忽略小区域
    if region.area < 100:
        region.area==0
        continue

    # 绘制外包矩形
    minr, minc, maxr, maxc = region.bbox
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor='white', linewidth=1)
    print(minr, minc, maxr, maxc)
    font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
    imgzi = cv2.putText(image, '{} {:.3f}'.format(maxc - minc,  maxr - minr), (420, 50), font, 100, (0, 0, 0), 3)
    ax1.add_patch(rect)
img1=fig.tight_layout()
plt.savefig('kuangjiabiaozhu.jpg')
plt.show()


num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)

#删除小的连通域
# 查看各个返回值
# 连通域数量
print('num_labels = ',num_labels)
# 连通域的信息：对应各个轮廓的x、y、width、height和面积
#print('stats = ',stats)
max=0
for i in range(num_labels-1):

    if stats[i+1][4]>max:
         max=stats[i+1][4]
         print(stats[i+1])
         width=stats[i+1][2]
    else:
        i+=1
print('max=',max)
k=max/width
print(k)
# 连通域的中心点
#print('centroids = ',centroids)

