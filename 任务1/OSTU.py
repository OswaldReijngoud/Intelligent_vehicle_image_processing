import cv2
import matplotlib.pyplot as plt

# 读取图像，这里以灰度图形式读取
img_path = "5.jpg"  # 请替换为你自己的图片路径
img = cv2.imread(img_path, 0)
if img is None:
    print("The picture cannot be read. Please check if the path is correct!")
    exit(1)

# 使用Otsu算法进行阈值分割
ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original grayscale image'), plt.axis('off')
plt.subplot(122), plt.imshow(thresh, cmap='gray')
plt.title('Binary image after Otsu threshold segmentation'), plt.axis('off')
plt.show()