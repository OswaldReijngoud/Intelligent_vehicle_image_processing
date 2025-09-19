import cv2
import numpy as np
from scipy.stats import mode


def otsu_threshold(hist):
    # 计算图像总像素数（直方图所有灰度级像素数量之和）
    N = np.sum(hist)
    # 计算灰度级0到当前级的累积像素数（用于快速求背景像素占比）
    sumi = np.cumsum(hist)
    # 计算灰度级0到当前级的“灰度值×像素数”累积和（用于快速求背景均值）
    # np.arange(1, len(hist)+1)生成1-256序列，对应灰度值0-255的权重（因hist索引0对应灰度0）
    sumi2 = np.cumsum(hist * np.arange(1, len(hist) + 1))

    # 初始化阈值和最大类间方差（初始值无实际意义，后续会更新）
    threshold = 0
    max_var = 0

    # 遍历所有可能的阈值（1到255，避免阈值为0或255导致某一类无像素）
    for i in range(1, len(hist)):
        # 计算背景类（灰度≤i-1）的像素占比w0
        w0 = sumi[i - 1] / N
        # 计算前景类（灰度≥i）的像素占比w1（两类占比和为1）
        w1 = 1 - w0

        # 计算背景类的平均灰度u0（避免w0=0时除以0）
        u0 = sumi2[i - 1] / N if w0 != 0 else 0
        # 计算前景类的平均灰度u1（sumi2[-1]是所有像素的“灰度值×像素数”总和，减去背景部分得前景部分）
        u1 = (sumi2[-1] - sumi2[i - 1]) / N if w1 != 0 else 0

        # 计算类间方差（Otsu算法核心指标，越大说明前景背景分离越好）
        between_var = w0 * w1 * (u0 - u1) ** 2
        # 若当前类间方差大于历史最大值，更新最大方差和对应阈值
        if between_var > max_var:
            max_var = between_var
            threshold = i

    # 返回最优阈值
    return threshold






# 导入图片（关键步骤）
img_path = "5.jpg"  # 替换为你的图片路径（相对路径或绝对路径）
image = cv2.imread(img_path, 0)  # 以灰度模式读取图片（0表示灰度模式）

# 检查图片是否成功导入
if image is None:
    print("无法读取图片，请检查路径是否正确！")
else:
    # 示例图像二值化
    # 1. 计算灰度图像的直方图：image.flatten()将二维图像转为一维像素数组
    # 256表示直方图分256个区间（对应灰度0-255），[0,256]是像素值的范围
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    # 2. 调用自定义Otsu函数计算最优阈值
    threshold = otsu_threshold(hist)
    # 3. 二值化：像素值大于阈值的设为True（后续可转为255），否则为False（后续可转为0）
    image_binary = image > threshold

    # 可选：显示原图和二值化结果
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
    plt.subplot(122), plt.imshow(image_binary, cmap='gray'), plt.title('Binary Image')
    plt.show()
