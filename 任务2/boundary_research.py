import cv2
import numpy as np

# 1. 定义二维平面坐标类
class Point:
    def __init__(self,row,col):
        self.row=row
        self.col=col

# 2. 定义赛道跟踪类，封装一堆赛道跟踪函数
class TrackTracking:
    def __init__(self):
        self.up_chop_rate = 0.1  # 上面要切掉的比例
        self.down_chop_rate = 0.1  # 下面要切掉的比例
        # 存储左右赛道边缘点集
        self.LeftPoints = []
        self.RightPoints = []
        # 起始行标志位（判断是否为最底部有效行）
        self.start_flag = True
        # 临时存储当前行白色色块（赛道区域）
        self._white_block = []
        # 有效色块宽度阈值（过滤噪声）
        self.min_valid_block_width = 50

    def crop_video_frame(self,frame):
        """
           裁剪视频帧，去除上面和下面部分，保留中间部分
            Parameters:
               frame: 原始视频帧
            Returns:
               cropped_frame: 裁剪后的视频帧
        """
        # 获取视频帧的高度和宽度，之后摄像头图像高度和宽度应该会给，这里就先直接从视频获取了
        height,width=frame.shape[:2]
        start_row=int(height*self.up_chop_rate)
        end_row=int(height*(1-self.down_chop_rate))
        return frame[start_row:end_row,:]

    def process(self, img_binary):
        pass
        #边线搜索主流程


# 3. 视频播放
#函数：播放边线搜索视频
def play_video(video_path):
    tracker = TrackTracking()# 实例化赛道跟踪类
    cap=cv2.VideoCapture(video_path)
    while True:
        ret,frame=cap.read()
        # 如果读取失败（视频结束），退出循环
        if not ret:
            break
        cropped_frame=tracker.crop_video_frame(frame)
        cv2.imshow('race track',frame)
        cv2.imshow('cropped race track', cropped_frame)
        if cv2.waitKey(30)&0xff==ord('q'):
            break
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


play_video('sample.avi')