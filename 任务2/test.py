import cv2
import numpy as np


# 1. 定义二维平面坐标类（文档🔶1-32要求，为后续扩展预留）
class Point:
    def __init__(self, row, col):
        self.row = row
        self.col = col


# 2. 定义赛道跟踪类（封装裁剪功能，文档🔶1-12、🔶1-13）
class TrackTracking:
    def __init__(self):
        self.up_chop_rate = 0.1  # 顶部裁剪比例
        self.down_chop_rate = 0.1  # 底部裁剪比例
        self.LeftPoints = []  # 预留：左边缘点集（文档🔶1-25）
        self.RightPoints = []  # 预留：右边缘点集（文档🔶1-25）
        self.start_flag = True  # 预留：起始行标志（文档🔶1-16）
        self._white_block = []  # 预留：临时色块（文档🔶1-18）
        self.min_valid_block_width = 50  # 预留：噪声过滤阈值（文档🔶1-16）

    def crop_video_frame(self, frame):
        """
        裁剪视频帧：去除上下无效区域，保留中间有效部分（文档🔶1-13）
        Parameters:
            frame: 原始视频帧
        Returns:
            cropped_frame: 裁剪后的视频帧
        """
        # 修复：增加帧有效性判断，避免shape调用报错（文档隐含“稳定处理”需求，🔶1-13）
        if frame is None or len(frame.shape) < 2:
            return frame
        height, width = frame.shape[:2]
        # 计算裁剪范围，确保合法（避免越界，文档“适当裁切”要求，🔶1-13）
        start_row = max(0, int(height * self.up_chop_rate))
        end_row = min(height, int(height * (1 - self.down_chop_rate)))
        return frame[start_row:end_row, :]

    def process(self, img_binary):
        # 边线搜索主流程（当前仅裁剪+可视化，暂不实现，预留接口符合文档模块化要求🔶1-31）
        pass  # 修复：添加pass占位，避免空方法语法错误


# 3. 视频播放函数（修复缩进：确保与上方代码块对齐，符合Python语法）
def play_video(video_path):
    tracker = TrackTracking()  # 实例化赛道跟踪类（文档🔶1-32类使用逻辑）
    cap = cv2.VideoCapture(video_path)  # 打开视频（文档🔶1-25提及“使用res中的视频”）

    if not cap.isOpened():
        print(f"无法打开视频:{video_path}")
        return

    while True:
        ret, frame = cap.read()
        # 修复：增加帧有效性判断，避免无效帧导致后续报错（文档“稳定处理”延伸需求）
        if not ret or frame is None:
            break

        cropped_frame = tracker.crop_video_frame(frame)
        # 可视化：符合文档“可视化辅助调试”要求（🔶1-21）
        cv2.imshow('race track', frame)
        cv2.imshow('cropped race track', cropped_frame)

        if cv2.waitKey(30) & 0xff == ord('q'):
            break

    # 释放资源（文档隐含“避免内存泄漏”需求）
    cap.release()
    cv2.destroyAllWindows()


# 调用播放函数（缩进正确：与函数定义对齐）
play_video('sample.avi')