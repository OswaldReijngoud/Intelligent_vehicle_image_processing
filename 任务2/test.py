import cv2
import numpy as np


# 二维平面坐标类（文档🔶2-32）
class Point:
    def __init__(self, row, col):
        self.row = row
        self.col = col


# 赛道跟踪类（含裁剪、起始行、遍历搜线与可视化，文档🔶2-12、🔶2-15、🔶2-17、🔶2-21）
class TrackTracking:
    def __init__(self):
        self.up_chop_rate = 0.05  # 顶部裁剪比例
        self.down_chop_rate = 0.05  # 底部裁剪比例
        self.min_valid_width = 50  # 有效色块宽度阈值（过滤噪声，文档🔶2-16）
        self.start_row = None  # 起始行行号
        self.start_left = None  # 起始行左边缘
        self.start_right = None  # 起始行右边缘
        self.LeftPoints = []  # 左边缘点集（文档🔶2-32）
        self.RightPoints = []  # 右边缘点集（文档🔶2-32）

    # 裁剪视频帧（文档🔶2-12、🔶2-13）
    def crop_frame(self, frame):
        h, w = frame.shape[:2]
        start = int(h * self.up_chop_rate)
        end = int(h * (1 - self.down_chop_rate))
        return frame[start:end, :]

    # 确定起始行（最底部有效赛道行，文档🔶2-15、🔶2-16）
    def find_start_line(self, binary):
        self.start_row = self.start_left = self.start_right = None
        h = binary.shape[0]
        for row in range(h - 1, 0, -1):
            cols = np.where(binary[row] == 255)[0]
            if len(cols) > 0 and (cols[-1] - cols[0]) >= self.min_valid_width:
                self.start_row, self.start_left, self.start_right = row, cols[0], cols[-1]
                self.LeftPoints.append(Point(row, cols[0]))
                self.RightPoints.append(Point(row, cols[-1]))
                break

    # 遍历搜线（从起始行向上搜，文档🔶2-17、🔶2-19）
    def search_lines(self, binary):
        if self.start_row is None:
            return
        h = binary.shape[0]
        # 从起始行上一行向上遍历
        for row in range(self.start_row - 1, 0, -1):
            cols = np.where(binary[row] == 255)[0]
            if len(cols) == 0:
                continue
            # 基于连通性找当前行边缘（参考上一行边缘，文档🔶2-19）
            last_left = self.LeftPoints[-1].col
            last_right = self.RightPoints[-1].col
            # 找靠近上一行左边缘的色块
            left_col = next((c for c in cols if abs(c - last_left) <= 10), cols[0])
            # 找靠近上一行右边缘的色块
            right_col = next((c for c in reversed(cols) if abs(c - last_right) <= 10), cols[-1])
            self.LeftPoints.append(Point(row, left_col))
            self.RightPoints.append(Point(row, right_col))

    # 可视化起始行与边缘点（文档🔶2-21、🔶2-22）
    def draw_all(self, frame):
        # 画边缘点（左绿右蓝，文档🔶2-22）
        for p in self.LeftPoints:
            cv2.circle(frame, (p.col, p.row), 2, (0, 255, 0), -1)
        for p in self.RightPoints:
            cv2.circle(frame, (p.col, p.row), 2, (255, 0, 0), -1)
        # 画起始行（红色）
        if self.start_row is not None:
            cv2.line(frame, (self.start_left, self.start_row), (self.start_right, self.start_row), (0, 0, 255), 2)
        return frame

    def process(self, img_binary):
        pass


# 视频播放与功能调用（文档🔶2-25）
def play_video(video_path):
    tracker = TrackTracking()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 核心流程：裁剪→二值化→找起始行→遍历搜线→可视化
        cropped = tracker.crop_frame(frame)
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tracker.LeftPoints.clear()  # 清空上一帧点集
        tracker.RightPoints.clear()
        tracker.find_start_line(binary)
        tracker.search_lines(binary)
        result = tracker.draw_all(cropped.copy())

        # 显示窗口
        cv2.imshow('Original', frame)
        cv2.imshow('Track Lines', result)

        if cv2.waitKey(30) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


play_video('sample.avi')