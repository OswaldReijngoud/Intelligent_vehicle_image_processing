import cv2
import numpy as np
#代码功能：实现边线搜索并利用中值法拟合中心线，将中心线绘制在图上

# 1. 定义二维平面坐标类
class Point:
    def __init__(self,row,col):
        self.row=row
        self.col=col

# 2. 定义赛道跟踪类，封装一堆赛道跟踪函数
class TrackTracking:
    def __init__(self):
        self.up_chop_rate = 0.3  # 上面要切掉的比例
        self.down_chop_rate = 0.3  # 下面要切掉的比例
        # 存储左右赛道边缘点集
        self.LeftPoints = []
        self.RightPoints = []
        # 起始行标志位（判断是否为最底部有效行）
        self.start_flag = True
        # 临时存储当前行白色色块（赛道区域）
        self._white_block = []
        # 有效色块宽度阈值（过滤噪声）
        self.min_valid_width = 50


        self.start_row = None # 起始行行号（裁剪后图像中的相对行）
        self.start_left = None# 起始行左边缘列号
        self.start_right = None # 起始行右边缘列号

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


    def find_start_line(self, binary):
        self.start_row=self.start_left=self.start_right=None
        h = binary.shape[0]
        for row in range(h - 1, 0, -1):
            cols = np.where(binary[row] == 255)[0]
            if len(cols) > 0 and (cols[-1] - cols[0]) >= self.min_valid_width:
                self.start_row, self.start_left, self.start_right = row, cols[0], cols[-1]
                self.LeftPoints.append(Point(row, cols[0]))
                self.RightPoints.append(Point(row, cols[-1]))
                break

    def search_lines(self,binary):
        #遍历搜线
        h=binary.shape[0]
        for row in range(self.start_row-1,0,-1):
            cols=np.where(binary[row]==255)[0]
            if len(cols)==0:
                continue
            last_left=self.LeftPoints[-1].col
            last_right=self.RightPoints[-1].col
            # 找当前行的左边缘：在白色像素中，找离上一行左边缘最近的点（距离不超过10）
            left_col = next((c for c in cols if abs(c - last_left) <= 10), cols[0])
            # 找当前行的右边缘：在白色像素中，找离上一行右边缘最近的点（距离不超过10）
            right_col = next((c for c in reversed(cols) if abs(c - last_right) <= 10), cols[-1])
            # 把找到的当前行边缘点存起来
            self.LeftPoints.append(Point(row, left_col))
            self.RightPoints.append(Point(row, right_col))
    def draw_all(self,frame):
        #可视化边缘点以及起始行
        for p in self.LeftPoints:
            cv2.circle(frame,(p.col,p.row),2,(0,255,0),-1)
        for p in self.RightPoints:
            cv2.circle(frame,(p.col,p.row),2,(255,0,0),-1)
        #cv2.line(frame,(self.start_left,self.start_row),(self.start_right,self.start_row),(0,0,255),2)
        return frame



    def process(self, frame):
        #边线搜索主流程
        """
               边线搜索完整流程：裁剪→二值化→找起始行→搜线→可视化
               Parameters:
                   frame: 原始视频帧
               Returns:
                   result: 带边缘标记的裁剪后图像
               """
        self.LeftPoints.clear()
        self.RightPoints.clear()
        cropped_frame = self.crop_video_frame(frame)  # 裁剪视频
        gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)  # 转灰度图
        _, binary_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津法二值化

        self.find_start_line(binary_frame)  # 用二值化图找起始行
        self.search_lines(binary_frame) #搜索边线
        result = self.draw_all(cropped_frame)#可视化
        return result




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
        result=tracker.process(frame)
        cv2.imshow('Original',frame)
        cv2.imshow('Track Lines', result)
        if cv2.waitKey(30)&0xff==ord('q'):
            break
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


play_video('sample.avi')