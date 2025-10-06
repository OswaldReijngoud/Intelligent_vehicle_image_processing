import cv2
import numpy as np
from math import factorial
#代码功能：实现边线搜索并利用贝塞尔拟合法拟合中心线，将中心线绘制在图上，中心线附近的粗红点是贝塞尔曲线的控制点

# 1. 定义二维平面坐标类
class Point:
    def __init__(self,row,col):
        self.row=row
        self.col=col

# 2. 定义赛道跟踪类，封装一堆赛道跟踪函数
class TrackTracking:
    def __init__(self):
        self.up_chop_rate = 0.25  # 上面要切掉的比例
        self.down_chop_rate = 0.25  # 下面要切掉的比例
        #左右赛道边缘点集
        self.LeftPoints=[]
        self.RightPoints=[]
        self.CenterPoints=[]#中心线点集
        self.bezier_input=[]#贝塞尔拟合的控制点
        self.start_flag = True# 起始行标志位（判断是否为最底部有效行）
        self._white_block = []# 临时存储赛道区域当前行白色色块
        self.min_valid_width = 50 # 有效色块宽度阈值（过滤噪声）
        self.start_row = None # 起始行行号
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
        """
           找起始行，运行后起始行行号起始行左右两点分别被存储
            Parameters:
               frame:二值化后的图像
            Returns:
               无
                """
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
        #遍历搜线法，得到左右边线
        if self.start_row is None:#要是连起始行都没有，那就别干了
            return
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

    def bezier_fit(self,input_points,dt=0.01):
        """
           贝塞尔曲线核心函数
           根据四个控制点生成三次曲线
            Parameters:
               input_points: 输入特征点
            Returns:
               output: 贝塞尔拟合后的点列表
                """
        output=[]
        #检验输入
        if len(input_points)!=4:
            print("控制点数量错误")
            return output
        t=0
        while t<=1.0+1e-6:#每个t对应图上一个点,dt控制平滑度,+1e-6是为了确保包含终点（t可能在0.99时t+dt=1.0000000001，跳过t=1.0）
            center_row= (1 - t) ** 3 * input_points[0].row + 3 * (1 - t) ** 2 * t * input_points[1].row + 3 * (1 - t) * t ** 2 * input_points[2].row + t ** 3 * input_points[3].row
            center_col= (1 - t) ** 3 * input_points[0].col + 3 * (1 - t) ** 2 * t * input_points[1].col + 3 * (1 - t) * t ** 2 * input_points[2].col + t ** 3 * input_points[3].col
            output.append(Point(round(center_row), round(center_col)))
            t+=dt
        return output


    def generate_bezier_center(self):
        #函数功能：生成贝塞尔拟合中心线
        self.CenterPoints.clear()
        def get_three_part_points(points):
            #函数功能：返回首点、尾点、三等分点
            n=len(points)
            return [
                points[0],points[n//3], points[2*n//3],points[-1]
            ]
        left_fiture=get_three_part_points(self.LeftPoints)
        right_fiture=get_three_part_points(self.RightPoints)
        self.bezier_input=[]
        for l_p,r_p in zip(left_fiture,right_fiture):
            mid_row=(l_p.row+r_p.row)/2
            mid_col=(l_p.col+r_p.col)/2
            self.bezier_input.append(Point(round(mid_row),round(mid_col)))
        self.CenterPoints=self.bezier_fit(self.bezier_input)


    def draw_all(self,frame):
        #函数功能：可视化
        for p in self.LeftPoints:#可视化边缘点
            cv2.circle(frame,(p.col,p.row),2,(0,255,0),-1)
        for p in self.RightPoints:#可视化边缘点
            cv2.circle(frame,(p.col,p.row),2,(255,0,0),-1)
        for p in self.bezier_input:#可视化控制点
            cv2.circle(frame, (p.col, p.row), 4, (0, 0, 255), -1)
        for i in range(len(self.CenterPoints)-1):#可视化中线
            p1,p2=self.CenterPoints[i],self.CenterPoints[i+1]
            cv2.line(frame,(p1.col,p1.row),(p2.col,p2.row),(0,0,255),2)



        return frame



    def process(self, frame):
        #边线搜索主流程
        """
               边线搜索完整流程：裁剪->转灰度图->二值化->找起始行->搜索边线->中心线拟合->可视化
               Parameters:
                   frame: 原始视频帧
               Returns:
                   result: 带边缘标记的裁剪后图像
               """
        self.LeftPoints.clear()
        self.RightPoints.clear()
        self.CenterPoints.clear()
        cropped_frame = self.crop_video_frame(frame)  # 裁剪视频
        gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)  # 转灰度图
        _, binary_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津法二值化

        self.find_start_line(binary_frame)  # 用二值化图找起始行
        self.search_lines(binary_frame) #搜索边线
        self.generate_bezier_center()#贝塞尔中心线拟合
        result = self.draw_all(cropped_frame)#可视化
        return result




# 3. 视频播放
#函数：调用主流程，并播放视频
def play_video(video_path):
    tracker = TrackTracking()# 实例化赛道跟踪类
    cap=cv2.VideoCapture(video_path)
    while True:
        ret,frame=cap.read()
        # 如果读取失败（视频结束），退出循环
        if not ret:
            break
        result=tracker.process(frame)
        cv2.imshow('Vedio',frame)
        #cv2.imshow('Track Lines', result)
        if cv2.waitKey(30)&0xff==ord('q'):
            break
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play_video('sample.avi')