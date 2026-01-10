import cv2
import numpy as np

'''代码功能：实现边线搜索并利用贝塞尔拟合法拟合中心线，将中心线绘制在图上
计算track中的左、右点集的方差，作为成员变量存储；计算center中的中心点集方差，作为成员变量存储。并绘制在图像上'''

# 1.定义二维平面坐标类
class Point:
    def __init__(self,row,col):
        self.row=row
        self.col=col

# 2.定义赛道数据类，负责边线、中心线
class Track:
    def __init__(self):
        self.up_chop_rate = 0.3  # 上面要切掉的比例
        self.down_chop_rate = 0.3  # 下面要切掉的比例
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
        h,w= binary.shape
        for row in range(h - 1, 0, -1):
            cols = np.where(binary[row] == 255)[0]
            cols=cols[cols<w]#过滤超出图像范围外的点，防止越界报错
            if len(cols) > 0 and (cols[-1] - cols[0]) >= self.min_valid_width:
                #print(f"起始行右边界：cols[-1]={cols[-1]}, 宽度={cols[-1] - cols[0]}")
                self.start_row, self.start_left, self.start_right = row, cols[0], cols[-1]
                self.LeftPoints.append(Point(row, cols[0]))
                self.RightPoints.append(Point(row, cols[-1]))
                break

    def search_lines(self,binary):
        #利用八邻域搜线法得到左右边线
        if self.start_row is None:#若没有起始行，就直接返回
            return
        h,w=binary.shape  # 得出高宽，防越界
        directions_L = np.array([#右->右上->上->左上->左->左下->下->右下
            [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]
        ])  # 逗号左边是row，因此是y坐标，逗号右边是col，因此是x坐标
        directions_R = np.array([#左->左上->上->右上->右->右下->下->左下
            [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1]
        ])



        #搜左边线
        cen_row,cen_col=self.start_row, self.start_left#确定开始时八邻域的九宫格中心
        while cen_row>0:#一直搜到最上方
            found=False#found是是否找到下一个点的flag
            for dir in range(8):
                #取出变化量数组里的值
                delta_row0,delta_col0=directions_L[dir]
                delta_row1,delta_col1= directions_L[(dir+1)%8]
                #八邻域九宫格里用来观察颜色的两点的坐标
                new_row0=cen_row+delta_row0
                new_col0=cen_col+delta_col0
                new_row1=cen_row+delta_row1
                new_col1=cen_col+delta_col1
                if not (0<=new_row0<h and 0<=new_col0<w and 0<=new_row1<h and 0<=new_col1<w):#防越界
                    continue
                if binary[new_row0,new_col0]==255 and binary[new_row1,new_col1]==0:
                    #print(f"添加右边界点：({new_row0}, {new_col0})")
                    self.LeftPoints.append(Point(new_row0,new_col0))
                    cen_row,cen_col =new_row0,new_col0#更新八邻域的九宫格中心
                    found=True#标记找到
                    break
            if not found:
                break

        # 搜右边线

        cen_row, cen_col = self.start_row, self.start_right
        while cen_row > 0:
            found = False  # found是是否找到下一个点的flag
            for dir in range(8):
                delta_row0, delta_col0 = directions_R[dir]
                delta_row1, delta_col1 = directions_R[(dir + 1) % 8]
                new_row0 = cen_row + delta_row0
                new_col0 = cen_col + delta_col0
                new_row1 = cen_row + delta_row1
                new_col1 = cen_col + delta_col1
                if not(0 <= new_row0 < h and 0 <= new_col0 < w and 0 <= new_row1 < h and 0 <= new_col1 < w):
                    continue
                if binary[new_row0,new_col0]==255 and binary[new_row1,new_col1]==0:
                    self.RightPoints.append(Point(new_row0,new_col0))
                    cen_row, cen_col = new_row0, new_col0
                    found = True  # 标记找到
                    break
            if not found:
                break

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

    def generate_bezier_center(self,h,w):
        #函数功能：生成贝塞尔拟合中心线
        #h和w是图像的高和宽，用来防越界
        self.CenterPoints.clear()
        def get_three_part_points(points):
            #函数功能：返回首点、尾点、三等分点
            #valid_points =[p for p in points if 0 <= p.row < h and 0 <= p.col < w]#过滤掉超出图像范围的点
            for p in points:
                p.row=max(0,min(p.row,h-1))
                p.col=max(0,min(p.col,w-1))
            n = len(points)

            return [
                points[0],points[n//3], points[2*n//3],points[-1]
            ]
        left_fiture=get_three_part_points(self.LeftPoints)
        right_fiture=get_three_part_points(self.RightPoints)
        self.bezier_input=[]
        for l_p,r_p in zip(left_fiture,right_fiture):
            mid_row=(l_p.row+r_p.row)/2
            mid_col=(l_p.col+r_p.col)/2
            #控制点也要防越界，但控制点必须确保为4个，所以不能直接过滤
            mid_row = max(0, min(round(mid_row), h - 1))
            mid_col = max(0, min(round(mid_col), w - 1))
            self.bezier_input.append(Point(round(mid_row),round(mid_col)))
        self.CenterPoints=self.bezier_fit(self.bezier_input)

    def process(self, frame):
        #赛道图像主流程：裁剪->转灰度图->二值化->找起始行->搜索边线->中心线拟合
        self.LeftPoints.clear()
        self.RightPoints.clear()
        self.CenterPoints.clear()
        cropped_frame = self.crop_video_frame(frame)  # 裁剪视频
        h,w=cropped_frame.shape[:2]#传图像大小，在具体函数中用来防越界
        gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)  # 转灰度图
        _, binary_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津法二值化

        self.find_start_line(binary_frame)  # 用二值化图找起始行
        self.search_lines(binary_frame) #搜索边线
        self.generate_bezier_center(h,w)#贝塞尔中心线拟合
        return cropped_frame

#3.定义处理类，负责方差和画图
class Analyser:
    def __init__(self):
        self.sigma_left = 0.0  # Variance of the left boundary line
        self.sigma_right = 0.0  # Variance of the right boundary line
        self.sigma_center = 0.0  # Variance of the center line

    def cal_sigma_of_all(self,tracker):
        def cal_var(points, dim):
            """
                Brief:Calculate the variance of points' coordinates
                Parameters:points
                           dim:dimension,when dim==0,calculate the var of row, when dim==1,calculate the var of col
                Returns:calculated variance
            """
            if len(points) < 2:
                return 0.0
            data = np.array([p.row if dim == 0 else p.col for p in points])
            return np.var(data)
        #You can only calculate the variance by passing an instance of the race track(前面的赛道数据类) data class.
        self.sigma_left=cal_var(tracker.LeftPoints,1)
        self.sigma_right=cal_var(tracker.RightPoints,1)
        self.sigma_center=cal_var(tracker.CenterPoints, 1)

    def draw_all(self,frame,tracker):
        # Brief:Visualize everything
        for p in tracker.LeftPoints:  # 可视化边缘点
            cv2.circle(frame, (p.col, p.row), 2, (0, 255, 0), -1)
        for p in tracker.RightPoints:  # 可视化边缘点
            cv2.circle(frame, (p.col, p.row), 2, (255, 0, 0), -1)
        # for p in tracker.bezier_input:#可视化控制点
        #    cv2.circle(frame, (p.col, p.row), 4, (0, 0, 255), -1)
        for i in range(len(tracker.CenterPoints) - 1):  # 可视化中线
            p1, p2 = tracker.CenterPoints[i], tracker.CenterPoints[i + 1]
            cv2.line(frame, (p1.col, p1.row), (p2.col, p2.row), (0, 0, 255), 2)

        return frame

    def process(self,frame,tracker):
        #赛道数据处理流程：计算方差->将所有东西可视化
        self.cal_sigma_of_all(tracker)
        result = self.draw_all(frame,tracker)# 可视化
        return result


# 4. 视频播放
def play_video(video_path):
    # 函数：调用主流程，并播放视频
    track = Track()# 实例化赛道数据类
    analyser=Analyser()#实例化处理类
    cap=cv2.VideoCapture(video_path)
    while True:
        ret,frame=cap.read()
        # 如果读取失败（视频结束），退出循环
        if not ret:
            break
        cropped_frame=track.process(frame)
        analyser.process(cropped_frame,track)
        '''可视化方差，把可视化方差放在这而不是在“处理类”中的draw_all函数和边线中心线一起画，理由如下：
        边缘点和中心线画在裁剪帧，方差文本画在原始帧上裁剪区外边。这样方差文本和画的线距离较远，文本才不会挡住边缘点和中心线
        
        边缘点和中心线与方差文本在两个不同的图上，会导致无法在同一个图上看结果吗？不会，因为：
        边缘点和中心线画在裁剪帧，裁剪帧和原始帧共享内存，因此边缘点和中心线在裁剪后的区域和原始帧都有画，
        方差文本出现在原始帧上，这样让边缘点和中心线和方差在同一个图都能被看见。
        '''
        text = [
            f"LVar:{analyser.sigma_left:.1f}", f"RVar:{analyser.sigma_right:.1f}", f"CVar:{analyser.sigma_center:.1f}"
        ]
        y = 30
        for txt in text:
            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            y += 30
        cv2.imshow('Video',frame)
        if cv2.waitKey(30)&0xff==ord('q'):
            break
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play_video('sample.avi')