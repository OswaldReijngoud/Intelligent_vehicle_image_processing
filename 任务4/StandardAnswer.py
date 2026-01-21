import cv2
import numpy as np
from enum import Enum

'''代码功能：
八邻域边线搜索，贝塞尔拟合中心线，绘制边线、中心线
判断最长白列、丢线
计算并存储两边线、中心线方差，绘制在图像上
'''

'''
Code Structure:
Point:Basic data structure
Track:Track analysis (boundary line/center line), and some simple track character analysis (longest white line/lost line)
Cross:Handle the condition of crossing
Visualize:Visualize special line and some text of data characteristics
Main:Orchestrates the entire code
'''

# 1.Define a 2D coordinate class.
class Point:
    def __init__(self,row,col):
        self.row=row
        self.col=col

# 2.Responsible for longest white line,boundary lines and the centerline.
class Track:
    def __init__(self):
        self.up_chop_rate=0     #Proportion of the top to be cropped
        self.down_chop_rate=0.3   #Proportion of the bottom to be cropped
        #Edge point sets for left and right track boundaries
        self.LeftPoints=[]
        self.RightPoints=[]
        #Lost points
        self.LeftPoints_LostNum=0
        self.RightPoints_LostNum=0
        self.LeftPoints_LostFlag=0#0 not lost;1 lost
        self.RightPoints_LostFlag=0
        self.LostThreshold=0.2

        self.CenterPoints=[]        #Set of points for the centerline
        self.bezier_input=[]        #Control points for Bezier curve fitting

        self.start_flag=True      #Starting row flag (identifies the bottom-most valid row)
        self._white_block=[]      #Temporary storage for white pixel blocks in the current row
        self.min_valid_width=50   #Minimum width threshold for valid blocks (noise filtering)

        self.start_row=None       #Row index of the starting row
        self.start_left=None      #Column index of the left edge in the starting row
        self.start_right=None     #Column index of the right edge in the starting row

        #About the longest white line
        self.Longest_White_Line_Top_Point=None #The peak point of the longest white line
        self.Longest_White_Line_Length=0

        #Corners
        self.LeftDownCorner=None
        self.RightDownCorner=None
        self.LeftUpCorner=None
        self.RightUpCorner=None

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

    def find_Longest_White_Line_Length(self,binary):
        h,w=binary.shape
        best_row,best_col=h-1,w//2 #Initialize the row and col of Longest_White_Line_Top_Point
        step=4  #Search step of col
        for col in range(0,w,step):   #Search from bottom to top
            current_row=0               #If no black pixel is hit, current_row must be zero
            for row in range(h-1,0,-1):
                if binary[row,col]==0:  #Hit black pixel (boundary)
                    current_row=row
                    break
            if current_row<best_row:    #Update the row and col of Longest_White_Line_Top_Point
                best_row=current_row
                best_col=col
        self.Longest_White_Line_Top_Point=Point(best_row,best_col)
        self.Longest_White_Line_Length=h-best_row
        return self.Longest_White_Line_Top_Point,self.Longest_White_Line_Length

    def search_lines(self,binary):
        #利用八邻域搜线法得到左右边线
        if self.start_row is None:#若没有起始行，就直接返回
            return
        h,w=binary.shape  # 得出高宽，防越界
        #Use "Visited" to ensure the search direction is always moving forward and avoids cycling.
        Visited=np.zeros_like(binary,np.uint8)#0 unvisited;1 visited
        directions_L = np.array([#右->右上->上->左上->左->左下->下->右下
            [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]
        ])  # 逗号左边是row，因此是y坐标，逗号右边是col，因此是x坐标
        directions_R = np.array([#左->左上->上->右上->右->右下->下->左下
            [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1]
        ])
        MaxIteration=h*3
        Count=0

        #搜左边线
        cen_row,cen_col=self.start_row, self.start_left#确定开始时八邻域的九宫格中心
        Visited[cen_row,cen_col]=1
        while cen_row>0 and Count<MaxIteration: #search all the way to the top;maximum iteration protection
            Count+=1
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
                #When the first point has not been searched, first point is white and next point is black
                if Visited[new_row0,new_col0]==0 and binary[new_row0,new_col0]==255 and binary[new_row1,new_col1]==0:
                    Visited[new_row0,new_col0]=1
                    #print(f"添加右边界点：({new_row0}, {new_col0})")
                    self.LeftPoints.append(Point(new_row0,new_col0))
                    cen_row,cen_col =new_row0,new_col0#更新八邻域的九宫格中心
                    found=True#标记找到
                    break
            if not found:
                break

        # 搜右边线
        #The `visited` set is NOT reset here, because left and right boundary usually do not overlap
        #Only when the track is too narrow or in some extreme conditions, these two lines will get in touch
        #This method prevents boundary line from crisscrossing(单例标记保护 Visited-set synchronization)
        cen_row, cen_col = self.start_row, self.start_right
        Visited[cen_row, cen_col] = 1
        Count=0#Reset counter
        while cen_row>0 and Count<MaxIteration:
            Count+=1
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
                if Visited[new_row0,new_col0]==0 and binary[new_row0,new_col0]==255 and binary[new_row1,new_col1]==0:
                    Visited[new_row0, new_col0] = 1
                    self.RightPoints.append(Point(new_row0,new_col0))
                    cen_row, cen_col = new_row0, new_col0
                    found = True  # 标记找到
                    break
            if not found:
                break
        #cv2.imshow('mask', Visited * 255)

        #Calculate lost points
        #We expect boundary points num is roughly the image height
        if self.start_row:
            expected_points=self.start_row
        else: expected_points=h

        self.LeftPoints_LostNum=max(0,expected_points-len(self.LeftPoints))
        self.RightPoints_LostNum=max(0,expected_points-len(self.RightPoints))
        if expected_points>0:
            if self.LeftPoints_LostNum/expected_points>self.LostThreshold:
                self.LeftPoints_LostFlag=1
            if self.RightPoints_LostNum/expected_points>self.LostThreshold:
                self.RightPoints_LostFlag=1

    '''def find_corners(self,binary):
        #Under the condition of left and right boudaries are lost:
        #Longest_White_Line_Length>0.6h -> cross
        #Longest_White_Line_Length<0.6h -> sharp corner
        h,w=binary.shape[:2]
        if not self.LeftPoints_LostFlag or not self.RightPoints_LostFlag or self.Longest_White_Line_Length<0.6*h:
            return
        #It makes no sense when lost line is too severe
        if (self.LeftPoints_LostNum>0.9*h or
            self.RightPoints_LostNum>0.9*h or
            len(self.LeftPoints)<20 or
            len(self.RightPoints)<10):
            return
        #find LeftDownCorner
        self.LeftDownCorner=None
        for i in range (5,len(self.LeftPoints)-11,2):#before corner:smooth, after corner:sharp
            if(self.LeftDownCorner is None and  #find the first corner that satisfy the facts
                abs(self.LeftPoints[i].col-self.LeftPoints[i-5].col)<5 and
                abs(self.LeftPoints[i].col-self.LeftPoints[i-10].col)<5 and
                abs(self.LeftPoints[i].col-self.LeftPoints[i+5].col)>5 and
                abs(self.LeftPoints[i].col-self.LeftPoints[i+10].col)>5):
                self.LeftDownCorner=self.LeftPoints[i]
                break
 
         #find RightDownCorner
        self.RightDownCorner=None
        for i in range (5,len(self.RightPoints)-6,2):
            if(self.RightDownCorner is None and
                abs(self.RightPoints[i].col-self.RightPoints[i-3].col)<5 and
                abs(self.RightPoints[i].col-self.RightPoints[i-5].col)<5 and
                abs(self.RightPoints[i].col-self.RightPoints[i+3].col)>5 and
                abs(self.RightPoints[i].col-self.RightPoints[i+5].col)>5):
                self.RightDownCorner=self.RightPoints[i]
                break'''
    # === 将以下两个函数添加到 Track 类中 ===

    def calculate_cos_angle(self, p_pre, p_cur, p_next):
        """
        计算向量 (p_cur - p_pre) 和 (p_next - p_cur) 之间夹角的余弦值
        """
        # 向量 v1: 入射向量
        v1_row = p_cur.row - p_pre.row
        v1_col = p_cur.col - p_pre.col

        # 向量 v2: 出射向量
        v2_row = p_next.row - p_cur.row
        v2_col = p_next.col - p_cur.col

        # 向量点积: x1*x2 + y1*y2
        dot_product = v1_row * v2_row + v1_col * v2_col

        # 向量模长
        norm_v1 = (v1_row**2 + v1_col**2)**0.5
        norm_v2 = (v2_row**2 + v2_col**2)**0.5

        if norm_v1 == 0 or norm_v2 == 0:
            return 1.0 # 避免除零，返回共线

        # 计算余弦值
        cos_theta = dot_product / (norm_v1 * norm_v2)
        return cos_theta

    def find_corners(self, binary):
        """
        使用 K值关联法 寻找下角点 (L形拐点)
        """
        h, w = binary.shape[:2]
        K = 8  # K值步长，建议取 5~10，越图像分辨率大K应该越大

        # 阈值设定：
        # 90度拐角 cos=0; 钝角(120度) cos=-0.5; 直线 cos=1
        # 十字路口通常是L型，角度剧烈，cos值会在 -0.5 到 0.5 之间
        # 考虑到搜线方向，我们寻找cos值最小的点（拐折最剧烈）
        cos_threshold = 0.5

        # --- 寻找左下角点 ---
        self.LeftDownCorner = None
        min_cos_left = 1.0

        # 遍历点集 (去掉头尾的 K 保护区)
        if len(self.LeftPoints) > 2 * K + 1:
            # 这里的遍历方向是从底到顶 (index从小到大)
            # 我们只需要找这一侧的第一个剧烈拐点（即最靠下的角点）
            found_index = -1

            # 限制搜索范围：只在图像的中下部搜索角点，防止误判远处的弯道
            search_limit = min(len(self.LeftPoints) - K, int(len(self.LeftPoints) * 0.8))

            for i in range(K, search_limit):
                p_pre = self.LeftPoints[i - K]
                p_cur = self.LeftPoints[i]
                p_next = self.LeftPoints[i + K]

                cos_val = self.calculate_cos_angle(p_pre, p_cur, p_next)

                # 核心判断：
                # 1. 角度足够剧烈 (cos < 阈值)
                # 2. 必须是向外拐 (十字路口左线是向左拐，col减小) -> 这步可以通过向量叉乘进一步判断，简单起见先只看角度
                # 3. 只有当它比之前的点更像角点时才更新，或者一旦找到符合条件的立刻break(取决于策略)

                if cos_val < cos_threshold:
                    # 这是一个候选角点
                    # 简单的策略：找到第一个满足条件的点即为下角点
                    self.LeftDownCorner = p_cur
                    break

        # --- 寻找右下角点 ---
        self.RightDownCorner = None

        if len(self.RightPoints) > 2 * K + 1:
            search_limit = min(len(self.RightPoints) - K, int(len(self.RightPoints) * 0.8))

            for i in range(K, search_limit):
                p_pre = self.RightPoints[i - K]
                p_cur = self.RightPoints[i]
                p_next = self.RightPoints[i + K]

                cos_val = self.calculate_cos_angle(p_pre, p_cur, p_next)

                if cos_val < cos_threshold:
                    self.RightDownCorner = p_cur
                    break
    # === Add the following two functions to the Track class ===

    def calculate_cos_angle(self, p_pre, p_cur, p_next):
        """
        Calculate the cosine of the angle between vectors (p_cur - p_pre) and (p_next - p_cur)
        """
        # Vector v1: Incoming vector
        v1_row = p_cur.row - p_pre.row
        v1_col = p_cur.col - p_pre.col

        # Vector v2: Outgoing vector
        v2_row = p_next.row - p_cur.row
        v2_col = p_next.col - p_cur.col

        # Dot product: x1*x2 + y1*y2
        dot_product = v1_row * v2_row + v1_col * v2_col

        # Vector magnitude (norm)
        norm_v1 = (v1_row**2 + v1_col**2)**0.5
        norm_v2 = (v2_row**2 + v2_col**2)**0.5

        if norm_v1 == 0 or norm_v2 == 0:
            return 1.0 # Avoid division by zero, return collinear

        # Calculate cosine value
        cos_theta = dot_product / (norm_v1 * norm_v2)
        return cos_theta

    def find_corners(self, binary):
        """
        Find the lower corner (L-shaped inflection point) using the K-value correlation method
        """
        h, w = binary.shape[:2]
        K = 8  # Step size K, recommended 5~10; larger K for higher image resolution

        # Threshold setting:
        # 90-degree corner cos=0; obtuse angle (120 deg) cos=-0.5; straight line cos=1
        # Crossroads are usually L-shaped with sharp angles; cos value is between -0.5 and 0.5
        # Considering search direction, find the point with minimum cos value (sharpest turn)
        cos_threshold = 0.5

        # --- Find Left Down Corner ---
        self.LeftDownCorner = None
        min_cos_left = 1.0

        # Iterate through points (exclude K points padding at ends)
        if len(self.LeftPoints) > 2 * K + 1:
            # Traversal direction is bottom to top (index small to large)
            # We only need the first sharp inflection point on this side (the lowest corner)
            found_index = -1

            # Limit search range: search only in middle-lower part to avoid mistaking distant curves
            search_limit = min(len(self.LeftPoints) - K, int(len(self.LeftPoints) * 0.8))

            for i in range(K, search_limit):
                p_pre = self.LeftPoints[i - K]
                p_cur = self.LeftPoints[i]
                p_next = self.LeftPoints[i + K]

                cos_val = self.calculate_cos_angle(p_pre, p_cur, p_next)

                # Core judgment:
                # 1. Angle is sharp enough (cos < threshold)
                # 2. Must turn outward (left line turns left, col decreases) -> can check cross product, checking angle only for simplicity
                # 3. Update only if it's more like a corner, or break immediately once found (depends on strategy)

                if cos_val < cos_threshold:
                    # This is a candidate corner
                    # Simple strategy: the first point meeting criteria is the lower corner
                    self.LeftDownCorner = p_cur
                    break

        # --- Find Right Down Corner ---
        self.RightDownCorner = None

        if len(self.RightPoints) > 2 * K + 1:
            search_limit = min(len(self.RightPoints) - K, int(len(self.RightPoints) * 0.8))

            for i in range(K, search_limit):
                p_pre = self.RightPoints[i - K]
                p_cur = self.RightPoints[i]
                p_next = self.RightPoints[i + K]

                cos_val = self.calculate_cos_angle(p_pre, p_cur, p_next)

                if cos_val < cos_threshold:
                    self.RightDownCorner = p_cur
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
        if len(self.LeftPoints)<1 or len(self.RightPoints)<1:return
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
        left_feature=get_three_part_points(self.LeftPoints)
        right_feature=get_three_part_points(self.RightPoints)
        self.bezier_input=[]
        for l_p,r_p in zip(left_feature, right_feature):
            mid_row=(l_p.row+r_p.row)/2
            mid_col=(l_p.col+r_p.col)/2
            #控制点也要防越界，但控制点必须确保为4个，所以不能直接过滤
            mid_row = max(0, min(round(mid_row), h - 1))
            mid_col = max(0, min(round(mid_col), w - 1))
            self.bezier_input.append(Point(round(mid_row),round(mid_col)))
        self.CenterPoints=self.bezier_fit(self.bezier_input)

    def process(self, frame):
        #赛道图像主流程：裁剪->转灰度图->二值化->找最长白列->找起始行->搜索边线->找角点(如有)->中心线拟合
        self.LeftPoints.clear()
        self.RightPoints.clear()
        self.CenterPoints.clear()
        self.Longest_White_Line_Length=0
        self.Longest_White_Line_Top_Point=None
        self.LeftPoints_LostFlag = 0
        self.RightPoints_LostFlag=0
        cropped_frame = self.crop_video_frame(frame)  # 裁剪视频
        h,w=cropped_frame.shape[:2]#传图像大小，在具体函数中用来防越界
        gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)  # 转灰度图
        _, binary_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津法二值化

        self.find_Longest_White_Line_Length(binary_frame)   #Find Longest_White_Line_Length and the top point of Longest_White_Line
        self.find_start_line(binary_frame)                  #用二值化图找起始行
        self.search_lines(binary_frame)                     #搜索边线
        self.find_corners(binary_frame)  #Find Corners
        self.generate_bezier_center(h,w)              #贝塞尔中心线拟合
        return cropped_frame

#3.Responsible for variance calculation and visualization.
class Analyse:
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

    def process(self,tracker):
        #赛道数据处理流程：计算方差->将所有东西可视化
        self.cal_sigma_of_all(tracker)

#4.Responsible for the crossroad
class Cross:
    def process(self):
        pass

#5.Responsible for visualize everything
class Visualize:
    def draw_points(self,frame,tracker,crosser):
        # Brief:Visualize everything
        h, w = frame.shape[:2]
        # Draw the longest white line
        if tracker.Longest_White_Line_Top_Point is not None:
            cv2.line(frame,
                     (tracker.Longest_White_Line_Top_Point.col, h - 1),
                     (tracker.Longest_White_Line_Top_Point.col, tracker.Longest_White_Line_Top_Point.row),
                     (255, 0, 255),
                     2)
        # 可视化边缘点
        for p in tracker.LeftPoints:
            cv2.circle(frame, (p.col, p.row), 2, (0, 255, 0), -1)
        for p in tracker.RightPoints:
            cv2.circle(frame, (p.col, p.row), 2, (255, 0, 0), -1)
        '''可视化中心线控制点
        for p in tracker.bezier_input:
           cv2.circle(frame, (p.col, p.row), 4, (0, 0, 255), -1)'''
        # 可视化中线
        for i in range(len(tracker.CenterPoints) - 1):
            p1, p2 = tracker.CenterPoints[i], tracker.CenterPoints[i + 1]
            cv2.line(frame, (p1.col, p1.row), (p2.col, p2.row), (0, 0, 255), 2)

        return frame
    def draw_text(self,frame,tracker,analyser,crosser):
        #Visualize data analysis
        font_scale=0.3
        font_thickness=1
        text = [
            f"LVar:{analyser.sigma_left:.1f}",
            f"RVar:{analyser.sigma_right:.1f}",
            f"CVar:{analyser.sigma_center:.1f}",
            f"LLostFlag:{tracker.LeftPoints_LostFlag:d}",
            f"RLostFlag:{tracker.RightPoints_LostFlag:d}"
        ]
        y = 30
        for txt in text:
            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 255), font_thickness)
            y += 30
        #Draw corners(Big yellow dots with red boarders)
        Corners=[tracker.LeftDownCorner,tracker.RightDownCorner,tracker.LeftUpCorner,tracker.RightUpCorner]
        for p in Corners:
            if p is not None:
                cv2.circle(frame,(p.col,p.row),6,(0,255,255),-1)
                cv2.circle(frame,(p.col,p.row),8,(0,0,255),2)
        return frame
    def process(self,frame,tracker,analyser,crosser):
        self.draw_points(frame, tracker,crosser)
        self.draw_text(frame,tracker,analyser,crosser)
        return frame

#Orchestrator
class Main:
    def __init__(self,video_path):
        self.cap=cv2.VideoCapture(video_path)
        self.tracker=Track()  # 实例化赛道数据类
        self.analyser=Analyse()  # 实例化处理类
        self.visualizer=Visualize()
        self.crosser=Cross()
    def run(self):
        # 函数：调用主流程，播放视频
        while True:
            ret,frame=self.cap.read()
            # 如果读取失败（视频结束），退出循环
            if not ret:
                break
            cropped_frame=self.tracker.process(frame)
            self.analyser.process(self.tracker)
            #visualizer must use frame after crop, or we must handle coordinate offset
            self.visualizer.process(cropped_frame,self.tracker,self.analyser,self.crosser)
            cv2.imshow('Video',frame)
            if cv2.waitKey(30) & 0xff == ord('q'):
                break
        # 释放资源
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app=Main('cross3.mp4')
    app.run()