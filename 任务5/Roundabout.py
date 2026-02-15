import cv2
import numpy as np
from enum import Enum

'''
Function:
8-neighbor edge search, Bezier centerline fitting, drawing edges/centerline.
Longest white column detection, lost line detection.
Variance calculation and visualization.
Crossroad State Machine Framework.
'''

'''
Code Structure:
VisionConfig:Configurations
Point:Basic data structure
Utils:Tools
Track:Track analysis (boundary line/center line), and some simple track character analysis (longest white line/lost line)
Cross:Handle crossing scenario
Visualize:Visualize special line and some text of data characteristics
Main:Orchestrates the entire code; Control video playback
'''

# region Configuration
class VisionConfig:

    # Track Preprocessing & Crop
    CROP_TOP_RATE = 0.0
    CROP_BOTTOM_RATE = 0.2
    GAUSSIAN_KERNEL = (5, 5)
    MORPH_KERNEL = (3, 3)
    BINARY_THRESHOLD = 0    # Threshold for binarization (OTSU is used, base is 0)

    # Tracking Params
    LWL_SAMPLE_STEP = 4         # Sampling step for longest white line search
    MIN_VALID_WIDTH = 50        # Minimum width of valid start line block
    START_LINE_SEARCH_LIMIT = 2/3 # Ratio of height to limit start line search

    # Boundaries & Lost Line
    MAX_SEARCH_ITERATION = 3    # Multiplier of height for max boundary search iterations
    LOSS_THRESHOLD = 0.2        # Threshold for lost line ratio
    LOST_END_ROW_RATIO = 2/3    # Height ratio threshold to determine if line ends too early

    FEA_SAMPLE_STEP = 5         # Sampling step for raw feature side lines

    # Down Corners Params
    CORNER_K = 8                # Step size for K-value correlation (only used in cross right now)
    CORNER_COS_THRESHOLD = 0.5  # Cosine threshold for corner detection
    CORNER_IGNORE_TOP_RATIO = 0.2 # Ignore corners found in the top X% of the image

    # Bezier Curve Fitting Params
    BEZIER_DT = 0.01            # Interpolation step size for Bezier curve

    # Cross Params
    # Entry and Patching Lines
    CROSS_LWL_BROAD_VIEW_RATIO = 0.5     # Longest white line length ratio for entry
    CROSS_ENTER_VAR_THRESHOLD = 50  # Variance threshold for entry
    CROSS_TRACK_HALF_WIDTH_RATIO = 0.4   # Default half-width ratio if start line is lost
    CROSS_FAR_WIDTH_RATIO = 0.25    # Ratio of top width to bottom width
    CROSS_PATCH_OFFSET = 0.9        # Offset factor for control point P1
    # EXIT
    CROSS_EXIT_MIN_FRAMES = 10      # Minimum frames to prevent state oscillation
    CROSS_EXIT_TIMEOUT = 150        # Maximum frames to force exit cross state
    CROSS_EXIT_ROW_RATIO = 0.8      # Start row height ratio for exit condition

    # Ring Params
    RING_APPROACH_TH = 30   # Minimum frames to enter the state "approach" to prevent oscillation
    RING_EXIT_TH = 300      # Minimum frames to enter the state "exit"
    RING_EXIT_TIMEOUT = 300 # Maximum frames to force enter "wait" state
    RING_WAIT_TIMEOUT = 300 # Maximum frames to force exit the ring
    RING_CORNER_K =8        # Step size for Ring K-value correlation
    RING_CORNER_WINDOW_SIZE = 5 # Window size for smoothing when finding the corners in ring
    RING_CORNER_MID_MARGIN = 10 # Crop marginal points to get center roi
    RING_CORNER_MID_PEAK_TH = 5 # Threshold determine if a point is peak in dim row to find middle point
    RING_CORNER_MID_PEAK_RADIUS = 20 # Check peak prominence within RING_CORNER_MID_PEAK_RADIUS.
    # Video
    DEBUG_DELAY_INITIAL_MS = 30         # Video playback delay in ms

    # Colors (BGR)
    COLOR_LONGEST_LINE = (255, 0, 255)  # Magenta
    COLOR_LEFT_POINT = (0, 255, 0)      # Green
    COLOR_RIGHT_POINT = (255, 0, 0)     # Blue
    COLOR_CENTER_LINE = (0, 0, 255)     # Red
    COLOR_CORNER = (0, 255, 255)        # Yellow
    COLOR_TEXT = (255, 0, 255)          # Magenta text
# endregion

#region Basic Structures (Point/Utils)

# Define a 2D coordinate class.
class Point:
    def __init__(self,row,col):
        self.row=int(row)
        self.col=int(col)

    def point2cv(self):
        # Convert point (row,col) to (x,y)
        return self.col,self.row

    @staticmethod
    def get_midpoint(p1,p2):
        return Point((p1.row+p2.row)//2,(p1.col+p2.col)//2)

class Utils:
    @staticmethod
    def bezier_fit(input_points,dt=VisionConfig.BEZIER_DT):
           #
           # 贝塞尔曲线核心函数
           # 根据四个控制点生成三次曲线
           #  Parameters:
           #     input_points: 输入特征点
           #  Returns:
           #     output: 贝塞尔拟合后的点列表
           #
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
# endregion

# region Tracking
# Responsible for longest white line,boundary lines and the centerline.
class Track:
    def __init__(self):

        # About crop
        self.up_crop_rate=VisionConfig.CROP_TOP_RATE     # Top crop ratio
        self.down_crop_rate=VisionConfig.CROP_BOTTOM_RATE   # Bottom crop ratio

        # Edge point sets for left and right track boundaries
        self.LeftPoints=[]
        self.RightPoints=[]

        # Lost points
        self.LeftPoints_LostNum=0
        self.RightPoints_LostNum=0
        self.LeftPoints_LostFlag=0  #0 not lost;1 lost
        self.RightPoints_LostFlag=0
        self.LossThreshold=VisionConfig.LOSS_THRESHOLD

        # About center points
        self.CenterPoints=[]        #Set of points for the centerline
        self.bezier_input=[]        #Control points for Bezier curve fitting

        # Start line
        self.start_flag=False      #Starting row flag (identifies the bottom-most valid row)
        self.start_row=None       #Row index of the starting row
        self.start_left=None      #Column index of the left edge in the starting row
        self.start_right=None     #Column index of the right edge in the starting row

        # The longest white line
        self.Longest_White_Line_Top_Point=None #The peak point of the longest white line
        self.Longest_White_Line_Length=0

        # Corners
        self.LeftDownCorner=None
        self.RightDownCorner=None
        self.LeftUpCorner=None
        self.RightUpCorner=None
        self.left_corner_index=None  # The index of the lower corners on the left side line
        self.right_corner_index=None

        # Side points to get features
        self.raw_features = {
            1: np.array([], dtype=np.int16), # Left scanning points
            -1: np.array([], dtype=np.int16) # Right scanning points
            }

        # Used to find upper corners in Cross (not adopted ultimately)
        self.ScanLeftPoints=None
        self.ScanRightPoints=None

        # Use in morphological opening operation
        self.kernel=cv2.getStructuringElement(cv2.MORPH_RECT,VisionConfig.MORPH_KERNEL)

        # is_external_control. False: fit center line by side lines; True: controlled by special section of the road
        self.is_external_control=False

    def preprocessing(self,frame):
        # 预处理：裁剪->转灰度图->高斯滤波->二值化->形态学运算(去孤立点)
        #Preprocessing: Cropping -> Converting to Grayscale Image -> Gaussian Filtering -> Binarization -> Morphological Operation (Removing Isolated Points)
        cropped_frame=self.crop_video_frame(frame)  # 裁剪视频
        gray_frame=cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)  # 转灰度图
        gray_frame=cv2.GaussianBlur(gray_frame,VisionConfig.GAUSSIAN_KERNEL,0)      # Gaussian filtering, remove high frequency noise
        _, binary_frame=cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津法二值化

        # Morphological operation
        binary_frame=cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, self.kernel)# Remove isolated small black dots
        binary_frame=cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, self.kernel)# Remove isolated small white dots
        return binary_frame,cropped_frame

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
        start_row=int(height * self.up_crop_rate)
        end_row=int(height * (1 - self.down_crop_rate))
        return frame[start_row:end_row,:]

    def find_start_line(self, binary,h,w):

           # 基于最长白列找起始行，运行后起始行行号起始行左右两点分别被存储
           #  Parameters:binary,Binarized image
           #  Returns:None
           # TODO: Numpy version

        # Clear the values of the track
        self.start_row=self.start_left=self.start_right=None
        self.start_flag=False
        self.LeftPoints.clear()
        self.RightPoints.clear()
        self.CenterPoints.clear()

        min_valid_width=VisionConfig.MIN_VALID_WIDTH   #Minimum width threshold for valid blocks (noise filtering)
        # Search for white pixel on the edge of the track, search from the column of the longest white line
        search_limit=int(h*VisionConfig.START_LINE_SEARCH_LIMIT)
        # Use Longest_White_Line_Top_Point.col as anchor_col if available, or use the center of the image
        if self.Longest_White_Line_Top_Point is not None:
            anchor_col=self.Longest_White_Line_Top_Point.col
        else:anchor_col=w//2
        # When a horizontal line is not able to be the start line, try the line above it. But it must be on the bottom part of the image
        for row in range(h-1,search_limit,-1):
            if binary[row,anchor_col]==0:
                continue
            # Find left edge point
            points_left_to_the_anchor_col=binary[row,:anchor_col][::-1]
            left_black_indices=np.where(points_left_to_the_anchor_col==0)[0]
            if (len(left_black_indices)>0):
                l_idx=anchor_col-left_black_indices[0]
            else:   l_idx=0
            # Find right edge point
            points_right_to_the_anchor_col=binary[row,anchor_col:]
            right_black_indices=np.where(points_right_to_the_anchor_col==0)[0]
            if len(right_black_indices)>0:
                r_idx=anchor_col+right_black_indices[0]-1
            else:   r_idx=w-1
            # Validate the width and set the start line
            if r_idx-l_idx>min_valid_width:
                self.start_row,self.start_left,self.start_right=row,l_idx,r_idx
                self.start_flag=True
                self.LeftPoints.append(Point(row, l_idx))
                self.RightPoints.append(Point(row, r_idx))
                break
        # Method: Traversal line-searching
        # self._white_block=[]      #Temporary storage for white pixel blocks in the current row
        # for row in range(h - 1, 0, -1):
        #     cols = np.where(binary[row] == 255)[0]
        #     cols=cols[cols<w]#过滤超出图像范围外的点，防止越界报错
        #     if len(cols) > 0 and (cols[-1] - cols[0]) >= self.min_valid_width:
        #         #print(f"起始行右边界：cols[-1]={cols[-1]}, 宽度={cols[-1] - cols[0]}")
        #         self.start_row, self.start_left, self.start_right = row, cols[0], cols[-1]
        #         self.LeftPoints.append(Point(row, cols[0]))
        #         self.RightPoints.append(Point(row, cols[-1]))
        #         break

    def find_longest_white_line(self,h,w,binary):
        # Numpy version
        # Find the longest white line of the image, return the length and top point of it.

        self.Longest_White_Line_Length=0
        self.Longest_White_Line_Top_Point=None

        # Slice the image to reduce the performance consumption
        step=VisionConfig.LWL_SAMPLE_STEP  #Search step of col
        col_of_interest=binary[:,::step]

        # Flip the image because we have to search from the bottom
        flipped_image=col_of_interest[::-1,:]
        black_mask=(flipped_image==0)

        white_line_length=np.argmax(black_mask, axis=0)

        # Handle the condition that the whole line is white
        # When the whole line is white, np.argmax returns 0, it's not true
        has_obstacle=np.any(black_mask,axis=0)
        white_line_length=np.where(has_obstacle,white_line_length,h)

        # Get Longest_White_Line_Top_Point and Longest_White_Line_Length
        best_col_in_subset=np.argmax(white_line_length)
        self.Longest_White_Line_Length=white_line_length[best_col_in_subset]
        best_col=best_col_in_subset * step
        best_row=h-self.Longest_White_Line_Length
        self.Longest_White_Line_Top_Point=Point(best_row,best_col)

        return self.Longest_White_Line_Top_Point,self.Longest_White_Line_Length

    # def find_longest_white_line(self,binary):
    #     # pure python version, too slow
    #     self.Longest_White_Line_Length=0
    #     self.Longest_White_Line_Top_Point=None
    #     h,w=binary.shape
    #     best_row,best_col=h-1,w//2 #Initialize the row and col of Longest_White_Line_Top_Point
    #     step=4  #Search step of col
    #     for col in range(0,w,step):   #Search from bottom to top
    #         current_row=0               #If no black pixel is hit, current_row must be zero
    #         for row in range(h-1,0,-1):
    #             if binary[row,col]==0:  #Hit black pixel (boundary)
    #                 current_row=row
    #                 break
    #         if current_row<best_row:    #Update the row and col of Longest_White_Line_Top_Point
    #             best_row=current_row
    #             best_col=col
    #     self.Longest_White_Line_Top_Point=Point(best_row,best_col)
    #     self.Longest_White_Line_Length=h-best_row
    #     return self.Longest_White_Line_Top_Point,self.Longest_White_Line_Length

    def search_boundaries(self,binary):
        # TODO: Optimize to inherit the last frame
        # CAN'T clear boundaries there because they have been cleaned up at function find_start_line
        # Clearing boundaries there will cause the loss of the point added by start line
        #self.LeftPoints.clear()
        #self.RightPoints.clear()
        #利用八邻域搜线法得到左右边线
        if self.start_row is None:#若没有起始行，就直接返回
            return
        h,w=binary.shape  # 得出高宽，防越界
        #Use "visited" to prevent backtracking and avoid cycling.
        visited=np.zeros_like(binary, np.uint8)#0 unvisited;1 visited
        directions_l = np.array([#右->右上->上->左上->左->左下->下->右下
            [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]
        ])  # 逗号左边是row，因此是y坐标，逗号右边是col，因此是x坐标
        directions_r = np.array([#左->左上->上->右上->右->右下->下->左下
            [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1]
        ])
        max_iteration= h * VisionConfig.MAX_SEARCH_ITERATION
        count=0

        #搜左边线
        cen_row,cen_col=self.start_row, self.start_left#确定开始时八邻域的九宫格中心#cen_point is white
        visited[cen_row,cen_col]=1
        while cen_row>0 and count<max_iteration: # Search all the way to the top;maximum iteration protection
            count+=1
            found=False # Indicate if the next point is found
            for direction in range(8):
                #取出变化量数组里的值
                delta_row0,delta_col0=directions_l[direction]
                delta_row1,delta_col1= directions_l[(direction + 1) % 8]
                #八邻域九宫格里用来观察颜色的两点的坐标
                new_row0=cen_row+delta_row0
                new_col0=cen_col+delta_col0
                new_row1=cen_row+delta_row1
                new_col1=cen_col+delta_col1
                if not (0<=new_row0<h and 0<=new_col0<w and 0<=new_row1<h and 0<=new_col1<w):#防越界
                    continue
                #When the first point has not been searched, first point is white and next point is black
                if visited[new_row0,new_col0]==0 and binary[new_row0,new_col0]==255 and binary[new_row1,new_col1]==0:
                    visited[new_row0,new_col0]=1
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
        #This method prevents boundary line from crossing(单例标记保护 visited-set synchronization)
        cen_row, cen_col = self.start_row, self.start_right
        visited[cen_row, cen_col] = 1
        count=0#Reset counter
        while cen_row>0 and count<max_iteration:
            count+=1
            found = False  # found是是否找到下一个点的flag
            for direction in range(8):
                delta_row0, delta_col0 = directions_r[direction]
                delta_row1, delta_col1 = directions_r[(direction + 1) % 8]
                new_row0 = cen_row + delta_row0
                new_col0 = cen_col + delta_col0
                new_row1 = cen_row + delta_row1
                new_col1 = cen_col + delta_col1
                if not(0 <= new_row0 < h and 0 <= new_col0 < w and 0 <= new_row1 < h and 0 <= new_col1 < w):
                    continue
                if visited[new_row0,new_col0]==0 and binary[new_row0,new_col0]==255 and binary[new_row1,new_col1]==0:
                    visited[new_row0, new_col0] = 1
                    self.RightPoints.append(Point(new_row0,new_col0))
                    cen_row, cen_col = new_row0, new_col0
                    found = True  # 标记找到
                    break
            if not found:
                break
        #cv2.imshow('mask', visited * 255)

    def detect_lost_line(self,h,w):
        self.LeftPoints_LostFlag = 0
        self.RightPoints_LostFlag=0
        # Calculate line loss metrics
        # We expect boundary points num is roughly the image height
        if self.start_row:
            expected_points=self.start_row
        else: expected_points=h

        self.LeftPoints_LostNum=max(0,expected_points-len(self.LeftPoints))
        self.RightPoints_LostNum=max(0,expected_points-len(self.RightPoints))
        if expected_points>0:
            if (len(self.LeftPoints)==0                                    or   # Do not find boundary
                self.LeftPoints_LostNum/expected_points>self.LossThreshold or   # Not enough points
                self.LeftPoints[-1].row>VisionConfig.LOST_END_ROW_RATIO*h  or   # Ends too early
                self.LeftPoints[-1].col==0                                      # Ends at side boundary
            ):
                self.LeftPoints_LostFlag=1
            if (len(self.RightPoints)==0                                     or
                self.RightPoints_LostNum/expected_points>self.LossThreshold  or
                self.RightPoints[-1].row>VisionConfig.LOST_END_ROW_RATIO*h   or
                self.RightPoints[-1].col==w-1
            ):
                self.RightPoints_LostFlag=1

    @staticmethod
    def cal_cos(pre_point,cur_point,nex_point):
        # Calculate the cosine of the angle between vectors (nex_point-cur_point) and (cur_point-pre_point)
        x1,y1=nex_point.col-cur_point.col,nex_point.row-cur_point.row   #vector (nex_point-cur_point)
        x2,y2=cur_point.col-pre_point.col,cur_point.row-pre_point.row   #vector (cur_point-pre_point)
        norm_v1=(x1**2+y1**2)**0.5
        norm_v2=(x2**2+y2**2)**0.5
        # Avoid division by zero, when denominator is zero, return 1 (collinear)
        if not (norm_v1 and norm_v2):
            return 1
        else: return (x1*x2+y1*y2)/(norm_v1*norm_v2)

    def find_down_corners(self,h,w):
        # Method: K-value correlation method (K值关联法)
        # Iterate the boundary points from bottom to find the inflection point
        k=VisionConfig.CORNER_K
        cos_threshold=VisionConfig.CORNER_COS_THRESHOLD   # Threshold for corner detection
        self.LeftDownCorner,self.RightDownCorner=None,None
        # Find Left Down Corner
        if len(self.LeftPoints)>2*k+1:
            min_cosine=1
            self.left_corner_index=None
            for i in range(k,len(self.LeftPoints)-k-1):
                if self.LeftPoints[i].row<h*VisionConfig.CORNER_IGNORE_TOP_RATIO:        # Ignore top CORNER_IGNORE_TOP_RATIO *100% of the image
                    continue
                pre_point,cur_point,nex_point=self.LeftPoints[i-k],self.LeftPoints[i],self.LeftPoints[i+k]
                current_cosine=self.cal_cos(pre_point,cur_point,nex_point)
                if current_cosine<min_cosine:
                    min_cosine=current_cosine
                    self.left_corner_index=i
            if min_cosine<cos_threshold and self.left_corner_index is not None:
                self.LeftDownCorner=self.LeftPoints[self.left_corner_index]
                # Cannot remove the points after the corner now because they will be used in calculate variance, we can remove them later
                #self.LeftPoints=self.LeftPoints[:self.left_corner_index+1]  # Remove the points after the corner, they are horizontal line of the crossroad
        #  Find Right Down Corner
        if len(self.RightPoints)>2*k+1:
            min_cosine=1
            self.right_corner_index=None
            for i in range(k,len(self.RightPoints)-k-1):
                if self.RightPoints[i].row<h*VisionConfig.CORNER_IGNORE_TOP_RATIO:
                    continue
                pre_point,cur_point,nex_point=self.RightPoints[i-k],self.RightPoints[i],self.RightPoints[i+k]
                current_cosine=self.cal_cos(pre_point,cur_point,nex_point)
                if current_cosine<min_cosine:
                    min_cosine=current_cosine
                    self.right_corner_index=i
            if min_cosine<cos_threshold and self.right_corner_index is not None:
                self.RightDownCorner=self.RightPoints[self.right_corner_index]
                #self.RightPoints=self.RightPoints[:self.right_corner_index+1]
        # Initially I tried this method but with poor results
        # It may be because the eight - neighborhood method is not suitable for judging corner points in a method with coordinate abrupt changes.
        # #Under the condition of left and right boundaries are lost:
        # #Longest_White_Line_Length>0.6h -> cross
        # #Longest_White_Line_Length<0.6h -> sharp corner
        # h,w=binary.shape[:2]
        # if not self.LeftPoints_LostFlag or not self.RightPoints_LostFlag or self.Longest_White_Line_Length<0.6*h:
        #     return
        # #Unreliable when lost line is too severe
        # if (self.LeftPoints_LostNum>0.9*h or
        #     self.RightPoints_LostNum>0.9*h or
        #     len(self.LeftPoints)<20 or
        #     len(self.RightPoints)<10):
        #     return
        # #find LeftDownCorner
        # self.LeftDownCorner=None
        # for i in range (5,len(self.LeftPoints)-11,2):#before corner:smooth, after corner:sharp
        #     if(self.LeftDownCorner is None and  #find the first corner that satisfy the facts
        #         abs(self.LeftPoints[i].col-self.LeftPoints[i-5].col)<5 and
        #         abs(self.LeftPoints[i].col-self.LeftPoints[i-10].col)<5 and
        #         abs(self.LeftPoints[i].col-self.LeftPoints[i+5].col)>5 and
        #         abs(self.LeftPoints[i].col-self.LeftPoints[i+10].col)>5):
        #         self.LeftDownCorner=self.LeftPoints[i]
        #         break
        #
        #  #find RightDownCorner
        # self.RightDownCorner=None
        # for i in range (5,len(self.RightPoints)-6,2):
        #     if(self.RightDownCorner is None and
        #         abs(self.RightPoints[i].col-self.RightPoints[i-3].col)<5 and
        #         abs(self.RightPoints[i].col-self.RightPoints[i-5].col)<5 and
        #         abs(self.RightPoints[i].col-self.RightPoints[i+3].col)>5 and
        #         abs(self.RightPoints[i].col-self.RightPoints[i+5].col)>5):
        #         self.RightDownCorner=self.RightPoints[i]
        #         break

    # def search_up_boundaries(self,binary,h,w):
    #     # Helper function of find_up_corners
    #     # Search boundary lines from top to bottom of the image, assist in identifying upper corners by analyzing slopes(in function find_up_corners).
    #
    #     # Find upper start line, similar to function find_start_line
    #     # Find from Longest_White_Line_Top_Point, it can reduce the number of traversals.
    #     up_start_row=up_start_left=up_start_right=None
    #     up_start_flag=False
    #     self.ScanLeftPoints=[]
    #     self.ScanRightPoints=[]
    #     min_valid_width=50
    #     if self.Longest_White_Line_Top_Point is not None:
    #         anchor_col=self.Longest_White_Line_Top_Point.col
    #         search_start=self.Longest_White_Line_Top_Point.row
    #     else:
    #         anchor_col=w//2
    #         search_start=0
    #     #search_end=int(h*1/4)
    #     # When the car is about to leave crossroad,
    #     # the upper corners are in the lower part of the image
    #     search_end=h-10
    #     if search_start>search_end:
    #         search_start,search_end=search_end,search_start
    #     for row in range(search_start,search_end):
    #         if binary[row,anchor_col]==0:
    #             continue
    #         points_left_to_the_anchor_col=binary[row,:anchor_col][::-1]
    #         left_black_indices=np.where(points_left_to_the_anchor_col==0)[0]
    #         if len(left_black_indices)>0:
    #             l_idx=anchor_col-left_black_indices[0]
    #         else:   l_idx=0
    #         points_right_to_the_anchor_col=binary[row,anchor_col:]
    #         right_black_indices=np.where(points_right_to_the_anchor_col==0)[0]
    #         if len(right_black_indices)>0:
    #             r_idx=anchor_col+right_black_indices[0]-1
    #         else:   r_idx=w-1
    #         # Validate the width and set the start line
    #         if r_idx-l_idx>min_valid_width:
    #             up_start_row,up_start_left,up_start_right=row,l_idx,r_idx
    #             up_start_flag=True
    #             self.ScanLeft.append(Point(row, l_idx))
    #             self.ScanRight.append(Point(row, r_idx))
    #             break
    #
    #     # Find upper boundaries
    #     if up_start_row is None:    return
    #     for row in range(up_start_row+1,search_end):
    #         # if binary[row,anchor_col]==0:
    #         #     continue
    #         points_left_to_the_anchor_col=binary[row,:anchor_col][::-1]
    #         left_black_indices=np.where(points_left_to_the_anchor_col==0)[0]
    #         if (len(left_black_indices)>0):
    #             l_idx=anchor_col-left_black_indices[0]
    #         else:   l_idx=0
    #         self.ScanLeftPoints.append(Point(row,l_idx))
    #
    #         points_right_to_the_anchor_col=binary[row,anchor_col:]
    #         right_black_indices=np.where(points_right_to_the_anchor_col==0)[0]
    #         if (len(right_black_indices)>0):
    #             r_idx=anchor_col+right_black_indices[0]-1
    #         else:   r_idx=w-1
    #         self.ScanRightPoints.append(Point(row, r_idx))
        #These code are based on eight-neighbourhood, the effect is not very good
        # when there is a sharp bend immediately after the crossroad
        # # Find upper boundaries, similar to function search_boundaries
        # if up_start_row is None:
        #     return up_LeftPoints,up_RightPoints
        # visited=np.zeros_like(binary, np.uint8)#0 unvisited;1 visited
        # directions_l_prime=np.array([
        #     [0, 1],   #right
        #     [1, 1],   #bottom_right
        #     [1, 0],   #down
        #     [1, -1],  #bottom_left
        #     [0, -1],  #left
        #     [-1, -1], #top_left
        #     [-1, 0],  #up
        #     [-1, 1]   #top_right
        # ])
        # directions_r_prime=np.array([
        #     [0, -1],  #left
        #     [1, -1],  #bottom_left
        #     [1, 0],   #down
        #     [1, 1],   #bottom_right
        #     [0, 1],   #right
        #     [-1, 1],  #top_right
        #     [-1, 0],  #up
        #     [-1, -1]  #top_left
        # ])
        # max_iteration=h*2
        # count=0
        #
        # # Left upper boundaries
        # cen_row,cen_col=up_start_row, up_start_left
        # visited[cen_row,cen_col]=1
        # while cen_row<h-1 and count<max_iteration:
        #     count+=1
        #     found=False
        #     for direction in range(8):
        #         delta_row0,delta_col0=directions_l_prime[direction]
        #         delta_row1,delta_col1= directions_l_prime[(direction + 1) % 8]
        #         new_row0=cen_row+delta_row0
        #         new_col0=cen_col+delta_col0
        #         new_row1=cen_row+delta_row1
        #         new_col1=cen_col+delta_col1
        #         if not (0<=new_row0<h and 0<=new_col0<w and 0<=new_row1<h and 0<=new_col1<w):
        #             continue
        #         if visited[new_row0,new_col0]==0 and binary[new_row0,new_col0]==255 and binary[new_row1,new_col1]==0:
        #             visited[new_row0,new_col0]=1
        #             up_LeftPoints.append(Point(new_row0,new_col0))
        #             cen_row,cen_col =new_row0,new_col0
        #             found=True
        #             break
        #     if not found:
        #         break
        #
        # # Right upper boundaries
        # cen_row, cen_col =up_start_row,up_start_right
        # visited[cen_row, cen_col] = 1
        # count=0 #Reset counter
        # while cen_row<h-1 and count<max_iteration:
        #     count+=1
        #     found = False
        #     for direction in range(8):
        #         delta_row0,delta_col0=directions_r_prime[direction]
        #         delta_row1,delta_col1=directions_r_prime[(direction + 1) % 8]
        #         new_row0=cen_row+delta_row0
        #         new_col0=cen_col+delta_col0
        #         new_row1=cen_row+delta_row1
        #         new_col1=cen_col+delta_col1
        #         if not(0 <= new_row0 < h and 0 <= new_col0 < w and 0 <= new_row1 < h and 0 <= new_col1 < w):
        #             continue
        #         if visited[new_row0,new_col0]==0 and binary[new_row0,new_col0]==255 and binary[new_row1,new_col1]==0:
        #             visited[new_row0, new_col0] = 1
        #             up_RightPoints.append(Point(new_row0,new_col0))
        #             cen_row, cen_col = new_row0, new_col0
        #             found = True
        #             break
        #     if not found:
        #         break
        # #cv2.imshow('mask', visited * 255)

    # def find_up_corners(self,binary,h,w):
    #     # Find up corners, similar to function find_down_corners,
    #     # but it must perform line_searching in upper part first use function search_up_boundaries
    #     # Method: K-value correlation method (K值关联法)
    #
    #     K=5
    #     cos_threshold=0.5
    #     self.LeftUpCorner,self.RightUpCorner=None,None
    #
    #     up_LeftPoints,up_RightPoints=self.search_up_boundaries(binary,h,w)
    #     # Find Left Up Corner
    #     if len(up_LeftPoints)>2*K+1:
    #         min_cosine=1
    #         min_cosine_index=None
    #         for i in range(K,len(up_LeftPoints)-K-1):
    #             if up_LeftPoints[i].row<h*0.2:
    #                 continue
    #             pre_point,cur_point,nex_point=up_LeftPoints[i-K],up_LeftPoints[i],up_LeftPoints[i+K]
    #             current_cosine=self.cal_cos(pre_point,cur_point,nex_point)
    #             if current_cosine<min_cosine:
    #                 min_cosine=current_cosine
    #                 min_cosine_index=i
    #         if min_cosine<cos_threshold and min_cosine_index is not None:
    #             self.LeftUpCorner=up_LeftPoints[min_cosine_index]
    #
    #             #The upper boundary line is not to be used for the time being.
    #             '''# Remove the points after the corner, they are horizontal line of the crossroad
    #             up_LeftPoints=up_LeftPoints[:min_cosine_index+1]'''
    #
    #     #  Find Right Up Corner
    #     if len(up_RightPoints)>2*K+1:
    #         min_cosine=1
    #         min_cosine_index=None
    #         for i in range(K,len(up_RightPoints)-K-1):
    #             if up_RightPoints[i].row<h*0.2:
    #                 continue
    #             pre_point,cur_point,nex_point=up_RightPoints[i-K],up_RightPoints[i],up_RightPoints[i+K]
    #             current_cosine=self.cal_cos(pre_point,cur_point,nex_point)
    #             if current_cosine<min_cosine:
    #                 min_cosine=current_cosine
    #                 min_cosine_index=i
    #         if min_cosine<cos_threshold and min_cosine_index is not None:
    #             self.RightUpCorner=up_RightPoints[min_cosine_index]
    #             #up_RightPoints=up_RightPoints[:min_cosine_index+1]
    #
    # # Not sure if it'll be used, but let's write it down anyway haha
    # def find_corners(self,binary,h,w):
    #     self.find_down_corners(h,w)
    #     self.find_up_corners(binary,h,w)

    def generate_bezier_center(self,h,w):
        #函数功能：生成贝塞尔拟合中心线
        #h和w是图像的高和宽，用来防越界
        self.CenterPoints.clear()
        if len(self.LeftPoints)<1 or len(self.RightPoints)<1:return
        def get_three_part_points(points):
            #函数功能：返回首点、尾点、三等分点
            n = len(points)
            # extracted_points=[]
            # index=[0,n//3,2*n//3,-1]
            # for idx in index:
            #     p=points[idx]
            #     p_clipped_row=int(np.clip(p.row,0,h-1))
            #     p_clipped_col=int(np.clip(p.col,0,w-1))
            #     extracted_points.append(Point(p_clipped_row,p_clipped_col))
            # return extracted_points
            for p in points:
                p.row=int(np.clip(p.row,0,h-1))
                p.col=int(np.clip(p.col,0,w-1))
            return [points[0],points[n//3],points[2*n//3],points[-1]]

        left_feature=get_three_part_points(self.LeftPoints)
        right_feature=get_three_part_points(self.RightPoints)
        self.bezier_input=[]
        for l_p,r_p in zip(left_feature, right_feature):
            mid_row=(l_p.row+r_p.row)/2
            mid_col=(l_p.col+r_p.col)/2
            #控制点也要防越界，但控制点必须确保为4个，所以不能直接过滤
            mid_row=round(np.clip(mid_row,0,h-1))
            mid_col=round(np.clip(mid_col,0,w-1))
            self.bezier_input.append(Point(round(mid_row),round(mid_col)))
        self.CenterPoints= Utils.bezier_fit(self.bezier_input)

    def _truncate_track_line(self):
        # Remove the points after the corner, they are horizontal line of the crossroad
        if self.LeftDownCorner is not None:
            self.LeftPoints=self.LeftPoints[:self.left_corner_index+1]
        if self.RightDownCorner is not None:
            self.RightPoints=self.RightPoints[:self.right_corner_index+1]

    def _scan_raw_features(self,h,w,binary):

        # self.raw_features[side][i] <-> image coordinates: (row: i*step, col: value)

        # Traversal scan to get edges, but it's about getting features, not to determine the boundaries
        # therefore the step can be a little large
        # This is only used by Ring Class right now, maybe we can extend it to Cross Class
        # Different scenario has different forms where corner points exist
        # Eg: The cross has a total of 4 corner points on both sides, while the ring has 3 corner points on one side.
        # so the corners should be decoupled from Track Class
        # TODO: Let Cross Class fully use Track.raw_features
        # TODO: Let Cross Class fully use Left-Right Decoupling like Ring Class
        # TODO: Change code structure, remove corners from Track and add corners in cross
        step = VisionConfig.FEA_SAMPLE_STEP  #Search step of row
        row_of_interest = binary[::step, :]

        if self.Longest_White_Line_Top_Point is not None:
            anchor_col = self.Longest_White_Line_Top_Point.col
        else:anchor_col = w // 2
        anchor_col = int(np.clip(anchor_col, 0, w-1))

        # Find left raw feature points
        points_left_to_the_anchor_col = row_of_interest[:, :anchor_col][:, ::-1]
        left_black_mask = points_left_to_the_anchor_col == 0
        left_first_black_indices = np.argmax(left_black_mask, axis=1)
        has_obstacle = np.any(left_black_mask, axis=1)  # The condition of all white
        self.raw_features[1] = np.where(has_obstacle, anchor_col-left_first_black_indices, 0)

        # Find right raw feature points
        points_right_to_the_anchor_col = row_of_interest[:, anchor_col:]
        right_black_mask = points_right_to_the_anchor_col == 0
        right_first_black_indices = np.argmax(right_black_mask, axis=1)
        has_obstacle = np.any(right_black_mask, axis=1)
        self.raw_features[-1] = np.where(has_obstacle, anchor_col+right_first_black_indices-1, w-1)

    def process(self, frame):
        #赛道图像主流程：预处理->找最长白列->找起始行->搜索边线->找角点(如有)->中心线拟合
        binary_frame,cropped_frame=self.preprocessing(frame)
        h,w=binary_frame.shape#传图像大小
        self.find_longest_white_line(
            h,w,binary_frame)  #Find Longest_White_Line_Length and the top point of Longest_White_Line
        self.find_start_line(binary_frame,h,w)                  #用二值化图找起始行
        self.search_boundaries(binary_frame)  #搜索边线
        self._scan_raw_features(h,w,binary_frame)
        self.detect_lost_line(h, w)                     # Lost line judgement
        self.find_down_corners(h,w)               # Find Down Corners

        return cropped_frame

    def update_center_line(self,h,w):
        if not self.is_external_control:
            self._truncate_track_line()
            self.generate_bezier_center(h,w)                #贝塞尔中心线拟合
        if self.is_external_control:
            pass
# endregion

# region Analysing
# Responsible for variance calculation and visualization.
class Analyse:
    def __init__(self):
        self.sigma_left = 0.0  # Variance of the left boundary line
        self.sigma_right = 0.0  # Variance of the right boundary line
        self.sigma_center = 0.0  # Variance of the center line

    def cal_sigma_of_all(self,tracker):
        def cal_var(points, dim):
                #
                # Brief:Calculate the variance of points' coordinates
                # Parameters:points
                #            dim:dimension,when dim==0,calculate the var of row, when dim==1,calculate the var of col
                # Returns:calculated variance
                #
            if len(points) < 2:
                return 0.0
            data = np.array([p.row if dim == 0 else p.col for p in points])
            return np.var(data)
        #You can only calculate the variance by passing an instance of the race track(前面的赛道数据类) data class.
        self.sigma_left=cal_var(tracker.LeftPoints,1)
        self.sigma_right=cal_var(tracker.RightPoints,1)
        self.sigma_center=cal_var(tracker.CenterPoints, 1)

    def process(self,tracker):
        # 计算方差
        self.cal_sigma_of_all(tracker)
#endregion

# region Scenarios
# Responsible for the crossroad
# The far-end center points are more stable than the inflection points of the near end edge
class Cross:

    # State Enum
    class CrossStep(Enum):
        NONE=0  # Not in a crossroad
        Fix=1   # Patching line and checking exit

    # Mode Enum
    class CrossMode(Enum):
        NONE=0
        Left=1      # Slanted entry from left
        Right=2     # Slanted entry from right
        Straight=3  # Straight entry

    def __init__(self):
        self.step=self.CrossStep.NONE
        self.mode=self.CrossMode.NONE
        self.track_half_width=0
        self.debug_info={}
        self._exit_clk=0

    def process(self,h,w,track,analyse):
        # State Machine
        # Returns: True: Cross
        #          False: Drive normally
        if self.step==self.CrossStep.NONE:
            if self._check_entry(h,w,track,analyse):
                track.is_external_control=True  # Tracker no longer control line generation
                self._exit_clk=0
                self.step=self.CrossStep.Fix
                self.mode=self._determine_mode(track,analyse)
                self._patch_lines(h,w,track)
                return True
            track.is_external_control=False
            return False
        elif self.step==self.CrossStep.Fix:
            self._patch_lines(h,w,track)
            self._exit_clk+=1
            if self._check_exit(h,w,track):
                track.is_external_control=False
                self.step=self.CrossStep.NONE
                self.mode=self.CrossMode.NONE
                return  False
            track.is_external_control=True
            return True

    def _check_entry(self,h,w,track,analyse):
        # Standard: Both left and right lost line/ Find at least one corner/ Vision is not too narrow
        lost_line=track.LeftPoints_LostFlag and track.RightPoints_LostFlag
        broad_view=track.Longest_White_Line_Length>h*VisionConfig.CROSS_LWL_BROAD_VIEW_RATIO
        #When the line is short, the variance method is inaccurate and should only be used as a reference.
        big_side_var=(analyse.sigma_left>VisionConfig.CROSS_ENTER_VAR_THRESHOLD
                      and analyse.sigma_right>VisionConfig.CROSS_ENTER_VAR_THRESHOLD)
        down_corners_exist=track.LeftDownCorner is not None or track.RightDownCorner is not None

        should_enter= lost_line and broad_view and down_corners_exist

        # Save debug info
        self.debug_info={
            "State":"OutsideCross",
            "BothLost":lost_line,
            "OpenView":broad_view,
            "CornExist":down_corners_exist,
            "BigVar":big_side_var,
            "ENTER":should_enter
        }
        return should_enter

    @staticmethod
    def _determine_mode(track,analyse):
        # Prioritize the side where the corner is detected
        if (track.LeftDownCorner and track.RightDownCorner and
                0.5*analyse.sigma_right<analyse.sigma_left<2*analyse.sigma_right):
            return Cross.CrossMode.Straight
        if track.LeftDownCorner is not None and track.LeftPoints_LostFlag:
            return Cross.CrossMode.Left
        elif track.RightDownCorner is not None and track.RightPoints_LostFlag:
            return Cross.CrossMode.Right
        else:   #TODO: We can do sth special when nothing fit
            return Cross.CrossMode.Straight

    def _patch_lines(self,h,w,track):
        # Determine the width of the track
        if track.start_flag:
            self.track_half_width=(track.start_right-track.start_left)//2
        else:   self.track_half_width=w*VisionConfig.CROSS_TRACK_HALF_WIDTH_RATIO

        # Determine control points:p0(start) p1(corner) p3(End) p2(mid point of p1 and p3)
        p0=p1=p2=p3=None

        # Determine p0
        if track.start_flag:
            p0 =Point.get_midpoint(track.LeftPoints[0],track.RightPoints[0])
        else:
            p0 = Point(h-1, w//2)

        # Determine p1
        # Find corner(s)->use current corner to determine p1
        if self.mode==self.CrossMode.Straight and track.LeftDownCorner and track.RightDownCorner:
            p1=Point.get_midpoint(track.LeftDownCorner,track.RightDownCorner)
        elif self.mode==self.CrossMode.Left and track.LeftDownCorner:
            p1=Point(track.LeftDownCorner.row,
                     track.LeftDownCorner.col
                     +VisionConfig.CROSS_PATCH_OFFSET*self._get_offset(track,track.LeftDownCorner.row))
        elif self.mode==self.CrossMode.Right and track.RightDownCorner:
            p1=Point(track.RightDownCorner.row,
                     track.RightDownCorner.col
                     -VisionConfig.CROSS_PATCH_OFFSET*self._get_offset(track,track.RightDownCorner.row))
        # Don't find corner(s)->use the midpoint of p3 (simplified version) and p0
        if not p1:
            temp_p3=track.Longest_White_Line_Top_Point if track.Longest_White_Line_Top_Point else Point(0,w//2)
            p1_row=Point.get_midpoint(p0,temp_p3).row
            p1=Point(p1_row,p0.col-10)

        # Determine p3: if the longest white line is available and have a broad view,
        # use the longest white line end point,
        # else use the intersection of extension p0p1 and the image top
        if track.Longest_White_Line_Length>VisionConfig.CROSS_LWL_BROAD_VIEW_RATIO*h:
            p3=track.Longest_White_Line_Top_Point
        else:
            dx,dy=p1.col-p0.col,p1.row-p0.row
            if dy==0: dy=-0.01  # Avoid division by zero
            p3_col=p0.col-p0.row*dx/dy
            p3=Point(0,np.clip(p3_col,0,w-1))

        # Determine p2: The midpoint of p1 and p3
        p2=Point.get_midpoint(p1,p3)

        # Generate center line
        track.CenterPoints= Utils.bezier_fit([p0, p1, p2, p3])
        # Generate side lines (Do not need to do it
        pass

    def _check_exit(self,h,w,track):
        # Standard: At least one line recovered/ Do not find corners/ Start row is near the bottom
        if self._exit_clk<VisionConfig.CROSS_EXIT_MIN_FRAMES:   # Prevent state oscillation
            self.debug_info={
                "State":"Locked","Time":self._exit_clk
            }
            return False
        if self._exit_clk>VisionConfig.CROSS_EXIT_TIMEOUT:  # Force quit on timeout
            self.debug_info={
                "State":"Timeout","EXIT":True
            }
            return True
        get_line=not track.LeftPoints_LostFlag or not track.RightPoints_LostFlag
        down_corners_not_found=track.LeftDownCorner is None and track.RightDownCorner is None
        low_start_row=False
        if track.start_row is not None:
            low_start_row=track.start_row>VisionConfig.CROSS_EXIT_ROW_RATIO*h

        should_exit=get_line and down_corners_not_found and low_start_row

        self.debug_info={
            "State":"InsideCross",
            'Mode':self.mode.name,
            "GetLine":get_line,
            "NoCorn":down_corners_not_found,
            "LowSt":low_start_row,
            "Time":self._exit_clk,
            "EXIT":should_exit
        }
        return should_exit

    def _get_offset(self,track,current_row):
        # When calculating side lines according to center line,
        # or determine one of the control point p1,we need to calculate offset
        if track.start_row is None or self.track_half_width==0:return 0
        # Assume that image top track width is VisionConfig.CROSS_FAR_WIDTH_RATIO times bottom track width
        top_track_half_width=VisionConfig.CROSS_FAR_WIDTH_RATIO*self.track_half_width
        if track.start_row==0: return top_track_half_width
        offset=top_track_half_width+current_row*(self.track_half_width-top_track_half_width)/track.start_row
        return offset

# Responsible for the scenario of Roundabout
class Ring:
    class RingStep(Enum):
        NONE = 0
        APPROACH = 1
        ENTER = 2
        INSIDE = 3
        EXIT = 4
        WAIT = 5

    class RingMode(Enum):
        NONE = 0
        LEFT = 1
        RIGHT = -1

    def __init__(self):
        self.step = self.RingStep.NONE
        self.mode = self.RingMode.NONE
        self.debug_info = {}

        self.corners = {
                1:  {'down': None, 'middle': None, 'up': None}, # Left Corners
                -1: {'down': None, 'middle': None, 'up': None}  # Right Corners
            }

        # Debouncing
        self._approach_clk = 0
        self._inside_clk = 0
        self._exit_clk = 0
        self._wait_clk = 0

        self._corners = None
        self._side = 0

    def reset(self):
        self.step = self.RingStep.NONE
        self.mode = self.RingMode.NONE
        self.debug_info = {}

        self._approach_clk = 0
        self._inside_clk = 0
        self._exit_clk = 0
        self._wait_clk = 0

        self._corners = None
        self._side = 0

    def process(self, h, w, track):
        self._side = self.mode.value

        # If ring features appear for some time, the car enter "approach" state.
        if self.step == self.RingStep.NONE:
            # Positive for left, negative for right.
            if self._check_approach(h, w, track, approach_side = 1):
                self._approach_clk += 1
            elif self._check_approach(h, w, track, approach_side = -1):
                self._approach_clk -= 1
            else:   self._approach_clk = 0

            if abs(self._approach_clk) >= VisionConfig.RING_APPROACH_TH:
                try:
                    self.step = self.RingStep.APPROACH
                    self._side = int(np.sign(self._approach_clk))
                    self.mode = self.RingMode(self._side)
                    self._approach_clk = 0
                except ValueError as e:
                    print(f"ValueError:{e}")
                    self._approach_clk = 0

        # Check entry according to the corners
        elif self.step == self.RingStep.APPROACH:
            self._connect_corners()
            if self._check_entry(h):
                self.step = self.RingStep.ENTER

        # Patch entry lines and check "inside" state
        elif self.step == self.RingStep.ENTER:
            track.is_external_control = True
            self._patch_entry_lines(track)
            if self._check_inside(track):
                self.step = self.RingStep.INSIDE
                self._inside_clk = 0

        # Let track class to fit center points and check exit
        elif self.step == self.RingStep.INSIDE:
            track.is_external_control = False
            self._inside_clk += 1
            if self._inside_clk >= VisionConfig.RING_EXIT_TH and self._check_exit(track):
                self.step = self.RingStep.EXIT
                self._exit_clk = 0

        # Patch exit lines and check "wait" state
        elif self.step == self.RingStep.EXIT:
            track.is_external_control = True
            self._exit_clk += 1
            if not self._patch_exit_lines(track) or self._exit_clk >= VisionConfig.RING_EXIT_TIMEOUT:
                self.step = self.RingStep.WAIT
                self._wait_clk = 0

        # Patch "wait" lines and check exit ring
        elif self.step == self.RingStep.WAIT:
            track.is_external_control = True
            self._wait_clk +=1
            if not self._connect_corners() or self._wait_clk >= VisionConfig.RING_WAIT_TIMEOUT:
                self.reset()
                track.is_external_control = False

    def _check_approach(self, h, w, track, approach_side):
        # 3个角点全齐：+3分
        # 另一侧是直线：+2分
        # 白列偏移：+1分
        # 中间黑区：+1分
        # 超过5分就算true
        pass

    def _connect_corners(self):
        # Dismiss the gap of the ring
        # APPROACH 传下+中角点，WAIT 传上+中角点
        pass

    def _check_entry(self, h):
        pass

    def _patch_entry_lines(self, track):
        pass

    def _check_inside(self, track):
        pass

    def _check_exit(self, track):
        pass

    def _patch_exit_lines(self, track):
        # If fail to patch exit lines, return false
        pass

    def find_corners(self, h, w, track, side):
        # @brief  Find corners (down, middle, up) in Ring
        # @param[in]  track: track object
        #             track.raw_features: np array
        #             track.raw_features[side][i] <-> image coordinates: (row: i*step, col: value)
        # @retval None. Results are stored in self.corners[side] dictionary
        # @note Similar to function find_down_corners in Track class, use K-value correlation method,
        #       but data stru is np.array so the code there cannot be reused there

        self.corners[side] = {'down': None, 'middle': None, 'up': None}
        step =  VisionConfig.FEA_SAMPLE_STEP
        k = VisionConfig.RING_CORNER_K

        # Filter out invalid points (close to left or right edge of the image)
        # valid_row is the row//step of valid points
        assert side in [1,-1],f"Invalid side::{side}"
        valid_row = np.where((track.raw_features[side] > 2) & (track.raw_features[side] < w-3))[0]
        if len(valid_row) <= 2 * k:
            return False
        valid_col = track.raw_features[side][valid_row]

        # Smoothing
        window = np.ones(VisionConfig.RING_CORNER_WINDOW_SIZE) / VisionConfig.RING_CORNER_WINDOW_SIZE
        valid_col = np.convolve(valid_col, window, mode='same')

        # dy = y_{near} - y_{far}
        # dx = x_{near} - x_{far}
        # Dislocation Subtraction
        dy = step * (valid_row[k:] - valid_row[:-k])
        dx = valid_col[k:] - valid_col[:-k]

        # Find angle sudden changes
        angle = np.degrees(np.arctan2(dy, dx))
        angle_diff = np.diff(angle)
        angle_diff_abs = np.abs(angle_diff)
        selected_points = np.where(angle_diff_abs >= 30)[0]

        # Find middle corner
        margin = VisionConfig.RING_CORNER_MID_MARGIN
        is_peak_threshold = VisionConfig.RING_CORNER_MID_PEAK_TH
        is_peak_radius = VisionConfig.RING_CORNER_MID_PEAK_RADIUS

        if len(valid_col) > 2 * margin:
            roi_col = valid_col[margin:-margin]
            mid_corn_idx = np.argmax(roi_col * side) + margin
            is_peak = False

            start_col = valid_col[max(0,mid_corn_idx-is_peak_radius)]
            end_col = valid_col[min(len(valid_col) - 1, mid_corn_idx+is_peak_radius)]
            if (valid_col[mid_corn_idx] * side > start_col * side + is_peak_threshold  and
                    valid_col[mid_corn_idx] * side > end_col * side + is_peak_threshold):
                is_peak = True
            if is_peak:
                # Since valid_col is indexed by valid_row, they map one-to-one.
                # If the peak is the mid_corn_idx-th element in valid_col,
                # then valid_row[mid_corn_idx] provides its original index.
                # Therefore, valid_row[mid_corn_idx] * step determines the peak's actual row in the full image.
                real_idx = valid_row[mid_corn_idx]
                self.corners[side]['middle'] = Point(real_idx * step, track.raw_features[side][real_idx])


        return True





#endregion

# region Visualization & Main
# Responsible for visualize all elements
class Visualize:

    @staticmethod
    def draw_points(frame,tracker):
        # Brief:Visualize everything
        h,w = frame.shape[:2]
        # Draw the longest white line
        if tracker.Longest_White_Line_Top_Point is not None:
            cv2.line(frame,
                     (tracker.Longest_White_Line_Top_Point.col, h - 1),
                     (tracker.Longest_White_Line_Top_Point.col, tracker.Longest_White_Line_Top_Point.row),
                     VisionConfig.COLOR_LONGEST_LINE,
                     2)
        # 可视化边缘点
        for p in tracker.LeftPoints:
            cv2.circle(frame, p.point2cv(), 2, VisionConfig.COLOR_LEFT_POINT, -1)
        for p in tracker.RightPoints:
            cv2.circle(frame, p.point2cv(), 2, VisionConfig.COLOR_RIGHT_POINT, -1)
        # 可视化中心线控制点
        # for p in tracker.bezier_input:
        #    cv2.circle(frame,p.point2cv(), 4, (0, 0, 255), -1)
        # 可视化中线
        for i in range(len(tracker.CenterPoints) - 1):
            p1, p2 = tracker.CenterPoints[i], tracker.CenterPoints[i + 1]
            cv2.line(frame, p1.point2cv(), p2.point2cv(), VisionConfig.COLOR_CENTER_LINE, 2)

        #Draw corners(Large yellow (VisionConfig.COLOR_CORNER) dots with red borders)
        corners=[tracker.LeftDownCorner, tracker.RightDownCorner, tracker.LeftUpCorner, tracker.RightUpCorner]
        for p in corners:
            if p is not None:
                cv2.circle(frame,p.point2cv(),6,VisionConfig.COLOR_CORNER,-1)
                cv2.circle(frame,p.point2cv(),8,(0,0,255),2)

        return frame

    @staticmethod
    def draw_text(h,w,frame,main,tracker,analyser,crosser):

        # Visualize delay time
        font_scale=0.5
        font_thickness=1
        text=f"Delay:{main.delay}ms"
        cv2.putText(frame, text, (int(w*0.3),30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2*font_thickness)
        cv2.putText(frame, text, (int(w*0.3),30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 0, font_thickness)

        # Visualize data analysis
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
            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        VisionConfig.COLOR_TEXT, font_thickness)
            y += 30

        # Visualize cross debug info
        font_scale=0.4
        font_thickness=1
        if hasattr(crosser,'debug_info') and crosser.debug_info:
            x=int(w*0.7)
            y=30
            for key,val in crosser.debug_info.items():
                color=(0,255,0) if val else (0,0,255)   #True: green; False: red
                if key in ['State','Mode','Time']:
                    color=(0,255,255)                   #Txt: Yellow
                    txt=f"{key}:{val}"
                else:
                    txt=f"{key}:{int(val)}"
                # Add a black border,otherwise it'll not clear to see
                cv2.putText(frame,txt,(x,y),cv2.FONT_HERSHEY_SIMPLEX,font_scale,0,2*font_thickness)
                cv2.putText(frame,txt,(x,y),cv2.FONT_HERSHEY_SIMPLEX,font_scale,color,font_thickness)
                y+=25

        return frame

    @staticmethod
    def process(h,w,frame,main,tracker,analyser,crosser):
        Visualize.draw_points(frame, tracker)
        Visualize.draw_text(h,w,frame,main,tracker,analyser,crosser)
        return frame

#Orchestrator
class Main:
    def __init__(self,video_path):
        self.cap=cv2.VideoCapture(video_path)
        self.tracker=Track()  # 实例化赛道数据类
        self.analyser=Analyse()  # 实例化处理类
        self.crosser=Cross()

        # Control the video play, help to debug
        self.delay=VisionConfig.DEBUG_DELAY_INITIAL_MS  # Initial delay in ms
        self.is_paused=False                            # Pause flag
        self.is_play_single_frame=False                 # Step one frame flag
        self.latest_valid_frame=None                    # Cached frame

    def run(self):
        # 函数：调用主流程，播放视频
        print("[Space] Pause/Resume, [W] Faster, [S] Slower, [D] Next Frame, [Q] Quit")
        while True:
            frame=None  #Reset the frame

            if (not self.is_paused) or self.is_play_single_frame:# Allowed to read next frame
                self.is_play_single_frame=False #Reset play single frame flag
                ret,frame=self.cap.read()
                if not ret:
                    print("Video End")
                    break

            if frame is not None:   # Image processing, only when there is a new frame
                # TODO: Package to a single function
                display_frame=frame.copy()
                cropped_frame=self.tracker.process(display_frame)
                h,w=cropped_frame.shape[:2]
                self.analyser.process(self.tracker)
                self.crosser.process(h,w,self.tracker, self.analyser)

                self.tracker.update_center_line(h,w)   # Generate final center line if is in straight road

                Visualize.process(h,w,
                                # Must use cropped_frame, not display_frame, or we must handle coordinate offset
                                cropped_frame,
                                self,
                                self.tracker,
                                self.analyser,
                                self.crosser
                                )

                self.latest_valid_frame=display_frame
            if self.latest_valid_frame is not None: #No matter if there is a new frame, we just need a previous valid frame
                final_show=self.latest_valid_frame.copy()
                if self.is_paused:
                    h,w=final_show.shape[:2]
                    cv2.putText(final_show,'PAUSED',
                                (int(w*0.3),int(h*0.5)),cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,0,255),2)
                cv2.imshow('Video',final_show)

            # Key control
            key=cv2.waitKey(self.delay) & 0xff
            if key==ord(' '):                            # Toggle pause
                self.is_paused=not self.is_paused
                print(f"Paused:{self.is_paused}")
            elif key==ord('w'):                          # Faster
               self.delay=max(1,self.delay-5)
            elif key==ord('s'):                          # Slower
               self.delay+=10
            elif key==ord('q'):                          # EXIT
                print("Video break")
                break
            elif key==ord('d'):                          # Step one frame
                self.is_play_single_frame=True
        # 释放资源
        self.cap.release()
        cv2.destroyAllWindows()
#endregion

if __name__ == "__main__":
    main=Main('cross1.mp4')
    main.run()