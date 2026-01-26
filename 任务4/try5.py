import cv2
import numpy as np

'''
代码功能：
基于“中心锚点逐行横向扫描”的十字路口角点检测
完全复刻 my_image.c 的搜线逻辑，解决弯道带跑、角点闪烁问题
'''

# 1. 基础数据结构
class Point:
    def __init__(self, row, col):
        self.row = row
        self.col = col
    def to_tuple(self):
        return (self.col, self.row)

# 2. 赛道追踪核心类
class Track:
    def __init__(self):
        # 图像裁剪参数
        self.up_chop_rate = 0
        self.down_chop_rate = 0.3

        # 基础点集
        self.LeftPoints = []
        self.RightPoints = []

        # 上半部分扫描点集 (核心修改)
        self.UpLeftPoints = []
        self.UpRightPoints = []

        # 丢线标志
        self.LeftPoints_LostNum = 0
        self.RightPoints_LostNum = 0
        self.LeftPoints_LostFlag = 0
        self.RightPoints_LostFlag = 0
        self.LostThreshold = 0.2

        # 中心线与贝塞尔
        self.CenterPoints = []
        self.bezier_input = []

        # 起始行与最长白列
        self.start_flag = False
        self.start_row = None
        self.start_left = None
        self.start_right = None
        self.Longest_White_Line_Top_Point = None
        self.Longest_White_Line_Length = 0

        # 角点
        self.LeftDownCorner = None
        self.RightDownCorner = None
        self.LeftUpCorner = None
        self.RightUpCorner = None

        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def preprocessing(self, frame):
        """预处理：裁剪 -> 灰度 -> 二值化 -> 开运算去噪"""
        cropped_frame = self.crop_video_frame(frame)
        gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        # 高斯滤波平滑噪点
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        # 大津法阈值
        _, binary_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 形态学操作保证边缘平滑
        binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, self.kernel)
        binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, self.kernel)
        return binary_frame, cropped_frame

    def crop_video_frame(self, frame):
        height, width = frame.shape[:2]
        start_row = int(height * self.up_chop_rate)
        end_row = int(height * (1 - self.down_chop_rate))
        return frame[start_row:end_row, :]

    def find_Longest_White_Line_Length(self, binary):
        """寻找最长白列（图像视野最远端）"""
        self.Longest_White_Line_Length = 0
        self.Longest_White_Line_Top_Point = None
        h, w = binary.shape
        best_row, best_col = h - 1, w // 2

        # 粗扫，步长4
        step = 4
        for col in range(0, w, step):
            current_row = 0
            # 从下往上找第一个黑点
            for row in range(h - 1, 0, -1):
                if binary[row, col] == 0:
                    current_row = row
                    break
            # 更新最高点（row越小越高）
            if current_row < best_row:
                best_row = current_row
                best_col = col

        self.Longest_White_Line_Top_Point = Point(best_row, best_col)
        self.Longest_White_Line_Length = h - best_row
        return self.Longest_White_Line_Top_Point, self.Longest_White_Line_Length

    def find_start_line(self, binary, h, w):
        """寻找底部起始行，用于基础搜线"""
        self.start_row = self.start_left = self.start_right = None
        self.start_flag = False
        self.LeftPoints.clear()
        self.RightPoints.clear()
        self.CenterPoints.clear()

        min_valid_width = 50
        search_limit = int(h * 2 / 3)

        if self.Longest_White_Line_Top_Point is not None:
            anchor_col = self.Longest_White_Line_Top_Point.col
        else:
            anchor_col = w // 2

        for row in range(h - 1, search_limit, -1):
            if binary[row, anchor_col] == 0: continue

            # 向左扫
            l_idx = 0
            for c in range(anchor_col, -1, -1):
                if binary[row, c] == 0:
                    l_idx = c
                    break

            # 向右扫
            r_idx = w - 1
            for c in range(anchor_col, w):
                if binary[row, c] == 0:
                    r_idx = c
                    break

            if r_idx - l_idx > min_valid_width:
                self.start_row, self.start_left, self.start_right = row, l_idx, r_idx
                self.start_flag = True
                self.LeftPoints.append(Point(row, l_idx))
                self.RightPoints.append(Point(row, r_idx))
                break

    def search_boundaries(self, binary):
        """基础八邻域搜线（下半部分），用于计算中心线和辅助下角点"""
        self.LeftPoints_LostFlag = 0
        self.RightPoints_LostFlag = 0
        if self.start_row is None: return
        h, w = binary.shape
        visited = np.zeros_like(binary, np.uint8)

        directions_l = np.array([[0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]])
        directions_r = np.array([[0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1]])
        max_iteration = h * 3
        count = 0

        # 左边线
        cen_row, cen_col = self.start_row, self.start_left
        visited[cen_row, cen_col] = 1
        while cen_row > 0 and count < max_iteration:
            count += 1
            found = False
            for direction in range(8):
                dr, dc = directions_l[direction]
                nr, nc = cen_row + dr, cen_col + dc
                dr_n, dc_n = directions_l[(direction + 1) % 8]
                nr_n, nc_n = cen_row + dr_n, cen_col + dc_n

                if not (0 <= nr < h and 0 <= nc < w and 0 <= nr_n < h and 0 <= nc_n < w): continue
                if visited[nr, nc] == 0 and binary[nr, nc] == 255 and binary[nr_n, nc_n] == 0:
                    visited[nr, nc] = 1
                    self.LeftPoints.append(Point(nr, nc))
                    cen_row, cen_col = nr, nc
                    found = True
                    break
            if not found: break

        # 右边线
        cen_row, cen_col = self.start_row, self.start_right
        visited[cen_row, cen_col] = 1
        count = 0
        while cen_row > 0 and count < max_iteration:
            count += 1
            found = False
            for direction in range(8):
                dr, dc = directions_r[direction]
                nr, nc = cen_row + dr, cen_col + dc
                dr_n, dc_n = directions_r[(direction + 1) % 8]
                nr_n, nc_n = cen_row + dr_n, cen_col + dc_n

                if not (0 <= nr < h and 0 <= nc < w and 0 <= nr_n < h and 0 <= nc_n < w): continue
                if visited[nr, nc] == 0 and binary[nr, nc] == 255 and binary[nr_n, nc_n] == 0:
                    visited[nr, nc] = 1
                    self.RightPoints.append(Point(nr, nc))
                    cen_row, cen_col = nr, nc
                    found = True
                    break
            if not found: break

        # 计算丢线情况
        expected = self.start_row if self.start_row else h
        self.LeftPoints_LostNum = max(0, expected - len(self.LeftPoints))
        self.RightPoints_LostNum = max(0, expected - len(self.RightPoints))
        if expected > 0:
            if self.LeftPoints_LostNum / expected > self.LostThreshold: self.LeftPoints_LostFlag = 1
            if self.RightPoints_LostNum / expected > self.LostThreshold: self.RightPoints_LostFlag = 1

    def cal_cos(self, pre_point, cur_point, nex_point):
        """计算向量夹角余弦值"""
        x1, y1 = nex_point.col - cur_point.col, nex_point.row - cur_point.row
        x2, y2 = cur_point.col - pre_point.col, cur_point.row - pre_point.row
        norm_v1 = (x1**2 + y1**2)**0.5
        norm_v2 = (x2**2 + y2**2)**0.5
        if not (norm_v1 and norm_v2): return 1.0
        return (x1 * x2 + y1 * y2) / (norm_v1 * norm_v2)

    def find_down_corners(self, h, w):
        """寻找下角点 (出口点)"""
        K = 8
        cos_threshold = 0.6
        self.LeftDownCorner, self.RightDownCorner = None, None

        # Left Down
        if len(self.LeftPoints) > 2 * K + 1:
            min_cos, best_idx = 1.0, None
            for i in range(K, len(self.LeftPoints) - K - 1):
                if self.LeftPoints[i].row < h * 0.3: continue
                val = self.cal_cos(self.LeftPoints[i - K], self.LeftPoints[i], self.LeftPoints[i + K])
                if val < min_cos: min_cos, best_idx = val, i
            if min_cos < cos_threshold and best_idx is not None:
                self.LeftDownCorner = self.LeftPoints[best_idx]
                self.LeftPoints = self.LeftPoints[:best_idx + 1]

        # Right Down
        if len(self.RightPoints) > 2 * K + 1:
            min_cos, best_idx = 1.0, None
            for i in range(K, len(self.RightPoints) - K - 1):
                if self.RightPoints[i].row < h * 0.3: continue
                val = self.cal_cos(self.RightPoints[i - K], self.RightPoints[i], self.RightPoints[i + K])
                if val < min_cos: min_cos, best_idx = val, i
            if min_cos < cos_threshold and best_idx is not None:
                self.RightDownCorner = self.RightPoints[best_idx]
                self.RightPoints = self.RightPoints[:best_idx + 1]

    # =========================================================================
    # 核心修改区域：Top-Down 逐行扫描搜线法 (完全模拟 my_image.c)
    # =========================================================================

    def search_up_boundaries(self, binary, h, w):
        """
        从最长白列顶点开始，逐行向下扫描，寻找上边界。
        不再使用爬虫/边缘追踪，而是强制扫描，确保线必然能“进入”十字路口。
        """
        self.UpLeftPoints = []
        self.UpRightPoints = []

        if self.Longest_White_Line_Top_Point is None:
            return

        start_r = self.Longest_White_Line_Top_Point.row
        anchor_col = self.Longest_White_Line_Top_Point.col

        # 扫描范围：从顶点向下扫到图像中部
        # 太低了会碰到下角点，所以限制在 0.6*h 左右
        end_r = int(h * 0.65)

        # 逐行遍历 (Row-by-Row Scan)
        for r in range(start_r, end_r):
            if r >= h: break

            # 如果中轴就是黑的，说明路断了或者有障碍，跳过
            if binary[r, anchor_col] == 0:
                continue

            # --- 向左扫 ---
            l_found = False
            l_pos = 0
            # 从 anchor_col 向左遍历直到 0
            for c in range(anchor_col, 0, -1):
                if binary[r, c] == 0: # 遇到黑点，记录为边界
                    l_pos = c
                    l_found = True
                    break

            if l_found:
                self.UpLeftPoints.append(Point(r, l_pos))
            else:
                # 扫到边界都没黑点，记录图像边缘
                self.UpLeftPoints.append(Point(r, 0))

            # --- 向右扫 ---
            r_found = False
            r_pos = w - 1
            # 从 anchor_col 向右遍历直到 w-1
            for c in range(anchor_col, w - 1):
                if binary[r, c] == 0:
                    r_pos = c
                    r_found = True
                    break

            if r_found:
                self.UpRightPoints.append(Point(r, r_pos))
            else:
                self.UpRightPoints.append(Point(r, w - 1))

    def detect_upper_corner_from_scan(self, points, h, is_left=True):
        """
        分析扫描点集，寻找“横向突变点”。
        原理：
        1. 直道/弯道入弯前：边界比较陡峭（垂直），dx/dy 较小。
        2. 十字路口入口：边界突然变平（水平），dx/dy 剧增。
        """
        if len(points) < 10: return None

        best_idx = -1
        max_slope_diff = 0

        # 遍历点集，寻找拐点
        # 步长 step 用于平滑局部噪点，计算斜率变化
        step = 4

        for i in range(step, len(points) - step):
            p_prev = points[i - step]
            p_curr = points[i]
            p_next = points[i + step]

            # 过滤1：位置太靠下的不要（那是下角点的区域）
            if p_curr.row > h * 0.55: continue

            # 计算上半段斜率 (Vertical part) -> 应该是比较直的
            dx1 = abs(p_curr.col - p_prev.col)
            dy1 = abs(p_curr.row - p_prev.row) + 1e-5
            slope1 = dx1 / dy1 # cot(theta), 垂直线接近0

            # 计算下半段斜率 (Horizontal part) -> 进入十字，横向拉宽
            dx2 = abs(p_next.col - p_curr.col)
            dy2 = abs(p_next.row - p_curr.row) + 1e-5
            slope2 = dx2 / dy2 # 十字入口横向拉伸，这个值会很大

            # 判据：
            # 1. 上半段必须比较“竖” (slope1 小)
            # 2. 下半段必须比较“横” (slope2 大)
            # 3. 突变幅度 (slope2 - slope1) 要大

            # 弯道特征：slope1 和 slope2 都是逐渐变大的，差值不会突变
            # 十字特征：slope2 突然爆炸

            if slope1 < 1.0 and slope2 > 1.5: # 阈值可调
                diff = slope2 - slope1
                if diff > max_slope_diff:
                    max_slope_diff = diff
                    best_idx = i

        if best_idx != -1 and max_slope_diff > 1.0: # 确认突变足够明显
            return points[best_idx]
        return None

    def find_up_corners(self, binary, h, w):
        """主上角点检测函数"""
        self.LeftUpCorner = None
        self.RightUpCorner = None

        # 1. 执行逐行扫描
        self.search_up_boundaries(binary, h, w)

        # 2. 分析左扫描线
        self.LeftUpCorner = self.detect_upper_corner_from_scan(self.UpLeftPoints, h, is_left=True)

        # 3. 分析右扫描线
        self.RightUpCorner = self.detect_upper_corner_from_scan(self.UpRightPoints, h, is_left=False)

        # 4. 逻辑校验：上角点必须在下角点上方
        if self.LeftDownCorner and self.LeftUpCorner:
            if self.LeftUpCorner.row >= self.LeftDownCorner.row:
                self.LeftUpCorner = None

        if self.RightDownCorner and self.RightUpCorner:
            if self.RightUpCorner.row >= self.RightDownCorner.row:
                self.RightUpCorner = None

    def find_corners(self, binary, h, w):
        self.find_down_corners(h, w)
        self.find_up_corners(binary, h, w)

    # =========================================================================

    def bezier_fit(self, input_points, dt=0.01):
        output = []
        if len(input_points) != 4: return output
        t = 0
        while t <= 1.0 + 1e-6:
            row = (1 - t) ** 3 * input_points[0].row + 3 * (1 - t) ** 2 * t * input_points[1].row + 3 * (1 - t) * t ** 2 * input_points[2].row + t ** 3 * input_points[3].row
            col = (1 - t) ** 3 * input_points[0].col + 3 * (1 - t) ** 2 * t * input_points[1].col + 3 * (1 - t) * t ** 2 * input_points[2].col + t ** 3 * input_points[3].col
            output.append(Point(round(row), round(col)))
            t += dt
        return output

    def generate_bezier_center(self, h, w):
        self.CenterPoints.clear()
        if len(self.LeftPoints) < 1 or len(self.RightPoints) < 1: return
        def get_three_part_points(points):
            for p in points:
                p.row = max(0, min(p.row, h - 1))
                p.col = max(0, min(p.col, w - 1))
            n = len(points)
            return [points[0], points[n // 3], points[2 * n // 3], points[-1]]
        left_feature = get_three_part_points(self.LeftPoints)
        right_feature = get_three_part_points(self.RightPoints)
        self.bezier_input = []
        for l_p, r_p in zip(left_feature, right_feature):
            mid_row = (l_p.row + r_p.row) / 2
            mid_col = (l_p.col + r_p.col) / 2
            mid_row = max(0, min(round(mid_row), h - 1))
            mid_col = max(0, min(round(mid_col), w - 1))
            self.bezier_input.append(Point(round(mid_row), round(mid_col)))
        self.CenterPoints = self.bezier_fit(self.bezier_input)

    def process(self, frame):
        binary_frame, cropped_frame = self.preprocessing(frame)
        h, w = binary_frame.shape
        self.find_Longest_White_Line_Length(binary_frame)
        self.find_start_line(binary_frame, h, w)
        self.search_boundaries(binary_frame)
        self.find_corners(binary_frame, h, w)
        self.generate_bezier_center(h, w)
        return cropped_frame

# 3. 分析类
class Analyse:
    def __init__(self):
        self.sigma_left = 0.0
        self.sigma_right = 0.0
        self.sigma_center = 0.0

    def cal_sigma_of_all(self, tracker):
        def cal_var(points, dim):
            if len(points) < 2: return 0.0
            data = np.array([p.row if dim == 0 else p.col for p in points])
            return np.var(data)
        self.sigma_left = cal_var(tracker.LeftPoints, 1)
        self.sigma_right = cal_var(tracker.RightPoints, 1)
        self.sigma_center = cal_var(tracker.CenterPoints, 1)

    def process(self, tracker):
        self.cal_sigma_of_all(tracker)

# 4. 十字路口处理类（占位）
class Cross:
    def process(self):
        pass

# 5. 可视化类
class Visualize:
    def draw_points(self, frame, tracker, crosser):
        h, w = frame.shape[:2]

        # 1. 绘制最长白列 (粉色线)
        if tracker.Longest_White_Line_Top_Point is not None:
            cv2.line(frame,
                     (tracker.Longest_White_Line_Top_Point.col, h - 1),
                     (tracker.Longest_White_Line_Top_Point.col, tracker.Longest_White_Line_Top_Point.row),
                     (255, 0, 255), 2)

        # 2. 绘制基础边线 (绿色、蓝色)
        for p in tracker.LeftPoints: cv2.circle(frame, (p.col, p.row), 2, (0, 255, 0), -1)
        for p in tracker.RightPoints: cv2.circle(frame, (p.col, p.row), 2, (255, 0, 0), -1)

        # 3. [调试重点] 绘制上方扫描线 (青色、黄色)
        # 这就是 "Top-Down Scanning" 的直接结果，你可以看到它们是如何切入十字路口的
        for p in tracker.UpLeftPoints: cv2.circle(frame, (p.col, p.row), 1, (255, 255, 0), -1) # 青色
        for p in tracker.UpRightPoints: cv2.circle(frame, (p.col, p.row), 1, (0, 255, 255), -1) # 黄色

        # 4. 绘制中心线
        for i in range(len(tracker.CenterPoints) - 1):
            p1, p2 = tracker.CenterPoints[i], tracker.CenterPoints[i + 1]
            cv2.line(frame, (p1.col, p1.row), (p2.col, p2.row), (0, 0, 255), 2)
        return frame

    def draw_text(self, frame, tracker, analyser, crosser):
        font_scale = 0.4

        # 绘制角点 (带文字标签)
        corners = [
            (tracker.LeftDownCorner, (0, 255, 255), "LD"),   # 下角点 - 黄色
            (tracker.RightDownCorner, (0, 255, 255), "RD"),
            (tracker.LeftUpCorner, (255, 0, 255), "LU"),     # 上角点 - 紫色
            (tracker.RightUpCorner, (255, 0, 255), "RU")
        ]

        for p, color, label in corners:
            if p is not None:
                # 绘制实心点
                cv2.circle(frame, (p.col, p.row), 6, color, -1)
                # 绘制空心圈强调
                cv2.circle(frame, (p.col, p.row), 9, (0, 0, 255), 1)
                # 绘制标签
                cv2.putText(frame, label, (p.col + 10, p.row), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

        return frame

    def process(self, frame, tracker, analyser, crosser):
        self.draw_points(frame, tracker, crosser)
        self.draw_text(frame, tracker, analyser, crosser)
        return frame

# 主程序类
class Main:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.tracker = Track()
        self.analyser = Analyse()
        self.visualizer = Visualize()
        self.crosser = Cross()

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret: break

            # 核心处理
            cropped_frame = self.tracker.process(frame)
            self.analyser.process(self.tracker)

            # 可视化 (直接画在 cropped_frame 上)
            self.visualizer.process(cropped_frame, self.tracker, self.analyser, self.crosser)

            # 显示
            cv2.imshow('Processed View', cropped_frame)

            # 按 'q' 退出
            if cv2.waitKey(20) & 0xff == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = Main('cross1.mp4')
    app.run()