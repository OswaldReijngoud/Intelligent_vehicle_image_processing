import cv2
import numpy as np

'''
代码功能：
基于“中心锚点扫描” + “严格向量斜率约束”的十字路口检测
核心：利用“横臂水平度”强制过滤急弯，解决角点闪烁和误判问题
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

        # 基础点集 (用于可视化和兼容)
        self.LeftPoints = []
        self.RightPoints = []

        # 扫描点集 (用于找上角点)
        self.ScanLeftPoints = []
        self.ScanRightPoints = []

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
        """预处理：裁剪 -> 灰度 -> 高斯 -> Otsu二值化 -> 形态学"""
        cropped_frame = self.crop_video_frame(frame)
        gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        _, binary_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 闭运算填补缝隙，开运算去除噪点
        binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, self.kernel)
        binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, self.kernel)
        return binary_frame, cropped_frame

    def crop_video_frame(self, frame):
        height, width = frame.shape[:2]
        start_row = int(height * self.up_chop_rate)
        end_row = int(height * (1 - self.down_chop_rate))
        return frame[start_row:end_row, :]

    def find_Longest_White_Line_Length(self, binary):
        """寻找最长白列（作为最稳健的参考锚点）"""
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

    # -------------------------------------------------------------------------
    # 核心：基于锚点的逐行扫描 + 严格几何筛选
    # -------------------------------------------------------------------------

    def scan_boundaries_from_anchor(self, binary, h, w):
        """
        以最长白列顶点为起点，向下逐行扫描，获取上半部分的边缘点集。
        这些点集将专门用于检测上角点。
        """
        self.ScanLeftPoints = []
        self.ScanRightPoints = []

        if self.Longest_White_Line_Top_Point is None:
            return

        start_r = self.Longest_White_Line_Top_Point.row
        anchor_col = self.Longest_White_Line_Top_Point.col

        # 扫描区间：从顶点向下扫到图像 2/3 处
        # 避开最顶部的噪点区(start_r + 5)
        # 避开最底部的车身区
        end_r = int(h * 0.65)

        for r in range(start_r + 5, end_r):
            if r >= h: break

            # 如果中轴是黑的，说明路断了，停止扫描
            if binary[r, anchor_col] == 0:
                continue

            # 向左扫
            l_pos = 0
            found_l = False
            for c in range(anchor_col, 0, -1):
                if binary[r, c] == 0:
                    l_pos = c # 记录黑白交界点
                    found_l = True
                    break
            if found_l: self.ScanLeftPoints.append(Point(r, l_pos))

            # 向右扫
            r_pos = w - 1
            found_r = False
            for c in range(anchor_col, w - 1):
                if binary[r, c] == 0:
                    r_pos = c
                    found_r = True
                    break
            if found_r: self.ScanRightPoints.append(Point(r, r_pos))

    def detect_upper_corner_with_slope_filter(self, points, h, is_left=True):
        """
        带【斜率过滤器】的角点检测。
        这是区分急弯和十字路口的关键。
        """
        if len(points) < 10: return None

        best_idx = -1
        max_score = 0

        # 步长：用于计算局部斜率，步长越大越抗噪，但会平滑掉尖角
        # 这里的点集是稠密的（逐行），所以取 5-8 比较合适
        step = 6

        for i in range(step, len(points) - step):
            p_prev = points[i - step] # 上方点 (Arm部分)
            p_curr = points[i]        # 当前点 (Corner)
            p_next = points[i + step] # 下方点 (Track部分)

            # 1. 基础位置过滤
            if p_curr.row > h * 0.6: continue # 太靠下的一律不要

            # 2. 计算两个向量
            # Vec1 (Arm): 从 Prev 指向 Curr (扫描线下来的方向)
            # 在图像上，这是赛道的远端边缘
            dy1 = p_curr.row - p_prev.row # 正值
            dx1 = p_curr.col - p_prev.col

            # Vec2 (Track): 从 Curr 指向 Next (继续向下的方向)
            # 这是赛道的近端垂直边缘
            dy2 = p_next.row - p_curr.row # 正值
            dx2 = p_next.col - p_curr.col

            # 3. 计算斜率 (Slope Analysis)
            # 注意：图像坐标系 y 向下。
            # 横臂斜率 (Arm Slope): dy/dx.
            # 十字路口的横臂应该是水平的，所以 dy/dx 应该很小，或者 dx 很大。
            # 垂直边斜率 (Track Slope): dx/dy.
            # 垂直边应该是竖直的，所以 dx/dy 应该很小。

            # === 核心过滤器：拒绝急弯 ===
            # 计算 Arm 部分相对于水平线的角度
            if dx1 == 0: angle_arm = 90
            else: angle_arm = np.degrees(np.arctan(abs(dy1/dx1)))

            # 判据：十字路口的 Arm 必须足够“平”。
            # 如果 Arm 的角度 > 30度 (说明它是斜着下来的)，那就是弯道！
            # 只有 < 25~30度，才认为是横向的十字入口。
            if angle_arm > 30:
                continue

            # === 核心过滤器2：拒绝非拐点 ===
            # 计算 Track 部分相对于垂直线的角度
            if dy2 == 0: angle_track = 90
            else: angle_track = np.degrees(np.arctan(abs(dx2/dy2)))

            # Track 部分必须足够“竖”。如果 > 40度，说明路歪了或者不是边缘
            if angle_track > 45:
                continue

            # 4. 几何突变检测
            # 如果通过了斜率测试，说明形状符合 "L" 型（十字特征）。
            # 现在找拐弯最急的那个点。

            # 方法：计算两个向量的角度差，或者简单的 dx 变化率
            # 十字路口处，dx 会发生剧烈变化。

            # 计算“外扩程度”
            # 十字路口：上方点很靠外，下方点很靠内
            # Left: Prev.col << Next.col (小 -> 大) ? 不对，左边x小。
            # Left: Prev.col (小) ... 诶不对，上方是十字臂，x应该很小(左)或很大(右)?
            # 让我们理一下：
            # 左上角点：
            #   Prev (上方, 十字臂): col 很小 (在图像左侧边缘附近)
            #   Curr (角点): col 突然变大 (回到赛道垂直边缘)
            #   Next (下方): col 保持较大 (垂直向下)

            # 右上角点：
            #   Prev (上方, 十字臂): col 很大 (在图像右侧边缘附近)
            #   Curr (角点): col 突然变小 (回到赛道垂直边缘)
            #   Next (下方): col 保持较小

            diff_x = 0
            if is_left:
                # 左侧：上方的 x 应该比 下方的 x 小很多 (因为上方是横向出去的)
                # 弯道：上方的 x 和 下方的 x 差别是渐变的
                diff_x = p_next.col - p_prev.col
            else:
                # 右侧：上方的 x 应该比 下方的 x 大很多
                diff_x = p_prev.col - p_next.col

            # 只有当横向突变足够大时，才置信
            if diff_x > 10: # 像素阈值
                # 分数可以是 角度锐利度 + 横向突变程度
                # 这里简单用 diff_x 作为置信度
                score = diff_x
                if score > max_score:
                    max_score = score
                    best_idx = i

        if best_idx != -1:
            return points[best_idx]
        return None

    def find_up_corners(self, binary, h, w):
        """主上角点检测流程"""
        self.LeftUpCorner = None
        self.RightUpCorner = None

        # 1. 扫描
        self.scan_boundaries_from_anchor(binary, h, w)

        # 2. 带过滤器的检测
        self.LeftUpCorner = self.detect_upper_corner_with_slope_filter(self.ScanLeftPoints, h, is_left=True)
        self.RightUpCorner = self.detect_upper_corner_with_slope_filter(self.ScanRightPoints, h, is_left=False)

        # 3. 物理约束：上下角点互锁
        # 如果找到的上角点比下角点还靠下（Y值更大），那绝对是误判（通常是把下角点当成了上角点）
        margin = 15
        if self.LeftDownCorner and self.LeftUpCorner:
            if self.LeftUpCorner.row >= self.LeftDownCorner.row - margin:
                self.LeftUpCorner = None # 丢弃

        if self.RightDownCorner and self.RightUpCorner:
            if self.RightUpCorner.row >= self.RightDownCorner.row - margin:
                self.RightUpCorner = None

    # -------------------------------------------------------------------------
    # 辅助功能：下角点、中心线、旧式搜线(仅画图用)
    # -------------------------------------------------------------------------

    def find_start_line(self, binary, h, w):
        """(兼容旧代码) 寻找底部起始行"""
        self.start_row = None
        self.LeftPoints.clear()
        self.RightPoints.clear()

        if self.Longest_White_Line_Top_Point is None: anchor = w // 2
        else: anchor = self.Longest_White_Line_Top_Point.col

        for r in range(h-1, h//2, -1):
            if binary[r, anchor] == 0: continue
            l, r_idx = 0, w-1
            for c in range(anchor, -1, -1):
                if binary[r, c] == 0:
                    l = c; break
            for c in range(anchor, w):
                if binary[r, c] == 0:
                    r_idx = c; break
            if r_idx - l > 50:
                self.start_row = r; self.start_left = l; self.start_right = r_idx
                self.LeftPoints.append(Point(r, l)); self.RightPoints.append(Point(r, r_idx))
                break

    def search_boundaries(self, binary):
        """(兼容旧代码) 简单八邻域，用于画绿线/蓝线给用户看"""
        if not self.start_row: return
        h, w = binary.shape
        # Left
        curr_r, curr_c = self.start_row, self.start_left
        for _ in range(300):
            curr_r -= 1
            if curr_r < 0: break
            # 简单跟踪
            found = False
            for offset in range(-2, 3):
                if 0 <= curr_c + offset < w and binary[curr_r, curr_c + offset] == 0:
                     if curr_c + offset + 1 < w and binary[curr_r, curr_c + offset + 1] == 255:
                        curr_c += offset
                        self.LeftPoints.append(Point(curr_r, curr_c))
                        found = True; break
            if not found: self.LeftPoints.append(Point(curr_r, curr_c))

        # Right
        curr_r, curr_c = self.start_row, self.start_right
        for _ in range(300):
            curr_r -= 1
            if curr_r < 0: break
            found = False
            for offset in range(-2, 3):
                 if 0 <= curr_c + offset < w and binary[curr_r, curr_c + offset] == 0:
                     if curr_c + offset - 1 >= 0 and binary[curr_r, curr_c + offset - 1] == 255:
                        curr_c += offset
                        self.RightPoints.append(Point(curr_r, curr_c))
                        found = True; break
            if not found: self.RightPoints.append(Point(curr_r, curr_c))

    def cal_cos(self, p1, p2, p3):
        x1, y1 = p2.col - p1.col, p2.row - p1.row
        x2, y2 = p3.col - p2.col, p3.row - p2.row
        n1 = (x1**2 + y1**2)**0.5
        n2 = (x2**2 + y2**2)**0.5
        if n1*n2 == 0: return 1
        return (x1*x2 + y1*y2)/(n1*n2)

    def find_down_corners(self, h, w):
        """寻找下角点"""
        K = 8
        self.LeftDownCorner, self.RightDownCorner = None, None

        if len(self.LeftPoints) > 20:
            best_cos = 0.6
            for i in range(K, len(self.LeftPoints) - K):
                if self.LeftPoints[i].row < h * 0.3: continue
                val = self.cal_cos(self.LeftPoints[i-K], self.LeftPoints[i], self.LeftPoints[i+K])
                if val < best_cos:
                    best_cos = val
                    self.LeftDownCorner = self.LeftPoints[i]

        if len(self.RightPoints) > 20:
            best_cos = 0.6
            for i in range(K, len(self.RightPoints) - K):
                if self.RightPoints[i].row < h * 0.3: continue
                val = self.cal_cos(self.RightPoints[i-K], self.RightPoints[i], self.RightPoints[i+K])
                if val < best_cos:
                    best_cos = val
                    self.RightDownCorner = self.RightPoints[i]

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
        def get_sample(pts):
            n = len(pts)
            return [pts[0], pts[n//3], pts[2*n//3], pts[-1]]
        l_pts = get_sample(self.LeftPoints)
        r_pts = get_sample(self.RightPoints)
        self.bezier_input = []
        for l, r in zip(l_pts, r_pts):
            self.bezier_input.append(Point((l.row+r.row)//2, (l.col+r.col)//2))
        self.CenterPoints = self.bezier_fit(self.bezier_input)

    def process(self, frame):
        binary, crop = self.preprocessing(frame)
        h, w = binary.shape

        # 1. 找最长白列 (稳健锚点)
        self.find_Longest_White_Line_Length(binary)

        # 2. 找基础线 (用于可视化和下角点)
        self.find_start_line(binary, h, w)
        self.search_boundaries(binary)

        # 3. 找下角点
        self.find_down_corners(h, w)

        # 4. 找上角点 (使用新的 slope-filter 算法)
        self.find_up_corners(binary, h, w)

        # 5. 生成中线
        self.generate_bezier_center(h, w)

        return crop

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

        # 最长白列
        if tracker.Longest_White_Line_Top_Point:
            cv2.line(frame, (tracker.Longest_White_Line_Top_Point.col, h),
                     (tracker.Longest_White_Line_Top_Point.col, tracker.Longest_White_Line_Top_Point.row), (255, 0, 255), 2)

        # 基础边线
        for p in tracker.LeftPoints: cv2.circle(frame, (p.col, p.row), 1, (0, 255, 0), -1)
        for p in tracker.RightPoints: cv2.circle(frame, (p.col, p.row), 1, (255, 0, 0), -1)

        # 扫描线 (调试用，青色/黄色) - 看看上角点扫描情况
        for p in tracker.ScanLeftPoints: cv2.circle(frame, (p.col, p.row), 1, (255, 255, 0), -1)
        for p in tracker.ScanRightPoints: cv2.circle(frame, (p.col, p.row), 1, (0, 255, 255), -1)

        # 中线
        for i in range(len(tracker.CenterPoints)-1):
            cv2.line(frame, (tracker.CenterPoints[i].col, tracker.CenterPoints[i].row),
                     (tracker.CenterPoints[i+1].col, tracker.CenterPoints[i+1].row), (0, 0, 255), 2)
        return frame

    def draw_text(self, frame, tracker, analyser, crosser):
        font = cv2.FONT_HERSHEY_SIMPLEX
        corners = [
            (tracker.LeftDownCorner, (0, 255, 255), "LD"),
            (tracker.RightDownCorner, (0, 255, 255), "RD"),
            (tracker.LeftUpCorner, (255, 0, 255), "LU"), # 紫色上角点
            (tracker.RightUpCorner, (255, 0, 255), "RU")
        ]
        for p, color, txt in corners:
            if p:
                cv2.circle(frame, (p.col, p.row), 10, (0, 0, 255), 2) # 红圈
                cv2.circle(frame, (p.col, p.row), 5, color, -1)       # 实心
                cv2.putText(frame, txt, (p.col+15, p.row), font, 0.5, color, 1)
        return frame

    def process(self, frame, tracker, analyser, crosser):
        self.draw_points(frame, tracker, crosser)
        self.draw_text(frame, tracker, analyser, crosser)
        return frame

# 主程序
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
            crop = self.tracker.process(frame)
            self.analyser.process(self.tracker)
            self.visualizer.process(crop, self.tracker, self.analyser, self.crosser)
            cv2.imshow('Processed', crop)
            if cv2.waitKey(20) & 0xFF == ord('q'): break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = Main('cross1.mp4')
    app.run()