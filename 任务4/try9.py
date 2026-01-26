import cv2
import numpy as np

'''
Code Function:
Advanced Crossroad Detection Logic (v6.0)
1. "Center Anchor Row-by-Row Scanning" (NumPy Optimized)
2. "Slope Filter" + "Border Escape Check" to distinguish crossroads from sharp turns.
3. Robust handling for oblique entry and intersection exit.
'''

# 1. Basic Data Structure
class Point:
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def point2cv(self):
        """Converts point to OpenCV (x, y) tuple."""
        return self.col, self.row

# 2. Core Track Tracking Class
class Track:
    def __init__(self):
        # Image cropping parameters
        self.up_chop_rate = 0
        self.down_chop_rate = 0.3

        # Basic point sets (Lower part, used for start line & visualization)
        self.LeftPoints = []
        self.RightPoints = []

        # Scan point sets (Upper part, used for upper corners)
        self.ScanLeftPoints = []
        self.ScanRightPoints = []

        # Lost line flags
        self.LeftPoints_LostNum = 0
        self.RightPoints_LostNum = 0
        self.LeftPoints_LostFlag = 0
        self.RightPoints_LostFlag = 0
        self.LostThreshold = 0.2

        # Center line and Bezier
        self.CenterPoints = []
        self.bezier_input = []

        # Start row and longest white column
        self.start_flag = False
        self.start_row = None
        self.start_left = None
        self.start_right = None
        self.Longest_White_Line_Top_Point = None
        self.Longest_White_Line_Length = 0

        # Corners
        self.LeftDownCorner = None
        self.RightDownCorner = None
        self.LeftUpCorner = None
        self.RightUpCorner = None

        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def preprocessing(self, frame):
        """Preprocessing: Crop -> Grayscale -> Gaussian -> Otsu -> Morphology"""
        cropped_frame = self.crop_video_frame(frame)
        gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        _, binary_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Logic: Closing to fill gaps (make road solid) -> Opening to remove noise (clean background)
        binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, self.kernel)
        binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, self.kernel)
        return binary_frame, cropped_frame

    def crop_video_frame(self, frame):
        height, width = frame.shape[:2]
        start_row = int(height * self.up_chop_rate)
        end_row = int(height * (1 - self.down_chop_rate))
        return frame[start_row:end_row, :]

    def find_Longest_White_Line_Length(self, binary):
        """Find the longest white column (Robust Anchor)"""
        self.Longest_White_Line_Length = 0
        self.Longest_White_Line_Top_Point = None
        h, w = binary.shape
        best_row, best_col = h - 1, w // 2

        # Coarse scan, step size 4
        step = 4
        for col in range(0, w, step):
            current_row = 0
            for row in range(h - 1, 0, -1):
                if binary[row, col] == 0:
                    current_row = row
                    break
            if current_row < best_row:
                best_row = current_row
                best_col = col

        self.Longest_White_Line_Top_Point = Point(best_row, best_col)
        self.Longest_White_Line_Length = h - best_row
        return self.Longest_White_Line_Top_Point, self.Longest_White_Line_Length

    # -------------------------------------------------------------------------
    # Core: Optimized Top-Down Scanning (NumPy Accelerated)
    # -------------------------------------------------------------------------

    def search_up_boundaries(self, binary, h, w):
        """
        [Optimized] Use NumPy slicing for fast scanning.
        Scans from the top of the white line down to the bottom to find upper boundaries.
        """
        # Initialize
        self.ScanLeftPoints = []
        self.ScanRightPoints = []

        up_start_row = None
        min_valid_width = 50

        # 1. Determine Anchor
        if self.Longest_White_Line_Top_Point is not None:
            anchor_col = self.Longest_White_Line_Top_Point.col
            search_start = self.Longest_White_Line_Top_Point.row
        else:
            anchor_col = w // 2
            search_start = 0

        # 2. Set Search Range
        # Extend to h-10 to catch corners when exiting the intersection
        search_end = h - 10
        if search_start >= search_end:
            return

        # -------------------------------------------------
        # Phase 1: Find valid start line (Loop 1)
        # -------------------------------------------------
        for row in range(search_start, search_end):
            # Left Slice
            points_left = binary[row, :anchor_col][::-1]
            left_indices = np.where(points_left == 0)[0]
            l_idx = anchor_col - left_indices[0] if left_indices.size > 0 else 0

            # Right Slice
            points_right = binary[row, anchor_col:]
            right_indices = np.where(points_right == 0)[0]
            r_idx = anchor_col + right_indices[0] if right_indices.size > 0 else w - 1

            # Width Check
            if r_idx - l_idx > min_valid_width:
                up_start_row = row
                self.ScanLeftPoints.append(Point(row, l_idx))
                self.ScanRightPoints.append(Point(row, r_idx))
                break

        # Safety Check: If no valid start line found, return immediately
        if up_start_row is None:
            return

        # -------------------------------------------------
        # Phase 2: Fast Tracking (Loop 2)
        # -------------------------------------------------
        # Start from up_start_row + 1 to avoid duplication
        for row in range(up_start_row + 1, search_end):
            # Scan Left
            points_left = binary[row, :anchor_col][::-1]
            left_indices = np.where(points_left == 0)[0]
            l_idx = anchor_col - left_indices[0] if left_indices.size > 0 else 0
            self.ScanLeftPoints.append(Point(row, l_idx))

            # Scan Right
            points_right = binary[row, anchor_col:]
            right_indices = np.where(points_right == 0)[0]
            r_idx = anchor_col + right_indices[0] if right_indices.size > 0 else w - 1
            self.ScanRightPoints.append(Point(row, r_idx))

    def detect_upper_corner_with_slope_filter(self, points, h, w, is_left=True):
        """
        [Advanced Corner Detection]
        Features:
        1. Border Escape Check: Rejects sharp turns (which don't touch image edges).
        2. Relaxed Slope/Height: Allows oblique entry and tracking while exiting.
        """
        if len(points) < 10: return None

        best_idx = -1
        max_score = 0
        step = 6 # Step size for slope calc

        for i in range(step, len(points) - step):
            p_prev = points[i - step] # Upper arm (Entry)
            p_curr = points[i]        # Corner
            p_next = points[i + step] # Lower arm (Track)

            # ---------------------------------------------------------
            # 1. Border Escape Check (Crucial for Sharp Turn Rejection)
            # ---------------------------------------------------------
            # A real crossroad arm must touch the image edge.
            margin = 5

            if is_left:
                # Left arm must start from left edge (col ~ 0)
                if p_prev.col > margin:
                    continue # Not at edge -> Sharp Turn -> Skip!
            else:
                # Right arm must start from right edge (col ~ w)
                if p_prev.col < w - margin:
                    continue # Not at edge -> Sharp Turn -> Skip!

            # ---------------------------------------------------------
            # 2. Position & Geometry Filters
            # ---------------------------------------------------------
            # Relax height limit to 0.95h to track corner when exiting
            if p_curr.row > h * 0.95: continue

            # Calculate vectors
            dy1 = p_curr.row - p_prev.row
            dx1 = p_curr.col - p_prev.col
            dy2 = p_next.row - p_curr.row
            dx2 = p_next.col - p_curr.col

            # Arm Angle (Entry)
            if dx1 == 0: angle_arm = 90
            else: angle_arm = np.degrees(np.arctan(abs(dy1/dx1)))

            # Relax angle limit to 50 degrees to allow Oblique Entry
            # (Border check already ensures safety)
            if angle_arm > 50:
                continue

            # Track Angle (Exit) - Should be relatively vertical
            if dy2 == 0: angle_track = 90
            else: angle_track = np.degrees(np.arctan(abs(dx2/dy2)))

            if angle_track > 45:
                continue

            # ---------------------------------------------------------
            # 3. Mutation Detection
            # ---------------------------------------------------------
            diff_x = 0
            if is_left:
                diff_x = p_next.col - p_prev.col
            else:
                diff_x = p_prev.col - p_next.col

            # Valid mutation threshold
            if diff_x > 8:
                score = diff_x
                if score > max_score:
                    max_score = score
                    best_idx = i

        if best_idx != -1:
            return points[best_idx]
        return None

    def find_up_corners(self, binary, h, w):
        """Main upper corner detection function"""
        self.LeftUpCorner = None
        self.RightUpCorner = None

        # 1. Execute optimized scanning
        self.search_up_boundaries(binary, h, w)

        # 2. Detection with filters (Pass 'w' for border check)
        self.LeftUpCorner = self.detect_upper_corner_with_slope_filter(self.ScanLeftPoints, h, w, is_left=True)
        self.RightUpCorner = self.detect_upper_corner_with_slope_filter(self.ScanRightPoints, h, w, is_left=False)

        # 3. Interlock Check (Upper must be above Lower)
        margin = 15
        if self.LeftDownCorner and self.LeftUpCorner:
            if self.LeftUpCorner.row >= self.LeftDownCorner.row - margin:
                self.LeftUpCorner = None

        if self.RightDownCorner and self.RightUpCorner:
            if self.RightUpCorner.row >= self.RightDownCorner.row - margin:
                self.RightUpCorner = None

    # -------------------------------------------------------------------------
    # Auxiliary Functions
    # -------------------------------------------------------------------------

    def find_start_line(self, binary, h, w):
        """(Old logic) Find bottom start row for visualization & lower corners"""
        self.start_row = None
        self.LeftPoints.clear()
        self.RightPoints.clear()

        if self.Longest_White_Line_Top_Point is None: anchor = w // 2
        else: anchor = self.Longest_White_Line_Top_Point.col

        for r in range(h-1, h//2, -1):
            if binary[r, anchor] == 0: continue
            l, r_idx = 0, w-1
            # Simple linear search for start line
            for c in range(anchor, -1, -1):
                if binary[r, c] == 0: l = c; break
            for c in range(anchor, w):
                if binary[r, c] == 0: r_idx = c; break

            if r_idx - l > 50:
                self.start_row = r; self.start_left = l; self.start_right = r_idx
                self.LeftPoints.append(Point(r, l)); self.RightPoints.append(Point(r, r_idx))
                break

    def search_boundaries(self, binary):
        """(Old logic) Simple 8-neighborhood search for bottom visualization"""
        if not self.start_row: return
        h, w = binary.shape
        # Left
        curr_r, curr_c = self.start_row, self.start_left
        for _ in range(300):
            curr_r -= 1
            if curr_r < 0: break
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
        """Find lower corners using Cosine law"""
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
            output.append(Point(int(row), int(col)))
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

        self.find_Longest_White_Line_Length(binary)
        self.find_start_line(binary, h, w)
        self.search_boundaries(binary)
        self.find_down_corners(h, w)

        # New logic call: Pass 'w' to corners
        self.find_up_corners(binary, h, w)

        self.generate_bezier_center(h, w)
        return crop

# 3. Analysis Class
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

# 4. Crossroad Processing Class (Placeholder)
class Cross:
    def process(self):
        # TODO: Implement line patching when both corners are detected
        pass

# 5. Visualization Class
class Visualize:
    def draw_points(self, frame, tracker, crosser):
        h, w = frame.shape[:2]

        # Longest white column (Pink line)
        if tracker.Longest_White_Line_Top_Point:
            top_pos = tracker.Longest_White_Line_Top_Point.point2cv()
            bottom_pos = (top_pos[0], h - 1)
            cv2.line(frame, bottom_pos, top_pos, (255, 0, 255), 2)

        # Basic boundary lines (Green, Blue)
        for p in tracker.LeftPoints: cv2.circle(frame, p.point2cv(), 1, (0, 255, 0), -1)
        for p in tracker.RightPoints: cv2.circle(frame, p.point2cv(), 1, (255, 0, 0), -1)

        # Scan lines (Debug, Cyan/Yellow)
        # Note: Corrected variable name from ScanLeft to ScanLeftPoints
        for p in tracker.ScanLeftPoints: cv2.circle(frame, p.point2cv(), 1, (255, 255, 0), -1)
        for p in tracker.ScanRightPoints: cv2.circle(frame, p.point2cv(), 1, (0, 255, 255), -1)

        # Center line
        for i in range(len(tracker.CenterPoints)-1):
            p1, p2 = tracker.CenterPoints[i], tracker.CenterPoints[i + 1]
            cv2.line(frame, p1.point2cv(), p2.point2cv(), (0, 0, 255), 2)
        return frame

    def draw_text(self, frame, tracker, analyser, crosser):
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Optimized Corner Visualization loop
        corner_configs = [
            (tracker.LeftDownCorner, (0, 255, 255), "LD"),
            (tracker.RightDownCorner, (0, 255, 255), "RD"),
            (tracker.LeftUpCorner, (255, 0, 255), "LU"),
            (tracker.RightUpCorner, (255, 0, 255), "RU")
        ]

        for p, color, txt in corner_configs:
            if p:
                pos = p.point2cv()
                cv2.circle(frame, pos, 10, (0, 0, 255), 2) # Red ring
                cv2.circle(frame, pos, 5, color, -1)       # Solid core
                cv2.putText(frame, txt, (pos[0]+15, pos[1]), font, 0.5, color, 1)
        return frame

    def process(self, frame, tracker, analyser, crosser):
        self.draw_points(frame, tracker, crosser)
        self.draw_text(frame, tracker, analyser, crosser)
        return frame

# Main Program Class
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