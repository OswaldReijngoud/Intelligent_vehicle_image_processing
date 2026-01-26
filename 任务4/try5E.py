import cv2
import numpy as np

'''
Code Function:
Crossroad corner detection based on "center anchor row-by-row scanning" + "strict vector slope constraint".
Core: Using "horizontal arm levelness" to forcibly filter sharp curves, solving corner flickering and misjudgment problems.
'''

# 1. Basic Data Structure
class Point:
    def __init__(self, row, col):
        self.row = row
        self.col = col
    def to_tuple(self):
        return (self.col, self.row)

# 2. Core Track Tracking Class
class Track:
    def __init__(self):
        # Image cropping parameters
        self.up_chop_rate = 0
        self.down_chop_rate = 0.3

        # Basic point sets (Used for visualization and compatibility)
        self.LeftPoints = []
        self.RightPoints = []

        # Scan point sets (Used for finding upper corners)
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
        """Preprocessing: Crop -> Grayscale -> Gaussian -> Otsu Binarization -> Morphology"""
        cropped_frame = self.crop_video_frame(frame)
        gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        _, binary_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Closing to fill gaps, Opening to remove noise
        binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, self.kernel)
        binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, self.kernel)
        return binary_frame, cropped_frame

    def crop_video_frame(self, frame):
        height, width = frame.shape[:2]
        start_row = int(height * self.up_chop_rate)
        end_row = int(height * (1 - self.down_chop_rate))
        return frame[start_row:end_row, :]

    def find_Longest_White_Line_Length(self, binary):
        """Find the longest white column (as the most robust reference anchor)"""
        self.Longest_White_Line_Length = 0
        self.Longest_White_Line_Top_Point = None
        h, w = binary.shape
        best_row, best_col = h - 1, w // 2

        # Coarse scan, step size 4
        step = 4
        for col in range(0, w, step):
            current_row = 0
            # Find the first black pixel from bottom to top
            for row in range(h - 1, 0, -1):
                if binary[row, col] == 0:
                    current_row = row
                    break
            # Update the highest point (smaller row means higher)
            if current_row < best_row:
                best_row = current_row
                best_col = col

        self.Longest_White_Line_Top_Point = Point(best_row, best_col)
        self.Longest_White_Line_Length = h - best_row
        return self.Longest_White_Line_Top_Point, self.Longest_White_Line_Length

    # -------------------------------------------------------------------------
    # Core: Anchor-based row-by-row scanning + Strict geometric filtering
    # -------------------------------------------------------------------------

    def scan_boundaries_from_anchor(self, binary, h, w):
        """
        Start from the top of the longest white column, scan downwards row by row to get the edge points of the upper part.
        These points will be specifically used to detect upper corners.
        """
        self.ScanLeftPoints = []
        self.ScanRightPoints = []

        if self.Longest_White_Line_Top_Point is None:
            return

        start_r = self.Longest_White_Line_Top_Point.row
        anchor_col = self.Longest_White_Line_Top_Point.col

        # Scanning range: Scan from vertex down to roughly 2/3 of the image
        # Avoid the top noise area (start_r + 5)
        # Avoid the bottom car body area
        end_r = int(h * 0.65)

        for r in range(start_r + 5, end_r):
            if r >= h: break

            # If the central axis is black, the road is broken, stop scanning
            if binary[r, anchor_col] == 0:
                continue

            # Scan left
            l_pos = 0
            found_l = False
            for c in range(anchor_col, 0, -1):
                if binary[r, c] == 0:
                    l_pos = c # Record black-white boundary point
                    found_l = True
                    break
            if found_l: self.ScanLeftPoints.append(Point(r, l_pos))

            # Scan right
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
        Corner detection with [Slope Filter].
        This is the key to distinguishing between sharp curves and crossroads.
        """
        if len(points) < 10: return None

        best_idx = -1
        max_score = 0

        # Step size: Used to calculate local slope. Larger step is more noise-resistant but might smooth out sharp corners.
        # Since points are dense (row-by-row), 5-8 is appropriate.
        step = 6

        for i in range(step, len(points) - step):
            p_prev = points[i - step] # Upper point (Arm part)
            p_curr = points[i]        # Current point (Corner)
            p_next = points[i + step] # Lower point (Track part)

            # 1. Basic position filtering
            if p_curr.row > h * 0.6: continue # Discard if too far down

            # 2. Calculate two vectors
            # Vec1 (Arm): From Prev to Curr (Direction of scanning line coming down)
            # On the image, this is the far edge of the track
            dy1 = p_curr.row - p_prev.row # Positive value
            dx1 = p_curr.col - p_prev.col

            # Vec2 (Track): From Curr to Next (Direction continuing downwards)
            # This is the near vertical edge of the track
            dy2 = p_next.row - p_curr.row # Positive value
            dx2 = p_next.col - p_curr.col

            # 3. Slope Analysis
            # Note: Image coordinate system y is downwards.
            # Arm Slope: dy/dx.
            # The arm of a crossroad should be horizontal, so dy/dx should be small, or dx large.
            # Vertical edge slope (Track Slope): dx/dy.
            # The vertical edge should be vertical, so dx/dy should be small.

            # === Core Filter: Reject Sharp Curves ===
            # Calculate angle of Arm part relative to horizontal line
            if dx1 == 0: angle_arm = 90
            else: angle_arm = np.degrees(np.arctan(abs(dy1/dx1)))

            # Criteria: The Arm of a crossroad must be "flat" enough.
            # If Arm angle > 30 degrees (means it comes down diagonally), it's a curve!
            # Only < 25~30 degrees is considered a horizontal crossroad entrance.
            if angle_arm > 30:
                continue

            # === Core Filter 2: Reject Non-inflection Points ===
            # Calculate angle of Track part relative to vertical line
            if dy2 == 0: angle_track = 90
            else: angle_track = np.degrees(np.arctan(abs(dx2/dy2)))

            # Track part must be "vertical" enough. If > 45 degrees, the road is skewed or it's not an edge.
            if angle_track > 45:
                continue

            # 4. Geometric Mutation Detection
            # If slope test passed, shape matches "L" (Crossroad feature).
            # Now find the sharpest turn point.

            # Method: Calculate angle difference between two vectors, or simply dx rate of change.
            # At crossroad, dx changes drastically.

            # Calculate "Outward Expansion Degree"
            # Crossroad: Upper point is very far out, lower point is very far in

            diff_x = 0
            if is_left:
                # Left side: Upper x should be much smaller than Lower x (because upper went out horizontally)
                # Curve: Difference between Upper x and Lower x is gradual
                diff_x = p_next.col - p_prev.col
            else:
                # Right side: Upper x should be much larger than Lower x
                diff_x = p_prev.col - p_next.col

            # Only confident when horizontal mutation is large enough
            if diff_x > 10: # Pixel threshold
                # Score can be angle sharpness + horizontal mutation degree.
                # Here simply use diff_x as confidence.
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

        # 1. Execute row-by-row scanning
        self.scan_boundaries_from_anchor(binary, h, w)

        # 2. Detection with filters
        self.LeftUpCorner = self.detect_upper_corner_with_slope_filter(self.ScanLeftPoints, h, is_left=True)
        self.RightUpCorner = self.detect_upper_corner_with_slope_filter(self.ScanRightPoints, h, is_left=False)

        # 3. Physical Constraint: Upper/Lower Corner Interlock
        # If found upper corner is lower than lower corner (larger Y), it is definitely a misjudgment (usually taking lower corner as upper)
        margin = 15
        if self.LeftDownCorner and self.LeftUpCorner:
            if self.LeftUpCorner.row >= self.LeftDownCorner.row - margin:
                self.LeftUpCorner = None # Discard

        if self.RightDownCorner and self.RightUpCorner:
            if self.RightUpCorner.row >= self.RightDownCorner.row - margin:
                self.RightUpCorner = None

    # -------------------------------------------------------------------------
    # Auxiliary Functions: Lower corners, center line, old-style line search (For drawing only)
    # -------------------------------------------------------------------------

    def find_start_line(self, binary, h, w):
        """(Compatible with old code) Find bottom start row"""
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
        """(Compatible with old code) Simple 8-neighborhood search, used to draw green/blue lines for user"""
        if not self.start_row: return
        h, w = binary.shape
        # Left
        curr_r, curr_c = self.start_row, self.start_left
        for _ in range(300):
            curr_r -= 1
            if curr_r < 0: break
            # Simple tracking
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
        """Find lower corners"""
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

        # 1. Find longest white column (Robust Anchor)
        self.find_Longest_White_Line_Length(binary)

        # 2. Find basic lines (For visualization and lower corners)
        self.find_start_line(binary, h, w)
        self.search_boundaries(binary)

        # 3. Find lower corners
        self.find_down_corners(h, w)

        # 4. Find upper corners (Use new slope-filter algorithm)
        self.find_up_corners(binary, h, w)

        # 5. Generate center line
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
        pass

# 5. Visualization Class
class Visualize:
    def draw_points(self, frame, tracker, crosser):
        h, w = frame.shape[:2]

        # Longest white column (Pink line)
        if tracker.Longest_White_Line_Top_Point:
            cv2.line(frame, (tracker.Longest_White_Line_Top_Point.col, h),
                     (tracker.Longest_White_Line_Top_Point.col, tracker.Longest_White_Line_Top_Point.row), (255, 0, 255), 2)

        # Basic boundary lines (Green, Blue)
        for p in tracker.LeftPoints: cv2.circle(frame, (p.col, p.row), 1, (0, 255, 0), -1)
        for p in tracker.RightPoints: cv2.circle(frame, (p.col, p.row), 1, (255, 0, 0), -1)

        # Scan lines (Debug, Cyan/Yellow) - Check upper corner scanning status
        for p in tracker.ScanLeftPoints: cv2.circle(frame, (p.col, p.row), 1, (255, 255, 0), -1)
        for p in tracker.ScanRightPoints: cv2.circle(frame, (p.col, p.row), 1, (0, 255, 255), -1)

        # Center line
        for i in range(len(tracker.CenterPoints)-1):
            cv2.line(frame, (tracker.CenterPoints[i].col, tracker.CenterPoints[i].row),
                     (tracker.CenterPoints[i+1].col, tracker.CenterPoints[i+1].row), (0, 0, 255), 2)
        return frame

    def draw_text(self, frame, tracker, analyser, crosser):
        font = cv2.FONT_HERSHEY_SIMPLEX
        corners = [
            (tracker.LeftDownCorner, (0, 255, 255), "LD"),
            (tracker.RightDownCorner, (0, 255, 255), "RD"),
            (tracker.LeftUpCorner, (255, 0, 255), "LU"), # Purple upper corner
            (tracker.RightUpCorner, (255, 0, 255), "RU")
        ]
        for p, color, txt in corners:
            if p:
                cv2.circle(frame, (p.col, p.row), 10, (0, 0, 255), 2) # Red circle
                cv2.circle(frame, (p.col, p.row), 5, color, -1)       # Solid
                cv2.putText(frame, txt, (p.col+15, p.row), font, 0.5, color, 1)
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