import cv2
import numpy as np
from enum import Enum

'''
Function:
8-neighbor edge search, Bezier centerline fitting, drawing edges/centerline.
Longest white column detection, lost line detection.
Variance calculation and visualization.
NEW: Crossroad State Machine Framework.
'''

'''
Code Structure:
Point: Basic data structure
Track: Track analysis (boundary line/center line), and some simple track character analysis (longest white line/lost line)
Cross: Handle the condition of crossing (State Machine Implementation)
Visualize: Visualize special line and some text of data characteristics, including Cross state
Main: Orchestrates the entire code
'''

# 1. Define a 2D coordinate class.
class Point:
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def point2cv(self):
        # Convert point (row,col) to (x,y)
        return self.col, self.row

# 2. Responsible for longest white line, boundary lines and the centerline.
class Track:
    def __init__(self):
        # About crop
        self.up_chop_rate = 0     # Proportion of the top to be cropped
        self.down_chop_rate = 0.3 # Proportion of the bottom to be cropped

        # Edge point sets for left and right track boundaries
        self.LeftPoints = []
        self.RightPoints = []

        # Lost points
        self.LeftPoints_LostNum = 0
        self.RightPoints_LostNum = 0
        self.LeftPoints_LostFlag = 0 # 0 not lost; 1 lost
        self.RightPoints_LostFlag = 0
        self.LostThreshold = 0.2

        # About center points
        self.CenterPoints = []        # Set of points for the centerline
        self.bezier_input = []        # Control points for Bezier curve fitting

        # Start line
        self.start_flag = False      # Starting row flag (identifies the bottom-most valid row)
        self.start_row = None        # Row index of the starting row
        self.start_left = None       # Column index of the left edge in the starting row
        self.start_right = None      # Column index of the right edge in the starting row

        # The longest white line
        self.Longest_White_Line_Top_Point = None # The peak point of the longest white line
        self.Longest_White_Line_Length = 0

        # Corners
        self.LeftDownCorner = None
        self.RightDownCorner = None
        self.LeftUpCorner = None
        self.RightUpCorner = None

        # Use in morphological opening operation
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def preprocessing(self, frame):
        # Preprocessing: Cropping -> Converting to Grayscale Image -> Gaussian Filtering -> Binarization -> Morphological Operation (Removing Isolated Points)
        cropped_frame = self.crop_video_frame(frame)
        gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)      # Gaussian filtering, remove high frequency noise
        _, binary_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Otsu Binarization

        # Morphological operation
        binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, self.kernel) # Remove isolated small black dots
        binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, self.kernel) # Remove isolated small white dots
        return binary_frame, cropped_frame

    def crop_video_frame(self, frame):
        height, width = frame.shape[:2]
        start_row = int(height * self.up_chop_rate)
        end_row = int(height * (1 - self.down_chop_rate))
        return frame[start_row:end_row, :]

    def find_start_line(self, binary, h, w):
        # Clear the values of the track
        self.start_row = self.start_left = self.start_right = None
        self.start_flag = False
        self.LeftPoints.clear()
        self.RightPoints.clear()
        self.CenterPoints.clear()

        min_valid_width = 50   # Minimum width threshold for valid blocks (noise filtering)
        # Search white pixel in the edge of the track, search from the column of the longest white line
        search_limit = int(h * 2 / 3)

        if self.Longest_White_Line_Top_Point is not None:
            anchor_col = self.Longest_White_Line_Top_Point.col
        else:
            anchor_col = w // 2

        # When a horizontal line is not able to be the start line, try the line above it. But it must be on the bottom part of the image
        for row in range(h - 1, search_limit, -1):
            if binary[row, anchor_col] == 0:
                continue
            # Find left edge point
            points_left_to_the_anchor_col = binary[row, :anchor_col][::-1]
            left_black_indices = np.where(points_left_to_the_anchor_col == 0)[0]
            if len(left_black_indices) > 0:
                l_idx = anchor_col - left_black_indices[0]
            else:
                l_idx = 0
            # Find right edge point
            points_right_to_the_anchor_col = binary[row, anchor_col:]
            right_black_indices = np.where(points_right_to_the_anchor_col == 0)[0]
            if len(right_black_indices) > 0:
                r_idx = anchor_col + right_black_indices[0] - 1
            else:
                r_idx = w - 1
            # Validate the width and set the start line
            if r_idx - l_idx > min_valid_width:
                self.start_row, self.start_left, self.start_right = row, l_idx, r_idx
                self.start_flag = True
                self.LeftPoints.append(Point(row, l_idx))
                self.RightPoints.append(Point(row, r_idx))
                break

    def find_Longest_White_Line_Length(self, binary):
        self.Longest_White_Line_Length = 0
        self.Longest_White_Line_Top_Point = None
        h, w = binary.shape
        best_row, best_col = h - 1, w // 2
        step = 4  # Search step of col
        for col in range(0, w, step):   # Search from bottom to top
            current_row = 0
            for row in range(h - 1, 0, -1):
                if binary[row, col] == 0:  # Hit black pixel (boundary)
                    current_row = row
                    break
            if current_row < best_row:    # Update the row and col of Longest_White_Line_Top_Point
                best_row = current_row
                best_col = col
        self.Longest_White_Line_Top_Point = Point(best_row, best_col)
        self.Longest_White_Line_Length = h - best_row
        return self.Longest_White_Line_Top_Point, self.Longest_White_Line_Length

    def search_boundaries(self, binary):
        self.LeftPoints_LostFlag = 0
        self.RightPoints_LostFlag = 0
        # Use 8-neighborhood search method
        if self.start_row is None:
            return
        h, w = binary.shape
        visited = np.zeros_like(binary, np.uint8) # 0 unvisited; 1 visited

        directions_l = np.array([ # Directions for left boundary
            [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]
        ])
        directions_r = np.array([ # Directions for right boundary
            [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1]
        ])
        max_iteration = h * 3
        count = 0

        # Search left boundary
        cen_row, cen_col = self.start_row, self.start_left
        visited[cen_row, cen_col] = 1
        while cen_row > 0 and count < max_iteration:
            count += 1
            found = False
            for direction in range(8):
                delta_row0, delta_col0 = directions_l[direction]
                delta_row1, delta_col1 = directions_l[(direction + 1) % 8]
                new_row0 = cen_row + delta_row0
                new_col0 = cen_col + delta_col0
                new_row1 = cen_row + delta_row1
                new_col1 = cen_col + delta_col1
                if not (0 <= new_row0 < h and 0 <= new_col0 < w and 0 <= new_row1 < h and 0 <= new_col1 < w):
                    continue
                # When the first point has not been searched, first point is white and next point is black
                if visited[new_row0, new_col0] == 0 and binary[new_row0, new_col0] == 255 and binary[new_row1, new_col1] == 0:
                    visited[new_row0, new_col0] = 1
                    self.LeftPoints.append(Point(new_row0, new_col0))
                    cen_row, cen_col = new_row0, new_col0
                    found = True
                    break
            if not found:
                break

        # Search right boundary
        cen_row, cen_col = self.start_row, self.start_right
        visited[cen_row, cen_col] = 1
        count = 0
        while cen_row > 0 and count < max_iteration:
            count += 1
            found = False
            for direction in range(8):
                delta_row0, delta_col0 = directions_r[direction]
                delta_row1, delta_col1 = directions_r[(direction + 1) % 8]
                new_row0 = cen_row + delta_row0
                new_col0 = cen_col + delta_col0
                new_row1 = cen_row + delta_row1
                new_col1 = cen_col + delta_col1
                if not(0 <= new_row0 < h and 0 <= new_col0 < w and 0 <= new_row1 < h and 0 <= new_col1 < w):
                    continue
                if visited[new_row0, new_col0] == 0 and binary[new_row0, new_col0] == 255 and binary[new_row1, new_col1] == 0:
                    visited[new_row0, new_col0] = 1
                    self.RightPoints.append(Point(new_row0, new_col0))
                    cen_row, cen_col = new_row0, new_col0
                    found = True
                    break
            if not found:
                break

        # Calculate lost points
        if self.start_row:
            expected_points = self.start_row
        else:
            expected_points = h

        self.LeftPoints_LostNum = max(0, expected_points - len(self.LeftPoints))
        self.RightPoints_LostNum = max(0, expected_points - len(self.RightPoints))
        if expected_points > 0:
            if self.LeftPoints_LostNum / expected_points > self.LostThreshold:
                self.LeftPoints_LostFlag = 1
            if self.RightPoints_LostNum / expected_points > self.LostThreshold:
                self.RightPoints_LostFlag = 1

    def cal_cos(self, pre_point, cur_point, nex_point):
        # Calculate the cosine of the angle between vectors
        x1, y1 = nex_point.col - cur_point.col, nex_point.row - cur_point.row
        x2, y2 = cur_point.col - pre_point.col, cur_point.row - pre_point.row
        norm_v1 = (x1**2 + y1**2)**0.5
        norm_v2 = (x2**2 + y2**2)**0.5
        if not (norm_v1 and norm_v2):
            return 1
        else:
            return (x1 * x2 + y1 * y2) / (norm_v1 * norm_v2)

    def find_down_corners(self, h, w):
        # Method: K-value correlation method
        K = 8
        cos_threshold = 0.5
        self.LeftDownCorner, self.RightDownCorner = None, None

        # Find Left Down Corner
        if len(self.LeftPoints) > 2 * K + 1:
            min_cosine = 1
            min_cosine_index = None
            for i in range(K, len(self.LeftPoints) - K - 1):
                if self.LeftPoints[i].row < h * 0.2:
                    continue
                pre_point, cur_point, nex_point = self.LeftPoints[i - K], self.LeftPoints[i], self.LeftPoints[i + K]
                current_cosine = self.cal_cos(pre_point, cur_point, nex_point)
                if current_cosine < min_cosine:
                    min_cosine = current_cosine
                    min_cosine_index = i
            if min_cosine < cos_threshold and min_cosine_index is not None:
                self.LeftDownCorner = self.LeftPoints[min_cosine_index]
                self.LeftPoints = self.LeftPoints[:min_cosine_index + 1]

        # Find Right Down Corner
        if len(self.RightPoints) > 2 * K + 1:
            min_cosine = 1
            min_cosine_index = None
            for i in range(K, len(self.RightPoints) - K - 1):
                if self.RightPoints[i].row < h * 0.2:
                    continue
                pre_point, cur_point, nex_point = self.RightPoints[i - K], self.RightPoints[i], self.RightPoints[i + K]
                current_cosine = self.cal_cos(pre_point, cur_point, nex_point)
                if current_cosine < min_cosine:
                    min_cosine = current_cosine
                    min_cosine_index = i
            if min_cosine < cos_threshold and min_cosine_index is not None:
                self.RightDownCorner = self.RightPoints[min_cosine_index]
                self.RightPoints = self.RightPoints[:min_cosine_index + 1]

    def find_corners(self, binary, h, w):
        self.find_down_corners(h, w)
        # self.find_up_corners(binary,h,w) # Can be added if needed

    def bezier_fit(self, input_points, dt=0.01):
        # Bezier curve fitting
        output = []
        if len(input_points) != 4:
            # print("Invalid number of control points")
            return output
        t = 0
        while t <= 1.0 + 1e-6:
            center_row = (1 - t) ** 3 * input_points[0].row + 3 * (1 - t) ** 2 * t * input_points[1].row + 3 * (1 - t) * t ** 2 * input_points[2].row + t ** 3 * input_points[3].row
            center_col = (1 - t) ** 3 * input_points[0].col + 3 * (1 - t) ** 2 * t * input_points[1].col + 3 * (1 - t) * t ** 2 * input_points[2].col + t ** 3 * input_points[3].col
            output.append(Point(round(center_row), round(center_col)))
            t += dt
        return output

    def generate_bezier_center(self, h, w):
        # Generate center line using Bezier
        self.CenterPoints.clear()
        if len(self.LeftPoints) < 1 or len(self.RightPoints) < 1:
            return
        def get_three_part_points(points):
            # Return start, 1/3, 2/3, and end points
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
        # Main processing flow
        binary_frame, cropped_frame = self.preprocessing(frame)
        h, w = binary_frame.shape
        self.find_Longest_White_Line_Length(binary_frame)
        self.find_start_line(binary_frame, h, w)
        self.search_boundaries(binary_frame)
        self.find_corners(binary_frame, h, w)
        self.generate_bezier_center(h, w)
        return cropped_frame

# 3. Responsible for variance calculation and visualization.
class Analyse:
    def __init__(self):
        self.sigma_left = 0.0
        self.sigma_right = 0.0
        self.sigma_center = 0.0

    def cal_sigma_of_all(self, tracker):
        def cal_var(points, dim):
            if len(points) < 2:
                return 0.0
            data = np.array([p.row if dim == 0 else p.col for p in points])
            return np.var(data)
        self.sigma_left = cal_var(tracker.LeftPoints, 1)
        self.sigma_right = cal_var(tracker.RightPoints, 1)
        self.sigma_center = cal_var(tracker.CenterPoints, 1)

    def process(self, tracker):
        self.cal_sigma_of_all(tracker)

# 4. Responsible for the crossroad (State Machine)
# Ref: Task 4 PDF - Using Enum and Switch-Case (Simulated)
class Cross:
    # State Enum
    class CrossStep(Enum):
        NONE = 0
        Fix = 1 # State for patching lines and checking exit

    # Mode Enum (Less critical for center-patching, but kept for debug)
    class CrossMode(Enum):
        NONE = 0
        Left = 1
        Right = 2
        Straight = 3

    def __init__(self):
        self.step = self.CrossStep.NONE
        self.mode = self.CrossMode.NONE
        self.debug_points = [] # For visualization of temporary points
        self.track_half_width = 70 # Half width of the track (tunable parameter)

    def process(self, track):
        # Python switch-case (state machine)
        if self.step == self.CrossStep.NONE:
            # Entry Judgment Operator (判入算子) [cite: 9, 23, 61]
            if self._check_entry(track):
                self.step = self.CrossStep.Fix
                # Mode determination is less critical for center-patching but can be logged
                # self.mode = ...
            return False

        elif self.step == self.CrossStep.Fix:
            # Center Line Patching Operator (中线补线算子) [cite: 10, 24, 65]
            self._patch_lines(track)

            # Exit Judgment Operator (判出算子) [cite: 11, 25]
            if self._check_exit(track):
                self.step = self.CrossStep.NONE
                self.mode = self.CrossMode.NONE
                self.debug_points = [] # Clear debug points
            return True

        return False

    def _check_entry(self, track):
        # Entry Logic: Detect if near lines are lost BUT far exit is visible
        # 1. Check if both sides have significant lost lines (e.g., entering the crossroad void)
        is_lost_near = track.LeftPoints_LostFlag and track.RightPoints_LostFlag

        # 2. Check if a valid "Exit" block exists at the top of the image
        # (Simulating: Check if we can find a white block at the top 1/3 of the image)
        # This assumes the camera can see the road on the other side of the crossroad.
        has_far_exit = False
        if track.Longest_White_Line_Top_Point is not None:
             if track.Longest_White_Line_Top_Point.row < 50: # Threshold for "Far"
                 has_far_exit = True

        return is_lost_near and has_far_exit

    def _patch_lines(self, track):
        # Logic: Connect Start (Car) and End (Exit) -> Virtual Center -> Virtual Edges

        # 1. Define Start Point (Near)
        # Usually the bottom-center of the image (assuming vehicle is centered)
        # Or use the last valid center point from the bottom
        h = 240 # Assuming image height, should ideally get from track/frame
        w = 320 # Assuming image width
        start_point = Point(h - 1, w // 2)

        # 2. Define End Point (Far)
        # Use the furthest valid point found (e.g., from Longest White Line or valid CenterPoints)
        # Here we use Longest_White_Line_Top_Point as the target anchor
        if track.Longest_White_Line_Top_Point is not None:
            end_point = track.Longest_White_Line_Top_Point
        else:
            # Fallback: Forward extension if no exit seen (blind move)
            end_point = Point(0, w // 2)

        # 3. Generate Virtual Center Line (Linear Interpolation for simplicity)
        # You can replace this with Bezier if you want a smoother path
        virtual_center_points = []
        steps = 20
        for i in range(steps):
            t = i / float(steps - 1)
            # Linear interpolation: P = (1-t)*P0 + t*P1
            r = int((1 - t) * start_point.row + t * end_point.row)
            c = int((1 - t) * start_point.col + t * end_point.col)
            virtual_center_points.append(Point(r, c))

        # Store for visualization
        self.debug_points = virtual_center_points

        # 4. Reconstruct Virtual Edges (Back-calculate edges from center)
        # Clear detected (broken) edges
        track.LeftPoints.clear()
        track.RightPoints.clear()

        for p in virtual_center_points:
            # Virtual Left = Center - HalfWidth
            # Virtual Right = Center + HalfWidth
            # Boundary checks should be applied in a real scenario
            l_col = max(0, p.col - self.track_half_width)
            r_col = min(w - 1, p.col + self.track_half_width)

            track.LeftPoints.append(Point(p.row, l_col))
            track.RightPoints.append(Point(p.row, r_col))

        # 5. Update CenterPoints to match our virtual line
        track.CenterPoints = virtual_center_points

    def _check_exit(self, track):
        # Exit Logic: Check if the "End Point" is close to the vehicle
        # If the top point of the white line is no longer "Far" (meaning we entered the next track segment)
        # Or if the original edge detection starts finding valid lines again (LostFlag clears)

        is_close_to_target = False
        if track.Longest_White_Line_Top_Point is not None:
            # If the "furthest" point is actually very close (e.g. > 180), we might be hitting a wall or done
            # But normally for exit, we check if valid edges reappear.
            pass

        # Simple exit condition: If side lines are found again naturally
        # (This requires the raw search to run before this machine every frame)
        lines_recovered = (not track.LeftPoints_LostFlag) and (not track.RightPoints_LostFlag)

        return lines_recovered
# 5. Responsible for visualize everything
class Visualize:
    def draw_points(self, frame, tracker, crosser):
        # Brief: Visualize everything
        h, w = frame.shape[:2]

        # Draw the longest white line
        if tracker.Longest_White_Line_Top_Point is not None:
            cv2.line(frame,
                     (tracker.Longest_White_Line_Top_Point.col, h - 1),
                     (tracker.Longest_White_Line_Top_Point.col, tracker.Longest_White_Line_Top_Point.row),
                     (255, 0, 255),
                     2)

        # Visualize edges
        for p in tracker.LeftPoints:
            cv2.circle(frame, p.point2cv(), 2, (0, 255, 0), -1)
        for p in tracker.RightPoints:
            cv2.circle(frame, p.point2cv(), 2, (255, 0, 0), -1)

        # Visualize Center Line
        for i in range(len(tracker.CenterPoints) - 1):
            p1, p2 = tracker.CenterPoints[i], tracker.CenterPoints[i + 1]
            cv2.line(frame, p1.point2cv(), p2.point2cv(), (0, 0, 255), 2)

        # Visualize Cross Debug Points (e.g. patch control points)
        for p in crosser.debug_points:
             cv2.circle(frame, p.point2cv(), 4, (0, 255, 255), -1)

        return frame

    def draw_text(self, frame, tracker, analyser, crosser):
        # Visualize data analysis
        font_scale = 0.4
        font_thickness = 1
        text = [
            f"LVar:{analyser.sigma_left:.1f}",
            f"RVar:{analyser.sigma_right:.1f}",
            f"CVar:{analyser.sigma_center:.1f}",
            f"LLost:{tracker.LeftPoints_LostFlag:d}",
            f"RLost:{tracker.RightPoints_LostFlag:d}",
            # Visualize Cross State and Mode
            f"CrossStep: {crosser.step.name}",
            f"CrossMode: {crosser.mode.name}"
        ]
        y = 30
        for txt in text:
            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 255), font_thickness)
            y += 20

        # Draw corners
        corners = [tracker.LeftDownCorner, tracker.RightDownCorner, tracker.LeftUpCorner, tracker.RightUpCorner]
        for p in corners:
            if p is not None:
                cv2.circle(frame, p.point2cv(), 6, (0, 255, 255), -1)
                cv2.circle(frame, p.point2cv(), 8, (0, 0, 255), 2)
        return frame

    def process(self, frame, tracker, analyser, crosser):
        self.draw_points(frame, tracker, crosser)
        self.draw_text(frame, tracker, analyser, crosser)
        return frame

# Orchestrator
class Main:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.tracker = Track()      # Instantiate Track class
        self.analyser = Analyse()   # Instantiate Analyse class
        self.visualizer = Visualize()
        self.crosser = Cross()      # Instantiate Cross State Machine

    def run(self):
        # Main loop
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 1. Basic Tracking (Find edges, corners)
            cropped_frame = self.tracker.process(frame)

            # 2. Crossroad Logic (State Machine: Entry -> Fix -> Exit)
            # This must happen after finding raw edges but before analysis/visualization
            self.crosser.process(self.tracker)

            # 3. Analysis (Variance, etc.)
            self.analyser.process(self.tracker)

            # 4. Visualization
            self.visualizer.process(cropped_frame, self.tracker, self.analyser, self.crosser)

            cv2.imshow('Video', frame)
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = Main('cross1.mp4')
    app.run()