import cv2
import numpy as np
from enum import Enum

'''
Refactored Code Structure:
1. Point: Basic data structure.
2. Track (Model): Handles image processing, line searching, and basic feature extraction (Lost lines, Longest white line).
3. Analyser (Helper): Calculates mathematical statistics (Variance).
4. CrossRoad (Controller): Handles Crossroad logic and state machine.
5. Visualizer (View): Handles all drawing and text display.
6. Main: Orchestrates the entire flow.
'''

# 1. Define a 2D coordinate class.
class Point:
    def __init__(self, row, col):
        self.row = row
        self.col = col

# 2. Define a track data class (Model)
class Track:
    def __init__(self):
        self.up_chop_rate = 0.05
        self.down_chop_rate = 0.05

        # Edge point sets
        self.LeftPoints = []
        self.RightPoints = []
        self.CenterPoints = []
        self.bezier_input = []

        # Search parameters
        self.start_flag = True
        self.min_valid_width = 50
        self.start_row = None
        self.start_left = None
        self.start_right = None

        # Longest white line features
        self.Longest_White_Line_Top_Point = None
        self.Longest_White_Line_Length = 0

        # Lose line detection (New Feature)
        self.lose_L = 0
        self.lose_R = 0

    def crop_video_frame(self, frame):
        """Crop the frame to ROI."""
        height, width = frame.shape[:2]
        start_row = int(height * self.up_chop_rate)
        end_row = int(height * (1 - self.down_chop_rate))
        return frame[start_row:end_row, :]

    def find_start_line(self, binary):
        """Find the starting row and points at the bottom."""
        self.start_row = self.start_left = self.start_right = None
        h, w = binary.shape
        for row in range(h - 1, 0, -1):
            cols = np.where(binary[row] == 255)[0]
            cols = cols[cols < w]
            if len(cols) > 0 and (cols[-1] - cols[0]) >= self.min_valid_width:
                self.start_row, self.start_left, self.start_right = row, cols[0], cols[-1]
                self.LeftPoints.append(Point(row, cols[0]))
                self.RightPoints.append(Point(row, cols[-1]))
                break

    def find_Longest_White_Line_Length(self, binary):
        """Find the longest vertical white column (God's eye view feature)."""
        h, w = binary.shape[:2]
        best_row, best_col = h - 1, w // 2
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

    def search_lines(self, binary):
        """8-neighborhood boundary search."""
        if self.start_row is None:
            # If no start row, assume full loss
            self.lose_L = binary.shape[0]
            self.lose_R = binary.shape[0]
            return

        h, w = binary.shape
        Visited = np.zeros_like(binary, np.uint8)

        # Directions for Left and Right searches
        directions_L = np.array([[0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]])
        directions_R = np.array([[0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1]])
        MaxIteration = h * 3
        Count = 0

        # Search Left
        cen_row, cen_col = self.start_row, self.start_left
        Visited[cen_row, cen_col] = 1
        while cen_row > 0 and Count < MaxIteration:
            Count += 1
            found = False
            for dir in range(8):
                delta_row0, delta_col0 = directions_L[dir]
                delta_row1, delta_col1 = directions_L[(dir + 1) % 8]
                new_row0, new_col0 = cen_row + delta_row0, cen_col + delta_col0
                new_row1, new_col1 = cen_row + delta_row1, cen_col + delta_col1

                if not (0 <= new_row0 < h and 0 <= new_col0 < w and 0 <= new_row1 < h and 0 <= new_col1 < w):
                    continue

                if Visited[new_row0, new_col0] == 0 and binary[new_row0, new_col0] == 255 and binary[new_row1, new_col1] == 0:
                    Visited[new_row0, new_col0] = 1
                    self.LeftPoints.append(Point(new_row0, new_col0))
                    cen_row, cen_col = new_row0, new_col0
                    found = True
                    break
            if not found:
                break

        # Search Right
        cen_row, cen_col = self.start_row, self.start_right
        Visited[cen_row, cen_col] = 1
        Count = 0
        while cen_row > 0 and Count < MaxIteration:
            Count += 1
            found = False
            for dir in range(8):
                delta_row0, delta_col0 = directions_R[dir]
                delta_row1, delta_col1 = directions_R[(dir + 1) % 8]
                new_row0, new_col0 = cen_row + delta_row0, cen_col + delta_col0
                new_row1, new_col1 = cen_row + delta_row1, cen_col + delta_col1

                if not (0 <= new_row0 < h and 0 <= new_col0 < w and 0 <= new_row1 < h and 0 <= new_col1 < w):
                    continue

                if Visited[new_row0, new_col0] == 0 and binary[new_row0, new_col0] == 255 and binary[new_row1, new_col1] == 0:
                    Visited[new_row0, new_col0] = 1
                    self.RightPoints.append(Point(new_row0, new_col0))
                    cen_row, cen_col = new_row0, new_col0
                    found = True
                    break
            if not found:
                break

        # Calculate Lost Points (Logic from my_image.c)
        # Expected points is roughly the height of the image (since we scan from bottom up)
        # We subtract start_row because we only care about the path ahead
        if self.start_row:
            expected_points = self.start_row
        else:
            expected_points = h

        self.lose_L = max(0, expected_points - len(self.LeftPoints))
        self.lose_R = max(0, expected_points - len(self.RightPoints))

    def bezier_fit(self, input_points, dt=0.01):
        """Bezier curve calculation."""
        output = []
        if len(input_points) != 4:
            return output
        t = 0
        while t <= 1.0 + 1e-6:
            center_row = (1 - t) ** 3 * input_points[0].row + 3 * (1 - t) ** 2 * t * input_points[1].row + 3 * (1 - t) * t ** 2 * input_points[2].row + t ** 3 * input_points[3].row
            center_col = (1 - t) ** 3 * input_points[0].col + 3 * (1 - t) ** 2 * t * input_points[1].col + 3 * (1 - t) * t ** 2 * input_points[2].col + t ** 3 * input_points[3].col
            output.append(Point(round(center_row), round(center_col)))
            t += dt
        return output

    def generate_bezier_center(self, h, w):
        """Generate centerline using Bezier fitting."""
        self.CenterPoints.clear()
        if len(self.LeftPoints) < 1 or len(self.RightPoints) < 1: return

        def get_three_part_points(points):
            for p in points:
                p.row = max(0, min(p.row, h - 1))
                p.col = max(0, min(p.col, w - 1))
            n = len(points)
            return [points[0], points[n // 3], points[2 * n // 3], points[-1]]

        left_fiture = get_three_part_points(self.LeftPoints)
        right_fiture = get_three_part_points(self.RightPoints)
        self.bezier_input = []
        for l_p, r_p in zip(left_fiture, right_fiture):
            mid_row = (l_p.row + r_p.row) / 2
            mid_col = (l_p.col + r_p.col) / 2
            mid_row = max(0, min(round(mid_row), h - 1))
            mid_col = max(0, min(round(mid_col), w - 1))
            self.bezier_input.append(Point(round(mid_row), round(mid_col)))
        self.CenterPoints = self.bezier_fit(self.bezier_input)

    def process(self, frame):
        """Main processing pipeline for Track data."""
        self.LeftPoints.clear()
        self.RightPoints.clear()
        self.CenterPoints.clear()
        self.Longest_White_Line_Length = 0
        self.Longest_White_Line_Top_Point = None

        cropped_frame = self.crop_video_frame(frame)
        h, w = cropped_frame.shape[:2]
        gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        _, binary_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        self.find_Longest_White_Line_Length(binary_frame)
        self.find_start_line(binary_frame)
        self.search_lines(binary_frame) # Now includes lose line calculation
        self.generate_bezier_center(h, w)
        return cropped_frame

# 3. Analyser Class (Helper)
class Analyser:
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

# 4. CrossRoad Class (Controller) - Empty for now
class CrossRoad:
    class State(Enum):
        None_State = 0  # Normal
        Found = 1       # Found Crossroad
        Patching = 2    # Patching lines

    def __init__(self):
        self.state = self.State.None_State

    def process(self, track: Track):
        """Crossroad logic will be implemented here."""
        pass

# 5. Visualizer Class (View)
class Visualizer:
    def draw_process(self, frame, tracker: Track, analyser: Analyser, cross: CrossRoad):
        """Draw everything on the frame."""
        h, w = frame.shape[:2]

        # 1. Draw Longest White Line (Purple)
        if tracker.Longest_White_Line_Top_Point is not None:
            cv2.line(frame,
                     (tracker.Longest_White_Line_Top_Point.col, h - 1),
                     (tracker.Longest_White_Line_Top_Point.col, tracker.Longest_White_Line_Top_Point.row),
                     (255, 0, 255), 2)
            cv2.circle(frame,
                       (tracker.Longest_White_Line_Top_Point.col, tracker.Longest_White_Line_Top_Point.row),
                       5, (255, 0, 255), -1)

        # 2. Draw Boundary Points (Left: Green, Right: Red)
        for p in tracker.LeftPoints:
            cv2.circle(frame, (p.col, p.row), 2, (0, 255, 0), -1)
        for p in tracker.RightPoints:
            cv2.circle(frame, (p.col, p.row), 2, (0, 0, 255), -1)

        # 3. Draw Center Line (Red Line)
        for i in range(len(tracker.CenterPoints) - 1):
            p1, p2 = tracker.CenterPoints[i], tracker.CenterPoints[i + 1]
            cv2.line(frame, (p1.col, p1.row), (p2.col, p2.row), (0, 0, 255), 2)

        # 4. Draw Text Info (Variances, Lose Lines, State)
        # Note: Since we are drawing on the cropped frame in this function,
        # coordinates might need adjustment if you want to draw on the original frame.
        # But per your structure, we return the drawn frame.

        text_info = [
            f"LVar:{analyser.sigma_left:.1f} RVar:{analyser.sigma_right:.1f}",
            f"Lose L: {tracker.lose_L} Lose R: {tracker.lose_R}",
            f"State: {cross.state.name}"
        ]

        y = 30
        for txt in text_info:
            # Draw on top-left
            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y += 30

        return frame

# 6. Main Class (Orchestrator)
class Main:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.track = Track()
        self.analyser = Analyser()
        self.cross = CrossRoad()
        self.vis = Visualizer()

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 1. Processing (Model)
            cropped_frame = self.track.process(frame)

            # 2. Analysis (Stats)
            self.analyser.process(self.track)

            # 3. Logic (Controller)
            self.cross.process(self.track)

            # 4. Visualization (View)
            # We draw on cropped_frame. If you want to draw on original,
            # you need to handle coordinate offsets.
            result_frame = self.vis.draw_process(cropped_frame, self.track, self.analyser, self.cross)

            cv2.imshow('SmartCar Main', result_frame)
            if cv2.waitKey(30) & 0xff == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace 'demo.avi' with your actual video path
    app = Main('demo.avi')
    app.run()