import cv2
import numpy as np


class TrackLineDetector:
    def __init__(self):
        self.left_points = []
        self.right_points = []
        self.image_width = 0
        self.image_height = 0
        self.scan_step = 1
        self.window_size = 100

    def preprocess_image(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        ret, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def search_line_points(self, binary_image):
        self.image_height, self.image_width = binary_image.shape[:2]
        self.left_points = []
        self.right_points = []

        max_white_pixels = 0
        for col in range(self.image_width):

            white_count = np.sum(binary_image[:, col] == 255)

            if white_count > max_white_pixels:
                max_white_pixels = white_count
                longest_white_col = col

        for y in range(self.image_height-1,10, -self.scan_step):
            if binary_image[y, longest_white_col] == 255:

               for x in range(longest_white_col, max(0, longest_white_col - self.window_size), -1):
                    if binary_image[y, x] == 0:
                        self.left_points.append((x, y))
                        break

               for x in range(longest_white_col, min(self.image_width, longest_white_col + self.window_size),+1):
                   if binary_image[y, x] == 0:
                       self.right_points.append((x, y))
                       break

        return self.left_points, self.right_points

    def draw_line_points(self, frame):

        for (x, y) in self.left_points:
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        for (x, y) in self.right_points:
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        return frame

    def process_video(self, video_path, show_process=True, save_output=False):
        """处理视频文件，逐帧检测赛道线"""
        cap = cv2.VideoCapture(video_path)


        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 预处理图像
            binary = self.preprocess_image(frame)

            # 搜索赛道点
            left, right= self.search_line_points(binary)

            # 绘制结果
            result = self.draw_line_points(frame.copy())

            if show_process:
                cv2.imshow('Binary Image', binary)
                cv2.imshow('Track Detection Result', result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if save_output:
            out.release()
        cv2.destroyAllWindows()

        return left, right


if __name__ == "__main__":
    detector = TrackLineDetector()

    detector.scan_step = 1
    detector.window_size = 100

    detector.process_video('D:/pythonproject/project4/res/sample.avi', show_process=True, save_output=False)




