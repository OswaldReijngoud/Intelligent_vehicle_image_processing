import cv2
import numpy as np

class Point:
    def __init__(self, row, col):
        self.x = col
        self.y = row

class Tracking:
    def __init__(self):
        self._top_cut = 70
        self._bottom_cut = 20
        self.LeftPoints = []
        self.RightPoints = []
        self.min_start_width = 50#最小白条的长度，防止噪声之类的

    def process(self, bin_img):
        h,w = bin_img.shape
        self.LeftPoints.clear()
        self.RightPoints.clear()

        self.to_find_start = True#表示不知道从哪里是起始行的开始，需要去寻找

        for row in range(h - self._bottom_cut - 1,self._top_cut - 1,-1):#每一行开始寻找
            white_blocks =self.search_white_blocks( bin_img , row, w)#返回得到了白色条带的最左点最右点

            if not white_blocks:#如果这一行没有找到白色条带，就去下一行找
                continue

            if self.to_find_start:#已经找到白条了，需要确定是不是起始行
                longest = max (white_blocks, key = lambda b :b[1] - b[0])#white_blocks得到的结果是一个列表，通过max找到横坐标相距最远的两个点，找到最长的白条
                if( longest[1]-longest[0] ) >= self.min_start_width :#如果找到的白条比设定的阈值长，则避免了噪点，确定了起始行的最左点和最右点
                    self.LeftPoints.append(Point(row, longest[0]))
                    self.RightPoints.append(Point(row, longest[1]))
                    self.to_find_start = False#做个标记，表示之后不需要再找起始行了


            else:#找到了白条，但是不是起始行了
                last_left = self.LeftPoints[-1].x#去获取目前已经有的点的最后一个，表示“上一行”的点的横坐标
                last_right = self.RightPoints[-1].x

                nearest_left = min(white_blocks, key = lambda b:abs( b[0]- last_left))#去找点集里面和“上一行”横坐标最接近的点，这样可以找到最接近的点，保证连通性
                nearest_right = min(white_blocks, key = lambda b:abs( b[1] - last_right))

                self.LeftPoints.append(Point(row,nearest_left[0]))
                self.RightPoints.append(Point(row,nearest_right[1]))


    def search_white_blocks(self, bin_img , row, w):#固定某一行去寻找白条，找到白条的起始点和终止点横坐标
        blocks = []
        in_white = False#表示没有找到白点
        start = 0#最左边的点还是0
        for col in range(w):#从左到右去找点
            pixel = bin_img[row, col]
            if pixel == 255 and not in_white:#如果这个点是白色并且之前标志位in_white非真，就证明是从黑到白的跳变点，白条起始点找到了
                start = col
                in_white = True
            elif pixel == 0 and in_white:#这个点是黑色，标志位是真，证明是从这个点开始，白条断开，最后一个白点的横坐标是目前这个点的col-1
                blocks.append((start, col-1))
                in_white = False

        if in_white:
            blocks.append((start, w-1))#如果标志位一直为真，那么代表右边没有找到黑点了

        return blocks


def draw_points(img, left_points, right_points):
    draw = img.copy()
    for pt in left_points:
        cv2.circle(draw, (pt.x, pt.y), 3, (0, 255, 0), -1)
    for pt in right_points:
        cv2.circle(draw, (pt.x, pt.y), 3, (255, 0, 0), -1)

    return draw



if __name__ == '__main__':

    cap = cv2.VideoCapture( 'res/sample.avi')
    record = cv2.VideoWriter_fourcc(*'XVID')
    out = None
    while cap.isOpened():
        retu, frame = cap.read()
        if not retu:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _,binary = cv2.threshold(gray , 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        tracker = Tracking()
        tracker.process(binary)
        draw = draw_points(frame, tracker.LeftPoints, tracker.RightPoints)
        if out is None:
            h, w = draw .shape[:2]#取前面两个的值，获取高和宽
            out = cv2.VideoWriter('output.avi', record, 30.0, (w, h))#30：每秒显示的帧数

        out.write( draw )
        cv2.imshow('frame', draw)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()