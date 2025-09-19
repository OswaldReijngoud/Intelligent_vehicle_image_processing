import cv2
def play_video(video_path):
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return  # 打开失败则退出函数
    while True:
        ret,frame=cap.read()
        if not ret:
            break
        cv2.imshow('Video Player', frame)

        # 5. 等待按键输入，30ms刷新一次，按'q'键退出
        # waitKey返回按键的ASCII码，ord('q')获取'q'的ASCII码
        if cv2.waitKey(30) & 0xFF == ord('q'):
           break
    cap.release()  # 释放视频文件资源
    cv2.destroyAllWindows()  # 关闭所有显示窗口


play_video('sample.avi')


