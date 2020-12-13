import os
import cv2

def save_all_frames(video_path, dir_path, basename, ext='jpg'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
            frame2 = cv2.resize(frame ,(240, 240))
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame2)
            n += 1
        else:
            return

save_all_frames('acktr-Pendulum-v0-step-0-to-step-3000.mp4', 'pendulum_pic', 'sample_video_img')
