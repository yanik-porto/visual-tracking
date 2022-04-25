import numpy as np
import cv2
import argparse
from tracker import Tracker

def parse():
    parser = argparse.ArgumentParser(description='particle filter tracking')
    parser.add_argument('--video', required=True, type=str, help='video where performing tracking')
    parser.add_argument('--rectangle', required=False, type=str, default=[133, 28, 44, 36], help='video where performing tracking')
    return parser.parse_args()

if __name__ == "__main__":
    print("particle filter tracking")
    
    args = parse()

    cap = cv2.VideoCapture(args.video)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    tracker = None

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if tracker is None:
                tracker = Tracker(frame, args.rectangle)
            else:
                tracker.track(frame)

        else:
            break

    cap.release()
    cv2.destroyAllWindows()