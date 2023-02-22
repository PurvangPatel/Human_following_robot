import cv2,argparse
import numpy as np
from VideoStream import *
from Human_Detection import *

def main():

    parser = argparse.ArgumentParser(description="Arguments for the script")
    parser.add_argument('--cam_idx', type=int, default=0,help='Camera index')
    parser.add_argument('--display_prediction', action='store_true', default=False,help='Display Human detection and segmentation')
    args = parser.parse_args()


    cam = VideoStream(args.cam_idx)
    H_detector = Human_Detection()

    
    while(cam.camera_isON()):
        frame = cam.next_frame()
        H_detector.detect(frame)
        imgaess = H_detector.mask_bg()
        print(len(imgaess))
        if(len(imgaess)!=0):
            cv2.imshow("CROPPED",imgaess[0])
        
        
        if (args.display_prediction):
            H_detector.display_detection()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cam.camera_OFF()
            break

if __name__ == '__main__':
    main()