import cv2,argparse,time
import numpy as np
from VideoStream import *
from Human_Detection import *
from FeatureExtraction import *
from utils.datasets import letterbox



def main():

    parser = argparse.ArgumentParser(description="Arguments for the script")
    parser.add_argument('--cam_idx', type=int, default=0,
                        help='Camera index(default: 0)')
    parser.add_argument('--img_H', type=int, default=480,
                        help='Input image height (default: 480).')
    parser.add_argument('--img_W', type=int, default=640,
                        help='Input image width (default:640).')

    args = parser.parse_args()


    cam = VideoStream(camera_ID=args.cam_idx,
                      img_width=args.img_W,
                      img_height=args.img_H)
    
    H_detector = Human_Detection()
    fe = SuperPointFrontend(nms_dist=4,conf_thresh=0.015,nn_thresh=0.7)
    
    

    input("Press enter to start configuration")

    #Configuration Loop
    while(cam.camera_isON()):
        frame = cam.next_frame()
        frame = letterbox(frame, 640, stride=64, auto=True)[0]
        
        H_detector.conf_box = [int(frame.shape[1]/2)-120,0,int(frame.shape[1]/2)+120,frame.shape[0]]
        H_detector.display(frame,H_detector.conf_box)

        masked_img = H_detector.configuration(frame)
        if masked_img is not None and masked_img.any():
            gray_masked = cv2.cvtColor(masked_img,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
            pts, desc, heatmap = fe.run(gray_masked)
            for i in range(pts.shape[1]):
                cv2.circle(masked_img, (int(pts[0][i]),int(pts[1][i])), radius=2, color=(0, 0, 255), thickness=-1)
            cv2.imshow("Masked",masked_img)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # cam.camera_OFF()
            break
        

if __name__ == '__main__':
    main()