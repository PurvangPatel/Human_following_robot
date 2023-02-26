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
    
    frame = letterbox(cam.next_frame(), 640, stride=64, auto=True)[0] 
    H_detector.conf_box = [int(frame.shape[1]/2)-120,0,
                           int(frame.shape[1]/2)+120,frame.shape[0]]

    input("Press enter to start configuration")

    #Configuration Loop
    frame_to_skip = 10
    j=0
    while(cam.camera_isON()):
        j+=1
        if(j!=frame_to_skip):
            continue
        j=0
        frame = cam.next_frame()
        frame = letterbox(frame, 640, stride=64, auto=True)[0]
        
        frame_copy = frame.copy()
        H_detector.draw(frame_copy,H_detector.conf_box)
        cv2.imshow("Configuration",frame_copy)
        
        H_detector.detect(frame)
        best_box,best_mask = H_detector.configuration()
        if(best_box is None or best_mask is None):
            continue
        
        masked_img = H_detector.mask_bg(frame,best_box,best_mask)
        gray_masked = cv2.cvtColor(masked_img,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        pts, desc, _ = fe.run(gray_masked) 
        
        indices = np.where(pts[2,:] >= 0.2)
        indices = indices[0]
        desc = desc[:,indices]

        H_detector.features =  np.concatenate((H_detector.features, desc), axis=1)
            
        for i in range(pts.shape[1]):
            cv2.circle(masked_img, (int(pts[0][i]),int(pts[1][i])), radius=2, color=(0, 0, 255), thickness=-1)
        cv2.imshow("Masked",masked_img)       

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # cam.camera_OFF()
            cv2.destroyAllWindows()
            break

    print("Recorded {} features of the person".format(H_detector.features.shape[1]))
    j=0
    while(cam.camera_isON()):
        # tic = time.time()
        j+=1
        if(j!=frame_to_skip):
            continue
        j=0
        frame = cam.next_frame()
        frame = letterbox(frame, 640, stride=64, auto=True)[0]
        
        H_detector.detect(frame)
        if(H_detector.isdetected()):
            bestmatches = 0
            best_box,best_mask = None, None
            for one_mask, bbox, cls, conf in H_detector.zipfile:
                if conf < H_detector.pred_conf or cls !=H_detector.person_class_idx:
                    continue
                masked_img = H_detector.mask_bg(frame,bbox,one_mask)
                gray_masked = cv2.cvtColor(masked_img,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
                pts, desc, _ = fe.run(gray_masked)
                if(desc is None or not desc.any()):
                    continue   
                indices = np.where(pts[2,:] >= 0.2)
                indices = indices[0]
                desc = desc[:,indices]
                matches = fe.nn_match_two_way(H_detector.features, desc,0.7)
                H_detector.draw(frame,bbox,one_mask,color=(255,0,0))  
                if(matches.shape[1]>bestmatches):
                    bestmatches = matches.shape[1]
                    best_box,best_mask = bbox,one_mask
            if(bestmatches>50):
                H_detector.draw(frame,best_box,best_mask,color=(0,255,0))   
            cv2.imshow("tracking",frame)   
        # timer = tic - time.time()
        # print("Timer:{}",format(timer))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cam.camera_OFF()
            cv2.destroyAllWindows()
            break    
    
        

if __name__ == '__main__':
    main()