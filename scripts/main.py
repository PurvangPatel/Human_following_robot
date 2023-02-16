import cv2
import numpy as np
from VideoStream import *
from Human_Detection import *

def main():
    cam = VideoStream(-1)
    H_detector = Human_Detection()

    while(cam.camera_isON):
        image = cam.next_frame()
        H_detector.detect(image)
        pnimg,zipfile = H_detector.detect(image)
        

        person_class_idx = H_detector.model.names.index('person')
        for one_mask, bbox, cls, conf in zipfile:
            if conf < 0.85 and cls !=person_class_idx:
                continue
            color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]               
            pnimg[one_mask] = pnimg[one_mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
            pnimg = cv2.rectangle(pnimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        cv2.imshow("RESULT",pnimg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()