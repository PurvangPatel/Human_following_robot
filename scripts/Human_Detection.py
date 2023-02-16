import cv2
import torch
import yaml
import numpy as np
from torchvision import transforms

from utils.datasets import letterbox
from utils.general import non_max_suppression_mask_conf

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image


class Human_Detection:
    def __init__(self) -> None:
        """Class to handle Human detection and segmentation
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            weigths = torch.load('data/yolov7-mask.pt')
            with open('data/hyp.scratch.mask.yaml','r') as f:
                self.hyp = yaml.load(f, Loader=yaml.FullLoader)   
        except (FileNotFoundError, OSError) as e:
            print(f"Could not load file: {e}")
        else:
            self.model = weigths['model']
            self.model = self.model.half().to(self.device)
            _ = self.model.eval()

    def Preprocessing(self,image):
        """Converts the RGB images to tensor of shape [1, 3, 640, 448]

        Args:
            image (numpy.ndarray): RGB image

        Returns:
            torch.Tensor: tensor of shape [1, 3, 640, 448]
        """
        image = letterbox(image, 640, stride=64, auto=True)[0]
        image_tensor = transforms.ToTensor()(image)
        image_tensor = torch.tensor(np.array([image_tensor.numpy()]))
        image_tensor = image_tensor.to(self.device)
        image_tensor = image_tensor.half()
        return image_tensor
    
    def Postprocessing(self,image_tensor,model_output):
        """_summary_

        Args:
            image_tensor (_type_): _description_
            model_output (_type_): _description_

        Returns:
            _type_: _description_
        """
        inf_out, attn, bases, sem_output = model_output['test'], model_output['attn'], model_output['bases'], model_output['sem']
        bases = torch.cat([bases, sem_output], dim=1)
        _, _, height, width = image_tensor.shape
        pooler_scale = self.model.pooler_scale
        pooler = ROIPooler(output_size=self.hyp['mask_resolution'], scales=(pooler_scale,), sampling_ratio=1, pooler_type='ROIAlignV2', canonical_level=2)
        output, output_mask, _, _, _ = non_max_suppression_mask_conf(inf_out, attn, bases, pooler, self.hyp, conf_thres=0.25, iou_thres=0.65, merge=False, mask_iou=None)
        pred, pred_masks = output[0], output_mask[0]

        bboxes = Boxes(pred[:, :4])
        original_pred_masks = pred_masks.view(-1, self.hyp['mask_resolution'], self.hyp['mask_resolution'])
        pred_masks = retry_if_cuda_oom(paste_masks_in_image)( original_pred_masks, bboxes, (height, width), threshold=0.5)
        pred_masks_np = pred_masks.detach().cpu().numpy()
        pred_cls = pred[:, 5].detach().cpu().numpy()
        pred_conf = pred[:, 4].detach().cpu().numpy()
        nimg = image_tensor[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nbboxes = bboxes.tensor.detach().cpu().numpy().astype(np.int64)
        pnimg = nimg.copy()

        return pnimg, zip(pred_masks_np, nbboxes, pred_cls, pred_conf)

    def detect(self,image):
        self.image = image
        image_tensor = self.Preprocessing(image)
        model_output = self.model(image_tensor)
        img,zipfile = self.Postprocessing(image_tensor,model_output)
        return img,zipfile


def main():
    image = cv2.imread('data/person2.jpg')
    H_Detect = Human_Detection() 
    img,zipfile = H_Detect.detect(image)

    pnimg = img.copy()
    person_class_idx = H_Detect.model.names.index('person')
    for one_mask, bbox, cls, conf in zipfile:
    
        if conf < 0.85 and cls !=person_class_idx:
            continue
        
        color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]               
        pnimg[one_mask] = pnimg[one_mask] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
        pnimg = cv2.rectangle(pnimg, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    cv2.imshow("RESULT",pnimg)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()