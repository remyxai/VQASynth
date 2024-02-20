import os
import sys

external_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'external', 'efficientvit'))
if external_path not in sys.path:
    sys.path.append(external_path)

from segment_anything.utils.transforms import ResizeLongestSide
from efficientvit.sam_model_zoo import create_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor

class EfficientVitSam:
    def __init__(self, weight_url="/app/models/10.pt", device="cuda"):
        model_name = weight_url.split("/")[-1].replace(".pt", "")
        efficientvit_sam = create_sam_model(
          name=model_name, weight_url=weight_url,
        )
        self.device = device
        if self.device=="cuda":
            efficientvit_sam = efficientvit_sam.cuda().eval()
        else:
            efficientvit_sam = efficientvit_sam.eval()

        self.efficientvit_sam_predictor = EfficientViTSamPredictor(efficientvit_sam)

    def predict_from_boxes(self, image, boxes, target_length=1024):
        self.efficientvit_sam_predictor.set_image(image)
        resizer = ResizeLongestSide(target_length)
        transformed_boxes = resizer.apply_boxes_torch(boxes, image.shape[:2]).to(self.device)

        masks, _, _ = self.efficientvit_sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(self.device),
            multimask_output = False,
        )
        return masks

