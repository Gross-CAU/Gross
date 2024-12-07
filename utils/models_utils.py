import torch
import numpy as np
from PIL import Image


def detect_objects_with_grounding_dino(image_path, processor, model, device, text_prompt, box_threshold=0.4, text_threshold=0.3):
    """
    Detect objects in an image using Grounding DINO.
    
    Parameters:
    - image_path: str, path to the image file
    - processor: AutoProcessor, Grounding DINO processor
    - model: AutoModelForZeroShotObjectDetection, Grounding DINO model
    - device: torch.device, computation device (CPU or GPU)
    - text_prompt: str, text prompt for object detection
    - box_threshold: float, confidence threshold for boxes
    - text_threshold: float, confidence threshold for text matching

    Returns:
    - input_boxes: np.ndarray, detected bounding boxes
    - class_names: List[str], names of the detected objects
    """
    image = Image.open(image_path)
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]]
    )
    
    input_boxes = results[0]["boxes"].cpu().numpy()
    class_names = results[0]["labels"]
    
    return input_boxes, class_names, np.array(image)

def generate_masks_with_sam(image_rgb, input_boxes, image_predictor):
    """
    Generate segmentation masks using SAM for detected objects.
    
    Parameters:
    - image_rgb: np.ndarray, RGB image as a NumPy array
    - input_boxes: np.ndarray, bounding boxes for objects
    - image_predictor: SAM2ImagePredictor, SAM 2 model for mask prediction

    Returns:
    - masks: np.ndarray, segmentation masks for the objects
    - class_names: List[str], names of the detected objects
    """
    # input_boxes가 None이거나 비어 있는 경우 처리
    if input_boxes is None or input_boxes.size == 0:  # NumPy 배열의 크기 확인
        print("No input boxes provided for SAM. Skipping mask generation.")
        return None

    
    image_predictor.set_image(image_rgb)
    
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=None,
        multimask_output=False,
    )
    
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    return masks


def register_objects_with_sam2(video_predictor, inference_state, ann_frame_idx, masks, class_names, input_boxes, prompt_type="box"):
    """
    Register objects with SAM 2 using masks, bounding boxes, or point prompts.
    
    Parameters:
    - video_predictor: SAM2VideoPredictor, SAM 2 video predictor model
    - inference_state: dict, video predictor inference state
    - ann_frame_idx: int, annotation frame index
    - masks: np.ndarray, segmentation masks
    - class_names: List[str], names of the detected objects
    - input_boxes: np.ndarray, bounding boxes for the objects
    - prompt_type: str, prompt type ("point", "box", "mask")
    """
    assert prompt_type in ["point", "box", "mask"], "Invalid prompt type for SAM 2"
    
    if prompt_type == "point":
        all_sample_points = sample_points_from_masks(masks=masks, num_points=10)
        for object_id, (label, points) in enumerate(zip(class_names, all_sample_points), start=1):
            labels = np.ones((points.shape[0]), dtype=np.int32)
            video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                points=points,
                labels=labels,
            )
    elif prompt_type == "box":
        for object_id, (label, box) in enumerate(zip(class_names, input_boxes), start=1):
            video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                box=box,
            )
    elif prompt_type == "mask":
        for object_id, (label, mask) in enumerate(zip(class_names, masks), start=1):
            labels = np.ones((1), dtype=np.int32)
            video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                mask=mask
            )
