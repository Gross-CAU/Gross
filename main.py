import sys
import os
import torch
import shutil
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils.video_utils import create_video_from_images
from utils.depth_utils import run
from utils.visualization_utils import process_video_segments, generate_image_paths
from utils.video_utils import extract_audio, combine_video_and_audio, save_video_frames
from utils.models_utils import detect_objects_with_grounding_dino, generate_masks_with_sam, register_objects_with_sam2
from config import GeneralConfig, MiDaSConfig, SAM2Config, TrackingConfig
# MiDaS
sys.path.append('./MiDaS/')
from midas.model_loader import load_model

# diffusion 모델 초기화
diffusion_model = "diffusion_model"
# diffusion_model = StableDiffusionInpaintPipeline.from_pretrained(
#     "runwayml/stable-diffusion-inpainting"
# ).to(device)
video_path = sys.argv[1]
filter_type = sys.argv[2]
filter_object = sys.argv[3]

TrackingConfig.VIDEO_PATH = video_path
# Get user input for TEXT_PROMPT
#TEXT_PROMPT = input("Enter the text prompt for object detection (e.g., 'fish . knife . dish'): ")
TEXT_PROMPT = filter_object
if not TEXT_PROMPT.endswith('.'):
    TEXT_PROMPT += '.'
prompt_words = [word.strip() for word in TEXT_PROMPT.split('.') if word.strip()]

# Prompt the user for the version
valid_versions = ['none', 'sticker', 'real']
#version = input(f"Select the version ({', '.join(valid_versions)}): ").strip().lower()
version = filter_type

# 1. Environment Initialization
# ---------------------------------------------------------
if GeneralConfig.first_execution:
    GeneralConfig.first_execution = False

device = MiDaSConfig.device # select device (GPU / CPU)

# Initialize MiDaS model and transform
midas_model, transform, net_w, net_h = load_model(
    device,
    MiDaSConfig.default_model_weights,
    MiDaSConfig.model_type,
    MiDaSConfig.optimize,
    MiDaSConfig.height,
    MiDaSConfig.square,
)

# Initialize SAM2 and Grounding DINO models
video_predictor = build_sam2_video_predictor(SAM2Config.model_cfg, SAM2Config.sam2_checkpoint)
sam2_image_model = build_sam2(SAM2Config.model_cfg, SAM2Config.sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)

processor = AutoProcessor.from_pretrained(TrackingConfig.MODEL_ID)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(TrackingConfig.MODEL_ID).to(device)

# Use bfloat16 for better precision and enable TF32 on Ampere GPUs
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# 2. Frame Extraction
# -------------------------------------------------------------------------------
frame_names = save_video_frames(TrackingConfig.VIDEO_PATH, TrackingConfig.SOURCE_VIDEO_FRAME_DIR)

# initialize video predictor state
inference_state = video_predictor.init_state(video_path=TrackingConfig.SOURCE_VIDEO_FRAME_DIR)
ann_frame_idx = 0  # the frame index we interact with


# 3. Object Detection with Grounding DINO
# ------------------------------------------------------------------------------
img_path = os.path.join(TrackingConfig.SOURCE_VIDEO_FRAME_DIR, frame_names[ann_frame_idx])
image = Image.open(img_path)
inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = grounding_model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

class_names = results[0]["labels"]

# # process the detection results
OBJECTS = class_names

# Step 2: Grounding DINO로 객체 탐지
img_path = os.path.join(TrackingConfig.SOURCE_VIDEO_FRAME_DIR, frame_names[ann_frame_idx])
input_boxes, class_names, image_rgb = detect_objects_with_grounding_dino(
    image_path=img_path,
    processor=processor,
    model=grounding_model,
    device=device,
    text_prompt=TEXT_PROMPT
)
print("Detected boxes:", input_boxes)
print("Detected objects:", class_names)

# 4. Mask Generation with SAM
# -----------------------------------------------------------------------
masks = generate_masks_with_sam(image_rgb=image_rgb, input_boxes=input_boxes, image_predictor=image_predictor)

# 5. Object Registration with SAM2
# --------------------------------------------------------------------
register_objects_with_sam2(
    video_predictor=video_predictor,
    inference_state=inference_state,
    ann_frame_idx=ann_frame_idx,
    masks=masks,
    class_names=class_names,
    input_boxes=input_boxes,
    prompt_type=TrackingConfig.PROMPT_TYPE_FOR_VIDEO
)

# 6. Propagate Predictions
# -------------------------------------------------------------------------
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# 7. Visualization and Depth Processing
# --------------------------------------
# Object-to-image mapping
# None일 경우, diffusion 모델 실행
ID_TO_IMAGE_PATH = generate_image_paths(
    prompt_words=prompt_words,
    base_path='./images',
    version=version
)

# 디렉토리 초기화
if os.path.exists(TrackingConfig.SAVE_TRACKING_RESULTS_DIR):
    shutil.rmtree(TrackingConfig.SAVE_TRACKING_RESULTS_DIR)  # 기존 디렉토리와 내부 파일 삭제
os.makedirs(TrackingConfig.SAVE_TRACKING_RESULTS_DIR)  # 빈 디렉토리 다시 생성

ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}

process_video_segments(
    video_segments=video_segments,
    frame_names=frame_names,
    SOURCE_VIDEO_FRAME_DIR=TrackingConfig.SOURCE_VIDEO_FRAME_DIR,
    SAVE_TRACKING_RESULTS_DIR=TrackingConfig.SAVE_TRACKING_RESULTS_DIR,
    transform=transform,
    midas_model=midas_model,
    device=device,
    ID_TO_OBJECTS=ID_TO_OBJECTS,
    ID_TO_IMAGE_PATH=ID_TO_IMAGE_PATH,
    scale_factor=1.5,
    version=version,  # 사용자 선택 (real, sticker, none)
    diffusion_model=diffusion_model if version == "none" else None
)

debug_image = Image.fromarray(image_rgb)
debug_image.save("debug_video_segments_output.jpg")


# 8. Final Video Generation
# --------------------------
create_video_from_images(TrackingConfig.SAVE_TRACKING_RESULTS_DIR, TrackingConfig.OUTPUT_VIDEO_PATH)
extract_audio(TrackingConfig.VIDEO_PATH, TrackingConfig.AUDIO_DIR)
combine_video_and_audio(TrackingConfig.OUTPUT_VIDEO_PATH, TrackingConfig.AUDIO_DIR, TrackingConfig.FINAL_VIDEO_PATH)