import torch

class GeneralConfig:
    first_execution = True

# MiDaS 관련 설정
class MiDaSConfig:
    default_model_weights = "./MiDaS/weights/dpt_beit_large_512.pt"
    default_input_path = "./MiDaS/input"
    default_output_path = ""
    model_type = "dpt_beit_large_512"
    side_by_side = False
    optimize = False
    height = None
    square = False
    grayscale = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SAM2 관련 설정
class SAM2Config:
    sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

# Grounding 및 Tracking 관련 설정
class TrackingConfig:
    MODEL_ID = "IDEA-Research/grounding-dino-tiny"
    # VIDEO_PATH = "./assets/fish.mp4"
    # VIDEO_PATH = "./assets/clown.mp4"
    VIDEO_PATH = "./assets/cup_knife.mp4"
    # VIDEO_PATH = "./assets/check_knife.mp4"
    # VIDEO_PATH = "./assets/chicken_knife.mp4"
    # VIDEO_PATH = "./assets/hammer.mp4"
    # VIDEO_PATH = "./assets/baek_midterm.mp4"
    # TEXT_PROMPT = "fish . knife . "
    OUTPUT_VIDEO_PATH = "./outputs/audio/fish_blackbox.mp4"
    SOURCE_VIDEO_FRAME_DIR = "./custom_video_frames_cyj"
    SAVE_TRACKING_RESULTS_DIR = "./tracking_results_black"
    AUDIO_DIR = "./outputs/audio/audio_fish.mp3"
    FINAL_VIDEO_PATH = "./outputs/audio/fish_blackbox_audio.mp4"
    PROMPT_TYPE_FOR_VIDEO = "box"  # choose from ["point", "box", "mask"]
