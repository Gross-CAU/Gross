import cv2
import os
from tqdm import tqdm
from moviepy.editor import VideoFileClip, AudioFileClip
import supervision as sv
from pathlib import Path


def extract_audio(video_path, output_audio_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    audio_clip.write_audiofile(output_audio_path)

def save_video_frames(video_path, output_dir, stride=1):
    """
    Save video frames to a directory.

    Parameters:
    - video_path: str, path to the input video
    - output_dir: str, directory to save the frames
    - stride: int, frame extraction stride

    Returns:
    - frame_names: List[str], sorted list of frame file names
    """
    video_info = sv.VideoInfo.from_video_path(video_path)
    print(video_info)

    source_frames = Path(output_dir)
    source_frames.mkdir(parents=True, exist_ok=True)

    frame_generator = sv.get_video_frames_generator(video_path, stride=stride, start=0, end=None)
    with sv.ImageSink(
        target_dir_path=source_frames, 
        overwrite=True, 
        image_name_pattern="{:05d}.jpg"
    ) as sink:
        for frame in tqdm(frame_generator, desc="Saving Video Frames"):
            sink.save_image(frame)

    frame_names = sorted(
        [
            p for p in os.listdir(output_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ],
        key=lambda p: int(os.path.splitext(p)[0])
    )
    return frame_names


def combine_video_and_audio(video_path, audio_path, output_video_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    final_clip = video_clip.set_audio(audio_clip)
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    final_clip.write_videofile(output_video_path, codec="libx264")

def create_video_from_images(image_folder, output_video_path, frame_rate=25):
    # define valid extension
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    
    # get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) 
                   if os.path.splitext(f)[1] in valid_extensions]
    image_files.sort()  # sort the files in alphabetical order
    print(image_files)
    if not image_files:
        raise ValueError("No valid image files found in the specified folder.")
    
    # load the first image to get the dimensions of the video
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape
    
    # create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # codec for saving the video
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    
    # write each image to the video
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)
    
    # source release
    video_writer.release()
    print(f"Video saved at {output_video_path}")

# 비디오에서 오디오 추출 (moviepy 사용)
def extract_audio(video_path, output_audio_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_audio_path)

# 처리한 비디오와 추출한 오디오 결합 (moviepy 사용)
def combine_video_and_audio(video_path, audio_path, output_video_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(output_video_path, codec="libx264")
