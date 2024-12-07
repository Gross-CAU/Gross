import os
import glob
import torch
import cv2
import numpy as np
from PIL import Image
import MiDaS.utils_MiDaS as utils_MiDaS

def run_depth_estimation(image_rgb, transform, midas_model, device):
    if isinstance(image_rgb, Image.Image):
        image_rgb = np.array(image_rgb)
    
    transformed_image = transform({"image": image_rgb})["image"]
    transformed_image = torch.from_numpy(transformed_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        depth_prediction = midas_model(transformed_image)
        depth_map = torch.nn.functional.interpolate(
            depth_prediction.unsqueeze(1), size=image_rgb.shape[:2][::-1], mode="bicubic", align_corners=False
        ).squeeze().cpu().float().numpy()
    
    return depth_map


def process(device, model, model_type, image, input_size, target_size, optimize, use_camera, first_execution=False):
    """Run the inference and interpolate."""
    sample = torch.from_numpy(image).to(device).unsqueeze(0)
    
    if optimize and device == torch.device("cuda") and first_execution:
        print("Optimization to half-floats activated.")
        sample = sample.to(memory_format=torch.channels_last).half()

    height, width = sample.shape[2:]
    if not use_camera:
        print(f"Input resized to {width}x{height} before entering the encoder")
    
    prediction = model.forward(sample)
    prediction = (
        torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=target_size[::-1], mode="bicubic", align_corners=False
        ).squeeze().cpu().float().numpy()
    )
    return prediction


def create_side_by_side(image, depth, grayscale):
    """Concatenate the RGB image and depth map side by side."""
    depth_min, depth_max = depth.min(), depth.max()
    normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)

    right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2)
    if not grayscale:
        right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)

    return np.concatenate((image, right_side), axis=1) if image is not None else right_side


def run(input_path, output_path, model, transform, device, model_type, side, optimize, height, square, grayscale):
    """Run MiDaS depth estimation."""
    print("Device:", device)
    print("Start processing")
    
    image_names = glob.glob(os.path.join(input_path, "*"))
    os.makedirs(output_path, exist_ok=True)

    for index, image_name in enumerate(image_names):
        print(f"Processing {image_name} ({index + 1}/{len(image_names)})")
        original_image_rgb = utils_MiDaS.read_image(image_name)
        image = transform({"image": original_image_rgb})["image"]

        with torch.no_grad():
            prediction = process(device, model, model_type, image, (None, None), original_image_rgb.shape[1::-1], optimize, False)

        filename = os.path.join(output_path, f"{os.path.splitext(os.path.basename(image_name))[0]}-{model_type}")
        if not side:
            utils_MiDaS.write_depth(filename, prediction, grayscale, bits=2)
        else:
            content = create_side_by_side(np.flip(original_image_rgb, 2) * 255, prediction, grayscale)
            cv2.imwrite(f"{filename}.png", content)
        utils_MiDaS.write_pfm(f"{filename}.pfm", prediction.astype(np.float32))

    print("Finished")

def run_depth_estimation(image_rgb, transform, midas_model, device):
    """Generates a depth map for a given image."""
    # Convert image to a NumPy array if it is not already
    if isinstance(image_rgb, Image.Image):
        image_rgb = np.array(image_rgb)
    
    # Apply the transformation to get a NumPy array
    transformed_image = transform({"image": image_rgb})["image"]
    
    # Convert to a PyTorch tensor
    transformed_image = torch.from_numpy(transformed_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        depth_prediction = midas_model(transformed_image)
        depth_map = torch.nn.functional.interpolate(
            depth_prediction.unsqueeze(1), size=image_rgb.shape[:2][::-1], mode="bicubic", align_corners=False
        ).squeeze().cpu().float().numpy()
    
    return depth_map

def extract_average_depth_for_mask(mask, depth_map):
    """Calculates the average depth for the coordinates inside a mask."""
    resized_depth_map = cv2.resize(depth_map, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_coords = np.where(mask == 1)
    mask_depth_values = resized_depth_map[mask_coords]
    average_depth = np.mean(mask_depth_values)
    return average_depth

def compute_objects_with_depth(detections, depth_map, ID_TO_OBJECTS, ID_TO_IMAGE_PATH):
    """Computes average depth for each object and returns sorted list of objects with depth."""
    objects_with_depth = []
    for obj_idx, mask in enumerate(detections.mask):
        avg_depth = extract_average_depth_for_mask(mask, depth_map)
        object_name = ID_TO_OBJECTS[detections.class_id[obj_idx]]
        overlay_image_path = ID_TO_IMAGE_PATH[object_name]
        bbox = detections.xyxy[obj_idx]
        
        # Collect object data: (mask, bounding box, average depth, image path)
        objects_with_depth.append((mask, bbox, avg_depth, overlay_image_path))

    # Sort objects by depth (descending order)
    objects_with_depth.sort(key=lambda x: -x[2])  # Sort by depth descending
    return objects_with_depth