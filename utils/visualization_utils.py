import os
import cv2
import numpy as np
import supervision as sv
from pycocotools import mask as mask_util
from PIL import Image, ImageEnhance
from utils.depth_utils import run_depth_estimation, compute_objects_with_depth

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def calculate_principal_axis(mask):
    """마스크 데이터를 기반으로 주성분 축(Principal Axis)을 계산하여 각도를 반환"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour_points = np.vstack(contours).squeeze()
        if len(contour_points.shape) == 2 and contour_points.shape[0] > 1:
            mean = np.mean(contour_points, axis=0)
            centered_points = contour_points - mean
            cov_matrix = np.cov(centered_points.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            max_eigenvalue_index = np.argmax(eigenvalues)
            principal_axis = eigenvectors[:, max_eigenvalue_index]
            angle = np.arctan2(principal_axis[1], principal_axis[0])
            angle_degrees = np.degrees(angle)
            return angle_degrees, mean  # 각도와 중심 반환
    return None, None

def overlay_image_alpha(background, overlay, mask, x_min, y_min):
    """알파 채널을 기반으로 이미지를 배경에 덧씌움"""
    if overlay.shape[2] == 4:  # 알파 채널이 있는 경우
        b, g, r, a = cv2.split(overlay)
        for i in range(overlay.shape[0]):
            for j in range(overlay.shape[1]):
                if mask[i, j] != 0:  # 마스크에서 투명하지 않은 부분만 덧씌움
                    if y_min + i < background.shape[0] and x_min + j < background.shape[1]:
                        background[y_min + i, x_min + j] = [b[i, j], g[i, j], r[i, j]]

def adjust_image_brightness_and_saturation(character_image, mask_brightness, mask_saturation,
                                           min_brightness_adjustment=0.8, max_brightness_adjustment=1.2,
                                           min_saturation_adjustment=0.8, max_saturation_adjustment=1.2): # 추가
    """
    character_image의 밝기와 채도를 mask_brightness와 mask_saturation에 맞게 조정
    """
    # 이미지를 PIL로 변환
    character_image_pil = Image.fromarray(cv2.cvtColor(character_image, cv2.COLOR_BGRA2RGBA))

    # 밝기 조정 비율 제한
    brightness_adjustment_factor = np.clip(mask_brightness, min_brightness_adjustment, max_brightness_adjustment)

    # 채도 조정 비율 제한
    saturation_adjustment_factor = np.clip(mask_saturation, min_saturation_adjustment, max_saturation_adjustment)

    # 밝기 보정
    brightness_enhancer = ImageEnhance.Brightness(character_image_pil)
    adjusted_image = brightness_enhancer.enhance(brightness_adjustment_factor)
    
    # 채도 보정
    saturation_enhancer = ImageEnhance.Color(adjusted_image)
    adjusted_image = saturation_enhancer.enhance(saturation_adjustment_factor)
    
    # 다시 OpenCV 형식으로 변환
    adjusted_image_cv = cv2.cvtColor(np.array(adjusted_image), cv2.COLOR_RGBA2BGRA)
    
    return adjusted_image_cv


def calculate_mask_brightness_and_saturation(mask, image): # 추가
    """
    마스크 내부의 명암과 채도 평균을 계산
    """
    # 마스크 범위 내의 이미지 픽셀들만 선택 (마스크가 1인 부분에 해당하는 원본 이미지 영역)
    mask_area = np.where(mask == 1)
    masked_pixels = image[mask_area]
    
    # 밝기(Lightness) 계산: RGB 값의 평균을 기준으로 계산 (0~255 범위에서)
    brightness = np.mean(masked_pixels) / 255.0  # 밝기 값은 0~1 사이로 정규화
    
    # 채도(Saturation) 계산: HSV 색 공간으로 변환 후 채널에서 S값을 추출
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masked_hsv_pixels = image_hsv[mask_area]
    saturation = np.mean(masked_hsv_pixels[:, 1]) / 255.0  # 채도 값도 0~1 사이로 정규화
    
    return brightness, saturation

def process_contours_and_overlay(ori_mask, overlay_image_path, annotated_frame, detections, frame_idx, SAVE_TRACKING_RESULTS_DIR, scale_factor, img):
    """
    주어진 윤곽선을 기반으로 PCA를 수행하여 주축을 찾고, 해당 각도로 캐릭터 이미지를 회전 및 배율 확대 후
    배경 이미지 위에 덧씌운다.

    Parameters:
    ori_mask: 마스크 데이터
    overlay_image_path: 회전시킬 캐릭터 이미지 경로
    annotated_frame: 배경 이미지 (프레임)
    detections: 바운딩 박스 좌표 (x_min, y_min, x_max, y_max)
    frame_idx: 현재 프레임 번호
    SAVE_TRACKING_RESULTS_DIR: 결과 이미지 저장 경로
    scale_factor: 이미지를 확대할 배율
    """
    
    # print(f"Contours found: {len(contours)}")  # contours의 개수 확인
    # if len(contours) == 0:
    #     print("Contours data is empty.")

    # 객체 바운딩 박스 좌표    
    x_min, y_min, x_max, y_max = map(int, detections)
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    rle = single_mask_to_rle(ori_mask)
    mask = mask_util.decode(rle)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 모든 윤곽선을 하나로 합치기
        contour_points = np.vstack(contours).squeeze()

        if len(contour_points.shape) == 2 and contour_points.shape[0] > 1:
            # NumPy를 사용한 PCA 구현 -> 주축 계산
            mean = np.mean(contour_points, axis=0)
            centered_points = contour_points - mean
            cov_matrix = np.cov(centered_points.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

            # 가장 큰 고유값에 대응하는 고유벡터가 주축이 됩니다.
            max_eigenvalue_index = np.argmax(eigenvalues)
            principal_axis = eigenvectors[:, max_eigenvalue_index]

            # 방향 추정
            angle = np.arctan2(principal_axis[1], principal_axis[0])
            angle_degrees = np.degrees(angle)

            # 캐릭터 이미지 읽기
            character_image = cv2.imread(overlay_image_path, cv2.IMREAD_UNCHANGED)

            # 마스크 명암과 채도 계산
            mask_brightness, mask_saturation = calculate_mask_brightness_and_saturation(mask, img) # 추가

            # 캐릭터 이미지의 명암 및 채도 조정
            adjusted_overlay = adjust_image_brightness_and_saturation(character_image, mask_brightness, mask_saturation) # 추가
            
            # 이미지의 중심 좌표 계산
            (h, w) = adjusted_overlay.shape[:2] # 수정
            center = (w // 2, h // 2)

            # 원본 칼 이미지의 각도와 동일한 각도로 캐릭터 이미지를 회전
            rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

            # 이미지 회전
            rotated_image = cv2.warpAffine(adjusted_overlay, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)) # 수정

            # 바운딩 박스 크기에 맞춰 이미지 크기 조정
            overlay_resized = cv2.resize(rotated_image, (bbox_width, bbox_height), interpolation=cv2.INTER_LINEAR)

            # 배율에 맞게 이미지 확대
            new_width = int(overlay_resized.shape[1] * scale_factor)
            new_height = int(overlay_resized.shape[0] * scale_factor)
            overlay_resized_scaled = cv2.resize(overlay_resized, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            # 확대된 이미지의 새로운 크기 계산
            new_h, new_w = overlay_resized_scaled.shape[:2]

            # 새롭게 확대된 이미지의 좌표를 원래 좌표에서 보정 (중심을 유지하기 위해)
            new_x_min = x_min - (new_w - bbox_width) // 2
            new_y_min = y_min - (new_h - bbox_height) // 2

            # 알파 채널을 분리하여 투명 배경 처리
            if overlay_resized_scaled.shape[2] == 4:
                b, g, r, a = cv2.split(overlay_resized_scaled)
                for i in range(overlay_resized_scaled.shape[0]):
                    for j in range(overlay_resized_scaled.shape[1]):
                        if new_y_min + i < annotated_frame.shape[0] and new_x_min + j < annotated_frame.shape[1]:
                            if a[i, j] != 0:  # 알파 채널이 0이 아닌 경우에만 덧씌움
                                annotated_frame[new_y_min + i, new_x_min + j] = [b[i, j], g[i, j], r[i, j]]
            else:
                # 알파 채널이 없으면 그냥 덧씌우기
                for i in range(overlay_resized_scaled.shape[0]):
                    for j in range(overlay_resized_scaled.shape[1]):
                        if new_y_min + i < annotated_frame.shape[0] and new_x_min + j < annotated_frame.shape[1]:
                            if overlay_resized_scaled[i, j].any():  # 검은색 배경은 제외
                                annotated_frame[new_y_min + i, new_x_min + j] = overlay_resized_scaled[i, j][:3]

            # 결과 이미지 저장
            cv2.imwrite(os.path.join(SAVE_TRACKING_RESULTS_DIR, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)
        else:
            print("주성분 분석을 수행하기에 충분한 데이터가 없습니다.")
    else:
        print("윤곽을 찾을 수 없습니다.")

def adjust_image_brightness(character_image, mask_brightness):
    """
    character_image의 밝기를 mask_brightness로 조정
    """
    # 이미지를 PIL로 변환
    character_image_pil = Image.fromarray(cv2.cvtColor(character_image, cv2.COLOR_BGRA2RGBA))
    
    # 명암 보정
    enhancer = ImageEnhance.Brightness(character_image_pil)
    adjusted_image = enhancer.enhance(mask_brightness)
    
    # 다시 OpenCV 형식으로 변환
    adjusted_image_cv = cv2.cvtColor(np.array(adjusted_image), cv2.COLOR_RGBA2BGRA)
    
    return adjusted_image_cv

def calculate_mask_brightness(mask, image):
    """
    마스크 내부의 명암 평균을 계산
    """
    # 마스크 범위 내의 이미지 픽셀들만 선택
    mask_area = np.where(mask == 1)
    masked_pixels = image[mask_area]
    
    # 밝기(Lightness)는 RGB 값의 평균을 기준으로 계산
    brightness = np.mean(masked_pixels) / 255.0  # 밝기 값은 0에서 1 사이
    
    return brightness

def annotate_frame(img, detections, ID_TO_OBJECTS, version):
    """Annotates a frame with bounding boxes, labels, and masks."""
    annotated_frame = img.copy()

    # 객체 제거 단계: none 버전 처리
    '''if version == "none":
        if hasattr(detections, 'xyxy'):  # bounding box 정보가 detections에 포함되어 있을 경우
            for box in detections.xyxy:  # detections.xyxy는 bounding box 좌표 리스트
                x_min, y_min, x_max, y_max = map(int, box)  # 좌표를 정수로 변환
                annotated_frame[y_min:y_max, x_min:x_max] = [0, 0, 0]  # bounding box 영역을 검정색으로 채움'''
    
    # 마스크를 검정색으로 처리
    if hasattr(detections, 'mask'):  # mask 정보가 detections에 포함되어 있을 경우
        for mask in detections.mask:
            annotated_frame[mask] = [0, 0, 0]  # RGB 값 (0, 0, 0) = 검정색
    
    return annotated_frame



def overlay_objects_by_depth(objects_with_depth, annotated_frame, frame_idx, SAVE_TRACKING_RESULTS_DIR, scale_factor):
    """Overlays objects on the frame in the order of their depth."""
    for mask, bbox, _, overlay_image_path in objects_with_depth:
        process_contours_and_overlay(
            ori_mask=mask,
            overlay_image_path=overlay_image_path,
            annotated_frame=annotated_frame,
            detections=bbox,  # Bounding box coordinates
            frame_idx=frame_idx,
            SAVE_TRACKING_RESULTS_DIR=SAVE_TRACKING_RESULTS_DIR,
            scale_factor=scale_factor,
            img=annotated_frame
        )


def process_video_segments(
    video_segments, 
    frame_names, 
    SOURCE_VIDEO_FRAME_DIR, 
    SAVE_TRACKING_RESULTS_DIR, 
    transform, 
    midas_model, 
    device, 
    ID_TO_OBJECTS, 
    ID_TO_IMAGE_PATH, 
    scale_factor=1.5,
    version='real',
    diffusion_model=None
):
    """
    Processes video segments by annotating frames, generating depth maps, and overlaying objects.
    """
    if not os.path.exists(SAVE_TRACKING_RESULTS_DIR):
        os.makedirs(SAVE_TRACKING_RESULTS_DIR)

    for frame_idx, segments in video_segments.items():
        img_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[frame_idx])
        img = cv2.imread(img_path)
        
        # 객체가 없는 프레임 처리
        if not segments:  # segments가 비어 있으면 스킵
            print(f"Frame {frame_idx}: No objects detected, skipping.")
            continue

        object_ids = list(segments.keys())
        masks = list(segments.values())
        masks = np.concatenate(masks, axis=0)

        # Create detections
        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
            mask=masks,  # (n, h, w)
            class_id=np.array(object_ids, dtype=np.int32),
        )

        # 객체 제거 단계: none 버전 처리
        if version == "none":
            #for mask in detections.mask:
            #    img[mask == 1] = (0, 0, 0)
            if detections is not None:  # input_boxes가 주어졌을 경우
                for box in detections.xyxy:
                    x_min, y_min, x_max, y_max = map(int, box)  # 좌표를 정수로 변환
                    img[y_min:y_max, x_min:x_max] = (0, 0, 0)  # bounding box 영역을 검정색으로 채움

            save_path = os.path.join(SAVE_TRACKING_RESULTS_DIR, f"masked_frame_{frame_idx:05d}.jpg")
            cv2.imwrite(save_path, img)
            continue
        
        # Annotate frame with bounding boxes, labels, and masks
        annotated_frame = annotate_frame(img, detections, ID_TO_OBJECTS, version)

        # Generate depth map
        depth_map = run_depth_estimation(img, transform, midas_model, device)

        # Compute average depth for each object
        objects_with_depth = compute_objects_with_depth(detections, depth_map, ID_TO_OBJECTS, ID_TO_IMAGE_PATH)

        # # clown과 knife의 depth 출력
        # for mask, bbox, avg_depth, overlay_image_path in objects_with_depth:
        #     object_name = overlay_image_path.split('/')[-1].split('_')[0]  # 또는 다른 방식으로 이름 확인
        #     if object_name in ["clown", "knife"]:
        #         print(f"Object: {object_name}, Depth: {avg_depth:.2f}")
        
        # Overlay objects in order of depth
        overlay_objects_by_depth(
            objects_with_depth, 
            annotated_frame, 
            frame_idx, 
            SAVE_TRACKING_RESULTS_DIR, 
            scale_factor
        )

def generate_image_paths(prompt_words, base_path='../images', version='real'):
    if version not in ["real", "sticker"]:
        if version == "none":
            return '../images/none/none.png'
        raise ValueError("Version must be 'real' or 'sticker'.")

    image_paths = {}
    fallback_image = os.path.join(base_path, version, f"mascot_{version}.png")  # Use mascot specific to version

    for word in prompt_words:
        # Construct the specific image path
        image_path = os.path.join(base_path, version, f"{word}_{version}.png")
        
        # Check if the file exists; use mascot if it doesn't
        if not os.path.exists(image_path):
            print(f"Image for '{word}' not found. Using fallback image: mascot_{version}.png")
            image_path = fallback_image
        
        # Assign the resolved path to the dictionary
        image_paths[word] = image_path
        
    return image_paths

def apply_diffusion_inpainting(diffusion_model, image, mask, bbox):
    # """
    # Removes objects in the image using a Diffusion-based inpainting model.
    
    # Parameters:
    # - diffusion_model: StableDiffusionInpaintPipeline
    # - image: numpy.ndarray - Original image (H, W, C)
    # - mask: numpy.ndarray - Binary mask of the object to remove (H, W)
    # - bbox: list - Bounding box coordinates [x_min, y_min, x_max, y_max]
    
    # Returns:
    # - numpy.ndarray - Image with the object removed.
    # """
    # # Convert image and mask to PIL format
    # image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # mask_pil = Image.fromarray((mask * 255).astype(np.uint8))  # Mask as binary image (0 or 255)

    # # Crop image and mask to the bounding box
    # x_min, y_min, x_max, y_max = map(int, bbox)
    # cropped_image = image_pil.crop((x_min, y_min, x_max, y_max))
    # cropped_mask = mask_pil.crop((x_min, y_min, x_max, y_max))

    # # Inpaint the cropped region
    # inpainted_cropped_image = diffusion_model(
    #     prompt="",
    #     image=cropped_image,
    #     mask_image=cropped_mask
    # ).images[0]

    # # Replace the inpainted region in the original image
    # inpainted_image = image_pil.copy()
    # inpainted_image.paste(inpainted_cropped_image, (x_min, y_min, x_max, y_max))

    # # Convert back to numpy array
    # return cv2.cvtColor(np.array(inpainted_image), cv2.COLOR_RGB2BGR)
    return 1