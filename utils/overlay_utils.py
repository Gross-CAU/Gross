import cv2

def overlay_image_alpha(background, overlay, mask, x_min, y_min):
    if overlay.shape[2] == 4:
        b, g, r, a = cv2.split(overlay)
        for i in range(overlay.shape[0]):
            for j in range(overlay.shape[1]):
                if mask[i, j] != 0:
                    if y_min + i < background.shape[0] and x_min + j < background.shape[1]:
                        background[y_min + i, x_min + j] = [b[i, j], g[i, j], r[i, j]]
