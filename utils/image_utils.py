import cv2
import numpy as np

def img_to_mat(image_file , as_rgb=False):
    out = cv2.imread(image_file).astype(np.float64, copy=False)
    if as_rgb:
        out = out[:,:,::-1]
    return out


def scale_img(mat, image_width, image_height):
    orig_h, orig_w, _ = [float(v) for v in mat.shape]
    resized_img_width = image_width
    resized_img_height = image_height
    x_scale = resized_img_width / orig_w
    y_scale = resized_img_height / orig_h
    scales = (x_scale, y_scale)

    # Resize and reshape mat to get ready to first layer
    scaled_image = cv2.resize(mat, (resized_img_width, resized_img_height))

    return scaled_image, scales, (orig_w, orig_h)
