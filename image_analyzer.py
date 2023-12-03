import numpy as np
import cv2
import os
import pandas as pd
from itertools import compress
from sklearn.neighbors import NearestNeighbors


def load_groundtruth(directory: str, images):
    image_positions = pd.read_csv("{}/labels_people_detection.csv".format(directory), delimiter=",", usecols=["Image", "X", "Y"])

    image_count = pd.DataFrame(columns=["Image", "NPeople"])

    for img in images:
        if img not in np.unique(image_positions["Image"]):
            image_count = image_count._append({"Image": img, "NPeople": 0}, ignore_index=True)

    for img in np.unique(image_positions["Image"]):
        n_people = np.sum(image_positions["Image"] == img)
        image_count = image_count._append({"Image": img, "NPeople": n_people}, ignore_index=True)

    return image_positions, image_count


def load_data(directory: str):
    image_filenames = os.listdir(directory)
    image_filenames.sort()
    image_filenames.pop()

    image_positions, image_count = load_groundtruth(directory, image_filenames)

    for i in range(len(image_filenames)):
        image_filenames[i] = "{}/{}".format(directory, image_filenames[i])

    #image_stack = np.stack([cv2.imread(filename, cv2.IMREAD_GRAYSCALE) for filename in image_filenames], axis=2)
    #image_stack = np.transpose(image_stack, (2, 0, 1))

    image_stack_color = np.stack(
        [cv2.cvtColor(cv2.imread(filename, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for filename in image_filenames],
        axis=2)
    image_stack_color = np.transpose(image_stack_color, (2, 3, 0, 1))

    return image_stack_color, image_positions, image_count


# original_img = image_stack[idx]
def compute_differences(original_img, empty_img):
    clahe = cv2.createCLAHE(clipLimit=1)

    empty_img_ = empty_img.copy()
    empty_img_ = cv2.GaussianBlur(empty_img_, (3, 3), 0) # Slight improvement of accuracy
    empty_img_[0:440, :] = 0

    image_mod_ = original_img.copy()
    image_mod_ = cv2.GaussianBlur(image_mod_, (3, 3), 0)
    image_mod_[0:440, :] = 0

    empty_image_ = clahe.apply(empty_img_)
    test_image_ = clahe.apply(image_mod_)

    diff = cv2.subtract(empty_image_, test_image_)

    kernelSize = (10, 18)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    tophat = cv2.morphologyEx(diff, cv2.MORPH_TOPHAT, kernel)

    return tophat


def skin_based_mask(image, tophat):
    edges = cv2.Canny(image=tophat, threshold1=100, threshold2=200)

    dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 10)))

    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 10)))

    image_used = np.copy(image)

    image_used[0][closed == 0] = 0
    image_used[1][closed == 0] = 0
    image_used[2][closed == 0] = 0

    image_used = np.transpose(image_used, (1, 2, 0))

    mask = skin_color_filter(image_used, cv2.COLOR_RGB2YCR_CB)

    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)))

    # mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3,10)))

    return mask[440:, :]


def skin_color_filter(image, palette):

    if palette == cv2.COLOR_RGB2YCR_CB:
        min_YCrCb = np.array([0, 133, 77], np.uint8)
        max_YCrCb = np.array([235, 173, 127], np.uint8)

        imageYCrCb = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
        skinRegion = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)
    elif palette == cv2.COLOR_RGB2HSV:
        min_hsv = np.array([0, 51, 102], np.uint8)
        max_hsv = np.array([25 // 2, 153, 255], np.uint8)

        imageHSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        skinRegion = cv2.inRange(imageHSV, min_hsv, max_hsv)

        # 2 steps
        min_hsv = np.array([335 // 2, 51, 102], np.uint8)
        max_hsv = np.array([360 // 2, 153, 255], np.uint8)
        skinRegion = cv2.bitwise_or(skinRegion, cv2.inRange(imageHSV, min_hsv, max_hsv))
    else:
        min_rgb = np.array([96, 0, 41], np.uint8)
        max_rgb = np.array([255, 39, 255], np.uint8)

        imageRGB = np.copy(image)
        skinRegion = cv2.inRange(imageRGB, min_rgb, max_rgb)
        skinRegion = 255*cv2.bitwise_and(skinRegion, np.array((np.max(imageRGB, axis=2) - np.min(imageRGB, axis=2)) > 15, dtype="uint8"))
        skinRegion = 255*cv2.bitwise_and(skinRegion, np.array(np.abs(imageRGB[:, :, 0] - imageRGB[:, :, 1]) > 15, dtype="uint8"))
        skinRegion = 255*cv2.bitwise_and(skinRegion, np.array((imageRGB[:, :, 0] > imageRGB[:, :, 1]), dtype="uint8"))
        skinRegion = 255*cv2.bitwise_and(skinRegion, np.array((imageRGB[:, :, 0] > imageRGB[:, :, 2]), dtype="uint8"))

    return skinRegion


# ------------------------------------------------------------
# ------------------------------------------------------------

def min_max_norm(image):
    min_val = np.min(image)
    max_val = np.max(image)

    norm_image = (image - min_val) / (max_val - min_val)
    return norm_image


def divide_img_sectors(image):
    # Find boundaries

    img_size = image.shape
    sqrt_arr = np.sqrt(np.linspace(1, img_size[0], 8, dtype=np.int_))
    norm_arr = min_max_norm(sqrt_arr)
    cut_boundaries = img_size[0] - (norm_arr * img_size[0]).astype(int)

    tmp = np.repeat(cut_boundaries, 2)[1:-1]
    cut_boundaries_paris = tmp[:(len(tmp) // 2 + 1) * 2].reshape(-1, 2)
    image_list_sectors = []

    for pair_i in cut_boundaries_paris:
        img = image[pair_i[1]:pair_i[0], :].copy()
        image_list_sectors.append(img)

    return image_list_sectors


def filter_blobs(bin_image, thresh_area):
    # Filter blobs by area
    contours, _ = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_cont = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= thresh_area:
            filtered_cont.append(contour)

    # Fill the new image with the filtered blobs
    canvas = np.zeros(bin_image.shape)
    cv2.drawContours(canvas, filtered_cont, -1, (255, 255, 255), cv2.FILLED)
    return canvas


def find_people_sec(image_list):
    p_sector_list = [None] * len(image_list)

    # FIRST SECTOR
    _, bin_0 = cv2.threshold(image_list[0], 16, 255, cv2.THRESH_BINARY)

    kernelSize_dil = (3, 3)
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize_dil)
    bin_0 = cv2.dilate(bin_0, kernel_dil, iterations=1)

    kernelSize_open = (6, 8)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize_open)
    bin_0 = cv2.morphologyEx(bin_0, cv2.MORPH_OPEN, kernel_open)

    kernelSize_close = (20, 20)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize_close)
    bin_0 = cv2.morphologyEx(bin_0, cv2.MORPH_CLOSE, kernel_close)

    p_sector_list[0] = filter_blobs(bin_0, 80)

    # SECOND SECTOR
    _, bin_1 = cv2.threshold(image_list[1], 16, 255, cv2.THRESH_BINARY)

    kernelSize_dil = (3, 3)
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize_dil)
    bin_1 = cv2.dilate(bin_1, kernel_dil, iterations=1)

    kernelSize_open = (5, 7)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize_open)
    bin_1 = cv2.morphologyEx(bin_1, cv2.MORPH_OPEN, kernel_open)

    kernelSize_close = (20, 20)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize_close)
    bin_1 = cv2.morphologyEx(bin_1, cv2.MORPH_CLOSE, kernel_close)

    p_sector_list[1] = filter_blobs(bin_1, 80)

    # THIRD SECTOR
    _, bin_2 = cv2.threshold(image_list[2], 55, 255, cv2.THRESH_BINARY)

    kernelSize_dil = (3, 3)
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize_dil)
    bin_2 = cv2.dilate(bin_2, kernel_dil, iterations=1)

    kernelSize_open = (2, 2)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize_open)
    bin_2 = cv2.morphologyEx(bin_2, cv2.MORPH_OPEN, kernel_open)

    kernelSize_close = (15, 15)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize_close)
    bin_2 = cv2.morphologyEx(bin_2, cv2.MORPH_CLOSE, kernel_close)

    p_sector_list[2] = filter_blobs(bin_2, 50)

    # FORTH SECTOR
    _, bin_3 = cv2.threshold(image_list[3], 30, 255, cv2.THRESH_BINARY)

    kernelSize_dil = (2, 2)
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize_dil)
    bin_3 = cv2.erode(bin_3, kernel_dil, iterations=1)

    kernelSize_close = (8, 12)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize_close)
    bin_3 = cv2.morphologyEx(bin_3, cv2.MORPH_CLOSE, kernel_close)

    kernelSize_open = (5, 5)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize_open)
    bin_3 = cv2.morphologyEx(bin_3, cv2.MORPH_OPEN, kernel_open)

    p_sector_list[3] = filter_blobs(bin_3, 20)

    # FIFTH SECTOR
    _, bin_4 = cv2.threshold(image_list[4], 20, 255, cv2.THRESH_BINARY)

    kernelSize_open = (6, 3)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize_open)
    bin_4 = cv2.morphologyEx(bin_4, cv2.MORPH_OPEN, kernel_open)

    kernelSize_close = (7, 8)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize_close)
    bin_4 = cv2.morphologyEx(bin_4, cv2.MORPH_CLOSE, kernel_close)

    p_sector_list[4] = filter_blobs(bin_4, 10)

    # SIXTH SECTOR
    _, bin_5 = cv2.threshold(image_list[5], 70, 255, cv2.THRESH_BINARY)

    kernelSize_close = (5, 5)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize_close)
    bin_5 = cv2.morphologyEx(bin_5, cv2.MORPH_CLOSE, kernel_close)

    kernelSize_open = (3, 3)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize_open)
    bin_5 = cv2.morphologyEx(bin_5, cv2.MORPH_OPEN, kernel_open)

    p_sector_list[5] = filter_blobs(bin_5, 10)

    # SEVENTH SECTOR
    _, bin_6 = cv2.threshold(image_list[6], 70, 255, cv2.THRESH_BINARY)

    kernelSize_close = (7, 7)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize_close)
    bin_6 = cv2.morphologyEx(bin_6, cv2.MORPH_CLOSE, kernel_close)

    kernelSize_dil = (4, 4)
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize_dil)
    bin_6 = cv2.dilate(bin_6, kernel_dil, iterations=1)

    p_sector_list[6] = filter_blobs(bin_6, 5)

    return p_sector_list


def sector_based_mask(tophat_img):
    # Cut mountain, boats and sky
    image_cut = tophat_img[440:, :].copy()

    # Divide image in sectors
    img_sectors = divide_img_sectors(image_cut)

    # Find people, create the mask
    mask = np.zeros(image_cut.shape, dtype=np.uint8)

    p_sector_list = find_people_sec(img_sectors)

    # Unify image sectors
    mask_h = mask.shape[0]
    pos_h = 0

    for img in p_sector_list:
        top_h = mask_h - (pos_h + img.shape[0])
        bot_h = mask_h - (pos_h)
        mask[top_h:bot_h, :] = img
        pos_h += img.shape[0]

    return mask


def get_outline_img(img):
    # Gradient of mask
    kernelSize_close = (3, 3)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize_close)
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel_close)


# color in (R,G,B)
def apply_mask_outline_to_img(img, mask_, color):
    image_detected = img.copy()
    overline = get_outline_img(mask_)

    image_detected[0][440:, :][overline == 255] = color[0]
    image_detected[1][440:, :][overline == 255] = color[1]
    image_detected[2][440:, :][overline == 255] = color[2]
    image_detected = np.transpose(image_detected, (1, 2, 0))

    return image_detected


def filter_regions(image, mask):
    contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    filtering_contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    keep_contours_ = []
    used_filters_ = np.zeros(len(filtering_contours[0]))

    for contour in contours[0]:
        keep_contour = False

        for idx, filter in enumerate(filtering_contours[0]):
            # if used_filters[idx]:
            #    continue

            representing_pixel = (int(filter[0][0][0]), int(filter[0][0][1]))

            if cv2.pointPolygonTest(contour=contour, pt=representing_pixel, measureDist=False) != -1:
                keep_contour = True
                used_filters_[idx] = 1
                break

        keep_contours_.append(keep_contour)

    result = np.zeros_like(image)
    filtered_contours = list(compress(list(contours[0]), keep_contours_))
    cv2.drawContours(result, filtered_contours, contourIdx=-1, color=255, thickness=-1)

    result = cv2.dilate(result, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))

    return result


def mask_combiner(skin_based_msk, sector_based_msk):
    intermediate_mask = cv2.bitwise_and(sector_based_msk, skin_based_msk)

    sea_mask = cv2.imread('res/beach_mask.png', cv2.IMREAD_GRAYSCALE)
    sea_detections = cv2.bitwise_and(sea_mask, sector_based_msk)
    intermediate_mask = cv2.bitwise_or(sea_detections, intermediate_mask)

    resulting_mask = filter_regions(sector_based_msk, intermediate_mask)

    return resulting_mask


def compute_centroids(bin_img):
    centroids = []

    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:

        mom = cv2.moments(contour)

        # Calculate centroid
        if mom['m00'] != 0:
            centroid_x = int(mom['m10'] / mom['m00'])
            centroid_y = int(mom['m01'] / mom['m00'])
        else:
            # Avoid division by zero if the area is zero
            centroid_x, centroid_y = 0, 0

        centroids.append([centroid_x, centroid_y + 440])

    return centroids


def analyze_image(image_color):
    empty_beach = cv2.imread('res/empty_beach.jpg', cv2.IMREAD_GRAYSCALE)
    image_gray = cv2.cvtColor(np.transpose(image_color, (1, 2, 0)), cv2.COLOR_RGB2GRAY)

    differences = compute_differences(image_gray, empty_beach)
    skin_based_msk = skin_based_mask(image_color, differences)
    sector_based_msk = sector_based_mask(differences)
    resulting_msk = mask_combiner(skin_based_msk, sector_based_msk)

    centroids = compute_centroids(resulting_msk)

    result_image = apply_mask_outline_to_img(image_color, resulting_msk, np.array([255, 0, 0]))
    result_image = np.ascontiguousarray(result_image)

    return result_image, centroids


# Evaluation for the whole algorithm
def evaluate_mse(gt, detections):
    return np.mean(np.square(gt - detections))


# Evaluation for a single image
def evaluate_accuracy(gt, detections, threshold):
    if len(gt) == 0:
        return 0

    nbrs = NearestNeighbors(n_neighbors=1).fit(gt)
    distances, indices = nbrs.kneighbors(detections)

    return np.sum(distances < threshold) / len(detections)
