import cv2
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt

def skincolor(images):
    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([235, 173, 127], np.uint8)

    img2 = cv2.imread("images/{}".format(images[5]), cv2.IMREAD_COLOR)
    imageYCrCb = cv2.cvtColor(img2, cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)

    skinYCrCb = cv2.bitwise_and(img2, img2, mask=skinRegionYCrCb)
    cv2.imshow("", img2)
    cv2.imshow("", skinYCrCb)
    cv2.waitKey(0)

def grayscale(img1, img2):
    img1_eq = cv2.equalizeHist(img1)
    img2_eq = cv2.equalizeHist(img2)

    out = np.abs(cv2.subtract(img1_eq, img2_eq))

    cv2.imshow("", out)
    cv2.waitKey(0)

    sobelxy = cv2.Sobel(src=out, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)  # Combined X and Y Sobel Edge Detection
    cv2.imshow("", sobelxy)
    cv2.waitKey(0)

    edges = cv2.Canny(image=out, threshold1=100, threshold2=200)
    cv2.imshow("", edges)
    cv2.waitKey(0)


    #_,out_thresh = cv2.threshold(out, 0, 255, type=cv2.THRESH_OTSU)
    #cv2.imshow("", out_thresh)
    #cv2.waitKey(0)

def bluring(img1, images):
    img1 = cv2.GaussianBlur(img1, (13, 13), 0)

    mask = np.zeros_like(img1)
    for image in images[1:]:
        img_ = cv2.imread("images/{}".format(image), cv2.IMREAD_GRAYSCALE)
        img_ = cv2.GaussianBlur(img_, (13,13), 0)
        dif = cv2.subtract(img1, img_)
        mask = cv2.bitwise_or(mask, dif)

    plt.imshow(mask, cmap="gray")
    plt.show()

    out = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #_,out = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU)
    plt.imshow(out, cmap="gray")
    plt.show()



if __name__ == "__main__":
    # Show effect
    images = os.listdir("images")
    images.pop()

    df = pd.read_csv("images/labels_people_detection.csv",delimiter=",",usecols=["Image","X","Y"])

    image_count = pd.DataFrame(columns=["Image","NPeople"])

    for img in images:
        if img not in np.unique(df["Image"]):
            image_count = image_count._append({"Image": img, "NPeople": 0}, ignore_index=True)

    for img in np.unique(df["Image"]):
        n_people = np.sum(df["Image"] == img)
        image_count = image_count._append({"Image": img,"NPeople": n_people}, ignore_index=True)

    for image in images:
        img = cv2.imread("images/{}".format(image), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for index, row in df[df["Image"] == image].iterrows():
            print(row["X"])
            cv2.circle(img, (row["X"],row["Y"]), 5, (0, 255, 0), 5)

        plt.imshow(img)
        plt.show()
