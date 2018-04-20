import cv2 as cv
import numpy as np
import random as rd
import os
from matplotlib import pyplot as pt

DATA_DIR = './data/'

RESULTS_DIR = './res/'

def drawEdges(path):
    img = cv.imread(path)
    height, width, channels = img.shape
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgGray = cv.GaussianBlur(imgGray, (3, 3), 0)
    imgGray = cv.medianBlur(imgGray, 3)
    highThresh, thresh_img = cv.threshold(imgGray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    lowThresh = 0.3 * highThresh
    edges = cv.Canny(imgGray, lowThresh, highThresh)
    edges = cv.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    imgCnt, contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    imgC = np.zeros((height, width, 1), np.uint8)
    for i in range(len(contours)):
        moments = cv.moments(contours[i])
        if moments['mu02'] < 400000.0:
            continue
        cv.drawContours(imgC, contours, i, (255, 255, 255), cv.FILLED)
    edges = cv.erode(imgC, np.ones((3, 3), np.uint8), iterations=2)
    highThresh, thresh_img = cv.threshold(edges, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    lowThresh = 0.3 * highThresh
    edges = cv.Canny(edges, lowThresh, highThresh)
    edges = cv.dilate(edges, np.ones((3, 3), np.uint8), iterations=3)
    imgCnt, contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        moments = cv.moments(contours[i])
        cv.drawContours(img, contours, i, (rd.randint(0,255), rd.randint(0,255), rd.randint(0,255)), 2)
        cv.circle(img, (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])), 5, (255, 255, 255), -1)
    return img

if __name__ == '__main__':
    def main():
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        fig, ax = pt.subplots(6, 3, figsize = (30, 40))
        for i, file in enumerate(sorted(os.listdir(DATA_DIR))):
            if file.endswith(".jpg"):
                img = drawEdges(DATA_DIR + file)
                ax[i // 3, i % 3].get_xaxis().set_visible(False)
                ax[i // 3, i % 3].get_yaxis().set_visible(False)
                ax[i // 3, i % 3].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
                cv.imwrite(RESULTS_DIR + 'edges_' + file, img)
        fig.savefig(RESULTS_DIR + 'mosaic.pdf')
    main()
