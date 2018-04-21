#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import glob
import cv2 as cv
import pickle
import random as rd

def find_vessels(image):
    cpy = image.copy()
    gray = cv.cvtColor(cpy, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.adaptiveThreshold(blurred, 256, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    

    display_image(thresh)

def drawEdges(path):
    img = cv.imread(path)
    height, width, channels = img.shape
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgGray = cv.GaussianBlur(imgGray, (3, 3), 0)
    imgGray = cv.medianBlur(imgGray, 3)
    highThresh, thresh_img = cv.threshold(imgGray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    lowThresh = 0.3 * highThresh
    edges = cv.Canny(imgGray, lowThresh, highThresh)
    display_image(edges)
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

def display_image(image, title='test'):
    cv.imshow(title, image)
    cv.waitKey(0)

def get_red_chanell(image):
    b = image.copy()
    # set green and red channels to 0
    b[:, :, 1] = 0
    b[:, :, 2] = 0


    g = image.copy()
    # set blue and red channels to 0
    g[:, :, 0] = 0
    g[:, :, 2] = 0

    r = image.copy()
    # set blue and green channels to 0
    r[:, :, 0] = 0
    r[:, :, 1] = 0


    # RGB - Blue
    cv.imshow('B-RGB', b)

    # RGB - Green
    cv.imshow('G-RGB', g)

    # RGB - Red
    cv.imshow('R-RGB', r)

    cv.waitKey(0)

def main():
    eye_image = cv.imread('original.png')
    #drawEdges('original.png')
    find_vessels(eye_image)

if __name__ == '__main__':
    main()