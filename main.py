#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import glob
import cv2
import pickle


def display_image(image, title='test'):
    cv2.imshow(title, image)
    cv2.waitKey(0)

def find_vessels(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    display_image(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    display_image(blurred)
    thresh = cv2.adaptiveThreshold(blurred, 256, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    display_image(thresh)
    (_, contours, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def main():
    eye_image = cv2.imread('original.png')
    find_vessels(eye_image)

if __name__ == '__main__':
    main()