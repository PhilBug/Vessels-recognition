#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import glob
import cv2
import pickle

with open('knowledgeBasePICKLE', 'rb') as handle:
    trainingSet2 = pickle.load(handle)

# print(trainingSet2)
# wartosci z obrazka wpisane na sztywno

trainingSet2 = [ord(i) for i in trainingSet2]
trainingSet2 = np.array(trainingSet2, np.float32)
trainingSet2.reshape((trainingSet2.size, 1))

_results = []


def display_image(image, title='test'):
    cv2.imshow(title, image)
    cv2.waitKey(0)


def create_knowledge_base(image, contours_sorted):  # x = cv[0], y = cv[1], w = cv[2], h = cv[3]
    dataSet = np.empty(shape=(0, 100))
    for cnt in contours_sorted:
        cropImg = image[cnt[1]:cnt[1] + cnt[3], cnt[0]:cnt[0] + cnt[2]]
        imgSmall = scale_down_image(cropImg)

        # display_image(imgSmall)

        singleImgData = imgSmall.reshape((1, 100))
        dataSet = np.append(dataSet, singleImgData, axis=0)
    return dataSet


def sort_contours(contours):
    sort = sorted(contours, key=lambda e: e[1] + e[0], reverse=False)
    return sort


def scale_down_image(cropImage):
    return cv2.resize(cropImage, (10, 10))


def find_digits(imageLearning, imageEquation):
    contours_to_sort = []
    equationContours = []
    img_c = imageLearning.copy()
    gray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 256, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    (_, contours, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    eq = imageEquation.copy()
    grayE = cv2.cvtColor(eq, cv2.COLOR_BGR2GRAY)
    blurredE = cv2.GaussianBlur(grayE, (5, 5), 0)
    threshE = cv2.adaptiveThreshold(blurredE, 256, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    (_, contoursE, _) = cv2.findContours(threshE, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        # display_image(cv2.rectangle(img_c, (x, y), (x + w, y + h), (0, 0, 255), 2))

        contours_to_sort.append([x, y, w, h])

    for cnt in contoursE:
        (x, y, w, h) = cv2.boundingRect(cnt)
        equationContours.append([x, y, w, h])

    # contours_sorted = sort_contours(contours_to_sort)

    dataSet = create_knowledge_base(thresh, contours_to_sort)  # contours_sorted
    return digit_recognition(threshE, dataSet, equationContours)  # contours_sorted


def digit_recognition(image, dataSet, contours_sorted):
    model = cv2.ml.KNearest_create()
    dataSet = dataSet.astype(np.float32)
    model.train(dataSet, cv2.ml.ROW_SAMPLE, trainingSet2)

    for cnt in contours_sorted:
        cropImg = image[cnt[1]:cnt[1] + cnt[3], cnt[0]:cnt[0] + cnt[2]]
        imgSmall = scale_down_image(cropImg)
        imgSmall = imgSmall.reshape((1, 100))
        imgSmall = np.float32(imgSmall)
        (_, results, _, _) = model.findNearest(imgSmall, k=1)
        sign = chr(int(results[0][0]))
        _results.append(sign)
    return calculate_result(_results)


def calculate_result(signs):
    try:
        if signs[0] == '/':
            return int(signs[2]) / int(signs[1])
        elif signs[0] == 'x':
            return int(signs[2]) * int(signs[1])
        elif signs[0] == '+':
            return int(signs[2]) + int(signs[1])
        elif signs[0] == '-':
            return int(signs[2]) - int(signs[1])
    except ZeroDivisionError:
        print('Cannot divide by zero')


def final(eqPath):
    learningSet = glob.glob('./knowledgebase/arial_black_final.png')
    imageLearning = cv2.imread(learningSet[0])
    equation = glob.glob(eqPath)
    imageEquation = cv2.imread(equation[0])
    return find_digits(imageLearning, imageEquation)


def main():
    try:
        for i in glob.glob('samples/*.png'):
            result = final(i)
            print('{} {} {} = {}'.format(_results[2], _results[0], _results[1], int(result)))
            _results.clear()
    except IndexError:
        print('No *.png files in project folder!')
        exit(1)


if __name__ == '__main__':
    main()