#!/usr/bin/python2
# -*- coding: utf-8 -*-
# vim: expandtab:ts=4:sw=4
# --------------------------------------------------------
import cv2
import copy
import numpy as np
from numpy import random


def bbox_area(bbox):
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])

def clip_box(bbox, labels, clip_box, alpha):
    """Clip the bounding boxes to the borders of an image
    """
    ar_ = (bbox_area(bbox))

    delta_area = ((ar_ - bbox_area(clip_box)) / ar_)

    mask = (delta_area < (1 - alpha)).astype(int)


    bbox = bbox[mask == 1, :]
    new_labels = labels[mask == 1]

    return bbox, new_labels

class Compose(object):
    def __init__(self, transfroms):
        self.transfroms = transfroms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transfroms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class ConvertFromInts(object):
    def __call__(self, img, boxes=None, labels=None):
        return img.astype(np.float32), boxes.astype(np.float32), labels

class ConvertFromFloats(object):
    def __call__(self, img, boxes=None, labels=None):
        return img.astype(np.int32), boxes.astype(np.int32), labels

class Normalization(object):
    """Normalization: Pixel Value (0, 255) to (0, 1)"""
    def __call__(self, img, boxes=None, labels=None):
        img /= 255
        return img, boxes, labels

class InverseNormalization(object):
    """Inverse Normalization: Pixel Value (0, 1) to (0, 255)"""
    def __call__(self, img, boxes=None, labels=None):
        img *= 255
        img = np.clip(img, 0, 255)
        return img, boxes, labels

class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, img, boxes=None, labels=None):
        img = img.astype(np.float32)
        img -= self.mean
        return img.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, img, boxes=None, labels=None):
        height, width, channels = img.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        return img, boxes, labels

class ToPercentCoords(object):
    def __call__(self, img, boxes=None, labels=None):
        height, width, labels = img.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height
        return img, boxes, labels

class HorizontalFlip(object):
    def __call__(self, img, boxes=None, labels=None):
        img = cv2.flip(img, 1)
        height, width, channels = img.shape
        boxes[:, 0] = width - boxes[:, 0]
        boxes[:, 2] = width - boxes[:, 2]
        return img, boxes, labels

class RandomHorizontalFlip(object):
    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2):
            img = cv2.flip(img, 1)
            height, width, channels = img.shape
            boxes[:, 0] = width - boxes[:, 0]
            boxes[:, 2] = width - boxes[:, 2]
        return img, boxes, labels

class VehicleFlip(object):
    def __call__(self, img, boxes=None, labels=None):
        img = cv2.flip(img, 0)
        height, width, channels = img.shape
        boxes[:, 1] = height - boxes[:, 1]
        boxes[:, 3] = height - boxes[:, 3]
        return img, boxes, labels


class RandomVehicleFlip(object):
    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2):
            img = cv2.flip(img, 0)
            height, width, channels = img.shape
            boxes[:, 1] = height - boxes[:, 1]
            boxes[:, 3] = height - boxes[:, 3]
        return img, boxes, labels


class Resize(object):
    """Opencv resize"""
    def __init__(self, width=300, height=300):
        self.width = width
        self.height = height

    def __call__(self, img, boxes=None, labels=None):
        height, width, channels = img.shape
        scale_x = float(self.width) / float(width)
        scale_y = float(self.height) / float(height)
        img=  cv2.resize(img, None, fx = scale_x, fy = scale_y, interpolation=cv2.INTER_CUBIC)
        boxes[:, 0] *= scale_x
        boxes[:, 2] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 3] *= scale_y
        return img, boxes, labels

class LetterBox(object):
    """Resize the image in accordance to 'image_letter_box' function in darknet
    The aspect ratio is maintained."""
    def __init__(self, width=600, height=600):
        self.width = width
        self.height = height

    def __call__(self, img, boxes=None, labels=None):
        height, width, channels = img.shape
        scale_x = float(self.width) / float(width)
        scale_y = float(self.height) / float(height)

        if scale_x > scale_y:
            scale = scale_y
        else:
            scale = scale_x

        new_w = int(width * scale)
        new_h = int(height * scale)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        new_img = np.full((self.height, self.width, 3), 128)

        new_img[(self.height - new_h)//2:(self.height - new_h)//2 + new_h,(self.width - new_w)//2:(self.width - new_w)//2 + new_w, :] = resized_img
        boxes *= scale
        boxes[:, 0] += (self.width - new_w)//2
        boxes[:, 2] += (self.width - new_w)//2
        boxes[:, 1] += (self.height - new_h)//2
        boxes[:, 3] += (self.height - new_h)//2

        return new_img, boxes, labels

class ConvertRGB2HSV(object):
    def __call__(self, img, boxes=None, labels=None):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        return img, boxes, labels

class ConvertHSV2RGB(object):
    def __call__(self, img, boxes, labels):
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img, boxes, labels

class RandomSaturation(object):
    """Transfrom the image in HSV color space"""
    def __init__(self, saturation=0.5):
        assert saturation > 0.0 and saturation < 1.0
        self.saturation = saturation

    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2):
            img[:, :, 1] *= random.uniform(-self.saturation, self.saturation)
        return img, boxes, labels

class RandomHue(object):
    """Transfrom the image in HSV color space"""
    def __init__(self, hue=18.0):
        assert hue >= 0.0 and hue <= 360.0
        self.hue = hue

    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2):
            img[:, :, 0] += random.uniform(-self.hue, self.hue)
            img[:, :, 0] = np.clip(img[:, :, 0], 0, 360)
        return img, boxes, labels

class RandomBrightness(object):
    """Tranfrom the image in RGB color space"""
    def __init__(self, brightness=32):
        assert brightness > 0.0 and brightness < 255.0
        self.brightness = brightness

    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2):
            img += random.uniform(-self.brightness, self.brightness)
            img = np.clip(img, 0.0, 255.0)
        return img, boxes, labels

class RandomContrast(object):
    """Tranfrom the image in RGB color space"""
    def __init__(self, contrast=0):
        assert contrast > 0.0 and contrast < 1.0
        self.contrast = contrast

    def __call__(self, img, boxes, labels):
        if random.randint(2):
            img *= random.uniform(-self.contrast, self.contrast)
            img = np.clip(img, 0.0, 255.0)
        return img, boxes, labels

class PhotometricDistort(object):
    def __init__(self):
        self.pd = Compose([
            RandomContrast(),
            ConvertRGB2HSV(),
            RandomSaturation(),
            RandomHue(),
            ConvertHSV2RGB(),
            RandomBrightness()
        ])

    def __call__(self, img, boxes=None, labels=None):
        return self.pd(img, boxes, labels)

class AddGaussNoise(object):
    def __init__(self, sigma=2):
        self.sigma = sigma

    def __call__(self, img, boxes=None, labels=None):
        sigma = random.uniform(0, self.sigma)
        img = cv2.GaussianBlur(img, (5,5), sigmaX=sigma)
        return img, boxes, labels

class Expand(object):
    """Expand the image
    mean: the pixel value of the expand area.
    ratio: the max ratio of the orgin image width/height with expanded image width.height """
    def __init__(self, mean=128, ratio=2):
        self.ratio = ratio
        self.mean = mean

    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2):
            return img, boxes, labels

        ratio = random.uniform(1, self.ratio)
        height, width, channels = img.shape
        new_w = int(width * ratio)
        new_h = int(height * ratio)
        left = random.randint(0, new_w - width)
        top = random.randint(0, new_h - height)
        expand_image = np.full((new_h, new_w, 3), self.mean)
        expand_image[top:top+height, left:left+width, :] = img
        img = expand_image

        boxes[:, 0] += left
        boxes[:, 2] += left
        boxes[:, 1] += top
        boxes[:, 3] += top

        return img, boxes, labels


class Four_Point_Crop(object):
    """crop_x and crop_y  range from -0.5 to 0.5
    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped.
    crop_x > 0 , crop_y > 0 crop image upper left part
    crop_x > 0 , crop_y < 0 crop image Lower left part
    crop_x < 0 , crop_y > 0 crop image upper right part
    crop_x < 0 , crop_y < 0 crop image right lower part
    """
    def __init__(self, crop_x=0.2, crop_y=0.2):
        assert crop_x < 0.5 and crop_x > -0.5
        assert crop_y < 0.5 and crop_y > -0.5

        self.crop_x = crop_x
        self.crop_y = crop_y

    def __call__(self, img, boxes=None, labels=None):
        height, width, channels = img.shape
        new_w = int(width * (1 - abs(self.crop_x)))
        new_h = int(height * (1 - abs(self.crop_y)))
        [left, top, right, bottom] = [0, 0, width, height]
        if self.crop_x >= 0 and self.crop_y >= 0:
            right = new_w
            bottom = new_h

        if self.crop_x <= 0 and self.crop_y <= 0:
            left = width - new_w
            top = height - new_h

        if self.crop_x >=0 and self.crop_y <= 0:
            top = height -new_h
            right = new_w

        if self.crop_x <= 0 and self.crop_y >= 0:
            left = width - new_w
            bottom = new_h
        new_img = img[top:bottom, left:right, :]

        new_boxes = copy.deepcopy(boxes)
        new_boxes[:, 0] = np.maximum(boxes[:, 0], left)
        new_boxes[:, 1] = np.maximum(boxes[:, 1], top)
        new_boxes[:, 2] = np.minimum(boxes[:, 2], right)
        new_boxes[:, 3] = np.minimum(boxes[:, 3], bottom)
        boxes, labels = clip_box(boxes, labels, new_boxes, 0.25)
        boxes[:, 0] -= left
        boxes[:, 1] -= top
        boxes[:, 2] -= left
        boxes[:, 3] -= top
        return new_img, boxes, labels

class CenterCrop(object):
    """crop_x and crop_y  range from 0 to 0.5
    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped.
    """
    def __init__(self, crop_x=0.5, crop_y=0.5):
        assert crop_x < 0.5 and crop_x > 0
        assert crop_y < 0.5 and crop_y > 0

        self.crop_x = crop_x
        self.crop_y = crop_y

    def __call__(self, img, boxes=None, labels=None):
        height, width, channels = img.shape
        new_w = int(width * (1 - abs(self.crop_x)))
        new_h = int(height * (1 - abs(self.crop_y)))
        left = (width - new_w) // 2
        top  = (height -new_h) // 2
        right = left + new_w
        bottom = top + new_h

        new_img = img[top:bottom, left:right, :]
        new_boxes = copy.deepcopy(boxes)
        new_boxes[:, 0] = np.maximum(boxes[:, 0], left)
        new_boxes[:, 1] = np.maximum(boxes[:, 1], top)
        new_boxes[:, 2] = np.minimum(boxes[:, 2], right)
        new_boxes[:, 3] = np.minimum(boxes[:, 3], bottom)
        boxes, labels = clip_box(boxes, labels, new_boxes, 0.25)
        boxes[:, 0] -= left
        boxes[:, 1] -= top
        boxes[:, 2] -= left
        boxes[:, 3] -= top
        return new_img, boxes, labels

class RandomFour_Point_Crop(object):
    """crop_x and crop_y  range from 0 to 0.5
    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped.
    """
    def __init__(self, crop_x=0.2, crop_y=0.2):
        assert crop_x < 0.5 and crop_x > 0
        assert crop_y < 0.5 and crop_y > 0

        self.crop_x = crop_x
        self.crop_y = crop_y

    def __call__(self, img, boxes=None, labels=None):
        crop_x = random.uniform(-self.crop_x, self.crop_x)
        crop_y = random.uniform(-self.crop_y, self.crop_y)
        height, width, channels = img.shape
        new_w = int(width * (1 - abs(crop_x)))
        new_h = int(height * (1 - abs(crop_y)))
        [left, top, right, bottom] = [0, 0, width, height]
        if crop_x >= 0 and crop_y >= 0:
            right = new_w
            bottom = new_h

        if crop_x <= 0 and crop_y <= 0:
            left = width - new_w
            top = height - new_h

        if crop_x >=0 and crop_y <= 0:
            top = height -new_h
            right = new_w

        if crop_x <= 0 and crop_y >= 0:
            left = width - new_w
            bottom = new_h
        new_img = img[top:bottom, left:right, :]

        new_boxes = copy.deepcopy(boxes)
        new_boxes[:, 0] = np.maximum(boxes[:, 0], left)
        new_boxes[:, 1] = np.maximum(boxes[:, 1], top)
        new_boxes[:, 2] = np.minimum(boxes[:, 2], right)
        new_boxes[:, 3] = np.minimum(boxes[:, 3], bottom)
        boxes, labels = clip_box(boxes, labels, new_boxes, 0.25)
        boxes[:, 0] -= left
        boxes[:, 1] -= top
        boxes[:, 2] -= left
        boxes[:, 3] -= top
        return new_img, boxes, labels

class RandomCenterCrop(object):
    """crop_x and crop_y  range from 0 to 0.5
    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped.
    """
    def __init__(self, crop_x=0.5, crop_y=0.5):
        assert crop_x < 0.5 and crop_x > 0
        assert crop_y < 0.5 and crop_y > 0

        self.crop_x = crop_x
        self.crop_y = crop_y

    def __call__(self, img, boxes=None, labels=None):
        crop_x = random.uniform(-self.crop_x, self.crop_x)
        crop_y = random.uniform(-self.crop_y, self.crop_y)
        height, width, channels = img.shape
        new_w = int(width * (1 - abs(crop_x)))
        new_h = int(height * (1 - abs(crop_y)))
        left = (width - new_w) // 2
        top  = (height -new_h) // 2
        right = left + new_w
        bottom = top + new_h

        new_img = img[top:bottom, left:right, :]
        new_boxes = copy.deepcopy(boxes)
        new_boxes[:, 0] = np.maximum(boxes[:, 0], left)
        new_boxes[:, 1] = np.maximum(boxes[:, 1], top)
        new_boxes[:, 2] = np.minimum(boxes[:, 2], right)
        new_boxes[:, 3] = np.minimum(boxes[:, 3], bottom)
        boxes, labels = clip_box(boxes, labels, new_boxes, 0.25)
        boxes[:, 0] -= left
        boxes[:, 1] -= top
        boxes[:, 2] -= left
        boxes[:, 3] -= top
        return new_img, boxes, labels

class RandomCrop(object):
    """crop_x and crop_y  range from 0 to 0.5
    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped.
    """
    def __init__(self, crop_x=0.3, crop_y=0.3):
        self.crop_x = crop_x
        self.crop_y = crop_y

    def __call__(self, img, boxes=None, labels=None):
        if random.randint(0, 5) == 0:
            randomcrop = Compose([RandomCenterCrop(self.crop_x, self.crop_y)])
        else:
            randomcrop = Compose([RandomFour_Point_Crop(self.crop_x, self.crop_y)])
        img, boxes, labels = randomcrop(img, boxes, labels)
        return img, boxes, labels

class Rotate(object):
    """Rotates an image.
    Note: the big angle make the lower-left corner of the image lost.
    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped.
    """
    def __init__(self, angle=3):
        assert angle > -360  and angle < 360
        self.angle = angle

    def __call__(self, img, boxes=None, labels=None):
        height, width, channels = img.shape
        center_x, center_y = width // 2, height // 2
        M = cv2.getRotationMatrix2D((center_x, center_y), self.angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int(height * sin + width * cos)
        nH = int(height * cos + width * sin)
        M[0, 2] += (nW / 2) - center_x
        M[1, 2] += (nH / 2) - center_y

        img = cv2.warpAffine(img, M, (nW, nH))

        corners = np.hstack((boxes[:, 0].reshape(-1, 1), boxes[:, 1].reshape(-1, 1),
                             boxes[:, 2].reshape(-1, 1), boxes[:, 1].reshape(-1, 1),
                             boxes[:, 0].reshape(-1, 1), boxes[:, 3].reshape(-1, 1),
                             boxes[:, 2].reshape(-1, 1), boxes[:, 3].reshape(-1, 1)))
        corners = corners.reshape(-1, 2)
        corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))
        corners = np.dot(M, corners.T).T
        corners = corners.reshape(-1, 8)

        corners_x = corners[:, [0, 2, 4, 6]]
        corners_y = corners[:, [1, 3, 5, 7]]
        xmin = np.min(corners_x, 1).reshape(-1, 1)
        ymin = np.min(corners_y, 1).reshape(-1, 1)
        xmax = np.max(corners_x, 1).reshape(-1, 1)
        ymax = np.max(corners_y, 1).reshape(-1, 1)
        boxes = np.hstack((xmin, ymin, xmax, ymax))

        new_boxes = copy.deepcopy(boxes)
        new_boxes[:, 0] = np.maximum(boxes[:, 0], 0)
        new_boxes[:, 1] = np.maximum(boxes[:, 1], 0)
        new_boxes[:, 2] = np.minimum(boxes[:, 2], width)
        new_boxes[:, 3] = np.minimum(boxes[:, 3], height)
        boxes, labels = clip_box(boxes, labels, new_boxes, 0.25)
        return img, boxes, labels

class RandomRotate(object):
    """Rotates an image.
    Note: the big angle make the lower-left corner of the image lost.
    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped.
    """
    def __init__(self, angle=10):
        assert angle > 0  and angle < 360
        self.angle = angle



    def __call__(self, img, boxes=None, labels=None):
        print(self.angle)
        angle = random.uniform(-self.angle, self.angle)
        print(angle)

        height, width, channels = img.shape
        center_x, center_y = width // 2, height // 2
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int(height * sin + width * cos)
        nH = int(height * cos + width * sin)
        M[0, 2] += (nW / 2) - center_x
        M[1, 2] += (nH / 2) - center_y

        img = cv2.warpAffine(img, M, (nW, nH))

        corners = np.hstack((boxes[:, 0].reshape(-1, 1), boxes[:, 1].reshape(-1, 1),
                             boxes[:, 2].reshape(-1, 1), boxes[:, 1].reshape(-1, 1),
                             boxes[:, 0].reshape(-1, 1), boxes[:, 3].reshape(-1, 1),
                             boxes[:, 2].reshape(-1, 1), boxes[:, 3].reshape(-1, 1)))
        corners = corners.reshape(-1, 2)
        corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))
        corners = np.dot(M, corners.T).T
        corners = corners.reshape(-1, 8)

        corners_x = corners[:, [0, 2, 4, 6]]
        corners_y = corners[:, [1, 3, 5, 7]]
        xmin = np.min(corners_x, 1).reshape(-1, 1)
        ymin = np.min(corners_y, 1).reshape(-1, 1)
        xmax = np.max(corners_x, 1).reshape(-1, 1)
        ymax = np.max(corners_y, 1).reshape(-1, 1)
        boxes = np.hstack((xmin, ymin, xmax, ymax))

        new_boxes = copy.deepcopy(boxes)
        new_boxes[:, 0] = np.maximum(boxes[:, 0], 0)
        new_boxes[:, 1] = np.maximum(boxes[:, 1], 0)
        new_boxes[:, 2] = np.minimum(boxes[:, 2], width)
        new_boxes[:, 3] = np.minimum(boxes[:, 3], height)
        boxes, labels = clip_box(boxes, labels, new_boxes, 0.25)
        return img, boxes, labels


class Augmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            RandomRotate(),
            ConvertFromFloats()
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)


def main():
    imgname = ''
    boxes =[]
    labels = []
    img = cv2.imread(imgname)
    dataAug = Augmentation()
    auged_img, auged_bboxes, auged_labels = dataAug(img, boxes, labels)

if __name__ == "__main__":
    main()