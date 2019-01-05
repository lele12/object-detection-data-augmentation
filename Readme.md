# Data augmentation for object detection

The data augmentation methods have crop, rotate, flip, resize,  add noise, expand, photometric distort and so on.

#### Parameters:

imgname : the path of image

boxes:  the coordinate of bounding boxes (xmin, ymin, xmax, ymax). data type: numpy.ndaaray

labels: the label of the bounding boxes. data type: numpy.ndaaray



return: 

auged_img: the image of augmentation

auged_boxes: the bounding boxes of the image

auged_labels: the labels of the bounding boxes 