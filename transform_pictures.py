#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image and
#   estimate their pose.  The pose takes the form of 68 landmarks.  These are
#   points on the face such as the corners of the mouth, along the eyebrows, on
#   the eyes, and so forth.
#
#   The face detector we use is made using the classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
#   and sliding window detection scheme.  The pose estimator was created by
#   using dlib's implementation of the paper:
#      One Millisecond Face Alignment with an Ensemble of Regression Trees by
#      Vahid Kazemi and Josephine Sullivan, CVPR 2014
#   and was trained on the iBUG 300-W face landmark dataset (see
#   https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):  
#      C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic. 
#      300 faces In-the-wild challenge: Database and results. 
#      Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.

import sys
import os
import dlib
import glob
import cv2
import numpy as np
from skimage import io

NUM_FEATURES = 68
predictor_path = "learning_data/shape_predictor_68_face_landmarks.dat"

def make_equil_triangle(l_pnt, r_pnt):
    l_pnt, r_pnt = np.array(l_pnt), np.array(r_pnt)
    rad = np.deg2rad(60)
    third_pnt =  np.dot(r_pnt-l_pnt, [[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
    third_pnt += l_pnt
    return third_pnt


def point2array(point):
    return np.array((point.x, point.y))


def transform_images(faces_folder_path):

    #create folders to save pictures
    new_folder = os.path.basename(faces_folder_path.strip("\/."))
    new_folder +='Transformed'
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    #load model
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    #set size of outputted images
    cols, rows = 600, 600

    #calculate transformation outpoints
    l_eye, r_eye = (180,200), (420,200)
    outPnts = np.array([l_eye, r_eye, make_equil_triangle(l_eye, r_eye)])

    #find feature points for jpg files in faces_folder_path
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        print("Processing file: {}".format(f))
        img = io.imread(f)

        dets = detector(img, 1)
        for k, d in enumerate(dets):
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            print "Part 36: {}, Part 45: {}".format(shape.part(36), shape.part(45))
            p1 = point2array(shape.part(36))
            p2 = point2array(shape.part(45))
            inPnts = np.array([p1, p2, make_equil_triangle(p1, p2)])
            M = cv2.estimateRigidTransform(inPnts, outPnts, False)
            img_transf = cv2.warpAffine(img,M,(cols,rows))
            io.imsave(os.path.join(new_folder, os.path.basename(f)), img_transf)

    return 0

def sum_features(faces_folder_path)
    avg_features = np.array([(0,0) for i in xrange(NUM_FEATURES)]) 
    num_images = 0

    #load model
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        print("Processing file: {}".format(f))
        img = io.imread(f)

        dets = detector(img, 1)
        for k, d in enumerate(dets):
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            for i, features in enumerate(avg_features):
                avg_features[i] += point2array(shape.part(i))

            num_images += 1

    return num_images, avg_features/num_images


if __name__ == '__main__':
    transform_images('test/')
    n, a = sum_features('test/')
    print n
    for i in a:
        print i
