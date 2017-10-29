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
import pandas as pd
from skimage import io

NUM_FEATURES = 68
PREDICTOR_PATH = "learning_data/shape_predictor_68_face_landmarks.dat"

#set size of transformed images
COLS, ROWS = 600, 600

def make_equil_triangle(l_pnt, r_pnt):
    l_pnt, r_pnt = np.array(l_pnt), np.array(r_pnt)
    rad = np.deg2rad(60)
    rotate =  [[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]]
    third_pnt = np.dot(r_pnt-l_pnt, rotate) + l_pnt
    return third_pnt


def point2array(point):
    return np.array((point.x, point.y))


def transform_images(faces_folder_path, out_path):

    #create folders to save pictures
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    #load model
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    #calculate transformation outpoints
    l_eye, r_eye = (180,200), (420,200)
    outPnts = np.array([l_eye, r_eye, make_equil_triangle(l_eye, r_eye)])

    #find feature points for jpg files in faces_folder_path
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):

        img = io.imread(f)

        dets = detector(img, 1)
        for d in dets:
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)

            p1 = point2array(shape.part(36))
            p2 = point2array(shape.part(45))

            inPnts = np.array([p1, p2, make_equil_triangle(p1, p2)])
            M = cv2.estimateRigidTransform(np.float32(inPnts), np.float32(outPnts), False)
            img_transf = cv2.warpAffine(img, M, (COLS,ROWS))
            io.imsave(os.path.join(out_path, os.path.basename(f)), img_transf)

    return 0


def get_average_points(faces_folder_path):

    avg_features = np.array([(0, 0) for i in xrange(NUM_FEATURES)]) 
    num_images = 0

    #load model
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
        img = cv2.imread(f); 
        dets = detector(img, 1)

        for d in dets:
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)

            for i, features in enumerate(avg_features):
                avg_features[i] += point2array(shape.part(i))

            num_images += 1

    return avg_features/num_images


def pointInRect(p, rect):
    #checks if point p is in rectangle rect
    if p[0] < rect[0]:
        return False
    elif p[1] < rect[1]:
        return False
    elif p[0] > rect[2]:
        return False
    elif p[1] > rect[3]:
        return False
    return True


def add_border_points(points):
    #add border points
    border_points = np.array([(0,0), (COLS/2,0), (COLS-1, 0), (0, ROWS/2), (0, ROWS-1),
                              (COLS-1,ROWS/2), (COLS-1,ROWS-1), (COLS/2, ROWS-1)])

    return np.append(points, border_points, axis=0)


def get_list_of_triangles(points):
    #get list of Delanay triangles
    
    rect = (0, 0, COLS, ROWS)
    subdiv = cv2.Subdiv2D(rect)
    
    for pnt in points:
        subdiv.insert((pnt[0], pnt[1]))

    triList = []
    for tri in subdiv.getTriangleList():
        tri = [(tri[0],tri[1]), (tri[2],tri[3]), (tri[4],tri[5])]

        #make sure triangle points are subset of feature points
        if pointInRect(tri[0], rect) and pointInRect(tri[1], rect) and pointInRect(tri[2], rect):
            ind = []
            for j in xrange(0, 3):
                for k in xrange(0, len(points)):                    
                    if (abs(tri[j][0] - points[k][0]) < 1.0 and abs(tri[j][1] - points[k][1]) < 1.0):
                        ind.append(k)                            
            if len(ind) == 3:                                                
                triList.append(ind)
            
    return triList


def applyAffineTransform(src, srcTri, dstTri, size):
    #based on code from https://github.com/spmallck/learnopencv
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri))
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst



def warp_triangle(img1, img2, tri1, tri2):
    #based on code from https://github.com/spmallick/learnopencv/
    #warps triangle tri1 in img1 to tri2 in img2

    # Find bounding rectangle for each triangle
    rect1 = cv2.boundingRect(np.float32([tri1]))
    rect2 = cv2.boundingRect(np.float32([tri2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in xrange(0, 3):
        t1Rect.append(((tri1[i][0] - rect1[0]), (tri1[i][1] - rect1[1])))
        t2Rect.append(((tri2[i][0] - rect2[0]), (tri2[i][1] - rect2[1])))
        t2RectInt.append(((tri2[i][0] - rect2[0]),(tri2[i][1] - rect2[1])))


    # Get mask by filling triangle
    mask = np.zeros((rect2[3], rect2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0] + rect1[2]]
    
    size = (rect2[2], rect2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] = img2[rect2[1]:rect2[1]+rect2[3], 
                                                                        rect2[0]:rect2[0]+rect2[2]] * ( (1.0, 1.0, 1.0) - mask )
     
    img2[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] = img2[rect2[1]:rect2[1]+rect2[3], 
                                                                        rect2[0]:rect2[0]+rect2[2]] + img2Rect
    return
    



def morph_to_average_face(folder_path, avg_points, output_diffs):

    #create output image
    output = np.zeros((COLS,ROWS,3), np.float32())

    #add border points for the average feature points
    avg_points = add_border_points(avg_points)

    #get Dalaunay Triangles for the average features
    avgTriangles = get_list_of_triangles(avg_points)

    #load model
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    if output_diffs:
        df = []

    #read in images
    for f in glob.glob(os.path.join(folder_path, "*.jpg")):

        img = io.imread(f)
        dets = detector(img, 1)

        for k, d in enumerate(dets):
            # Get the landmarks/parts for the face in box d.

            new_img = np.zeros((COLS,ROWS,3), np.float32())

            shape = predictor(img, d)
            pnts_features = []
            for i in range(NUM_FEATURES):
                x_coord, y_coord = min(COLS-1, shape.part(i).x), min(ROWS-1, shape.part(i).y)
                pnts_features.append(np.array((x_coord, y_coord)))

            #add border points
            pnts_features = add_border_points(pnts_features)
            
            if output_diffs:
                #calculate the distance between points of two pictures
                mean = np.mean([np.sqrt(np.sum((x-y)**2)) for x, y in zip(pnts_features, avg_points)])
                df.append((f.strip(".jpg").split('/')[1], mean))

            #warp each triangle to new image
            for tri in avgTriangles:
                inTri = [pnts_features[ind] for ind in tri]
                outTri = [avg_points[ind] for ind in tri]

                warp_triangle(img, new_img, inTri, outTri)

            output += new_img

    num_images = len(glob.glob(os.path.join(folder_path, "*.jpg")))
    output = output/(255.0*num_images)
    io.imsave("output.jpg", output)

    if output_diffs:
        df_out = pd.DataFrame(df, columns=['TD', 'difference'])
        df_out.to_csv('differences.csv', index=False)

    return


if __name__ == '__main__':
    path = 'test/'
    transfPath = 'testTransform/'
    #normalise images
    print "Transforming images to common dimensions..."
    transform_images(path, transfPath)

    #average the facial feature locations
    print "Averaging facial feature locations..."

    avg_points = get_average_points(transfPath)

    #morph transformed images into the average
    print "Morphing images into output image..."

    morph_to_average_face(transfPath, avg_points)

    print "Done!"
