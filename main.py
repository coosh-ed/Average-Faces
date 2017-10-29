#!/usr/bin/python
"""This program runs the main functions required to average faces. To run, please pass the directory containing 
the images as a commmand line argument as so: 
     ./main.py path/to/images"""

import datetime, time
import sys
from transform_pictures import transform_images, get_average_points, morph_to_average_face

if __name__ == "__main__":
    if len(sys.argv)<2 or sys.argv[-1] == "--calc-diff":
        print "Please add the directories containing images as command line arguments."
        exit()

    output_diffs = True if "--calc-diff" in sys.argv else False

    #suffix to use when saving transformed images
    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M')
    suffix = "Translate"+time_stamp

    path = sys.argv[-1]
    out_path = path.strip('\/')+"Tranformed/"

    #normalise images
    print "Transforming images to common dimensions..."

    transform_images(path, out_path)
        
    #average the facial feature locations
    print "Averaging facial feature locations..."

    avg_points = get_average_points(out_path)
    
    #morph transfored images into the average
    print "Morphing images into output image..."
    morph_to_average_face(out_path, avg_points, output_diffs)

    print "Done!"
