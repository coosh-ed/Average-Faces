import sys
from transform_pictures import transform_images, sum_features


if __name__ == "__main__":
    if len(sys.argv)<1:
        print "Please add the directories containing images as command line arguments."
        exit()

    paths = sys.argv[1:]
    
    for f in paths:
        transform_images(f)

    #average the facial feature locations
    num_images, avg_points = sum_features(paths[0])
    if len(paths)>1:
        for f in paths[1:]:
            num, pts = get_average_features(f)
            num_images += num
            avg_points += pts
