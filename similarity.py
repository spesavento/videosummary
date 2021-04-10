# import the necessary packages

# python -m pip install -U scikit-image, Cython, numpy, scipi, matplotlib, networkx, pillow,
# imageio, tifffile, PyWavelets
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import numpy as np # may not be needed
import cv2
import glob

# funtion calc_ssim caclulate the "structural similarity" between 2 images x.jpg and y.jpg in path framePath
# ssim is strucural image similarity ref: https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html
def calc_ssim(framePath, x, y):
    frame_x = cv2.imread(framePath+str(x)+'.jpg')
    frame_y = cv2.imread(framePath+str(y)+'.jpg')
    frame_x_bw = cv2.cvtColor(frame_x, cv2.COLOR_BGR2GRAY)
    frame_y_bw = cv2.cvtColor(frame_y, cv2.COLOR_BGR2GRAY)
    s = ssim(frame_x_bw, frame_y_bw)
    return s

# calculates where to set the similarity threshold number
# because every video is different in action, this samples 30 adjacent frames
# and calculates the similarity theshold point about 20% above min value
def similarity_threshold(framePath):
    # the total number of image frames in framePath
    total_num_frames = int(len(glob.glob1(framePath,"*.jpg")))
    # use about 20 adjacent frames
    y = int(total_num_frames/40)
    x = y
    min_sim = 1.0
    max_sim = 0.0
    set_threshold = 0.00
    for i in range (1,30):
        adjacent_sim = calc_ssim(framePath, x, x+1)
        print ('similarity of frames '+str(x)+' and '+str(x+1)+' is '+str(adjacent_sim))
        x = x + y
        if (adjacent_sim < min_sim):
            min_sim = adjacent_sim
        if (adjacent_sim > max_sim):
            max_sim = adjacent_sim
    print('adjacent similarity range study: min_sim is '+str(min_sim)+ ', max_sim is '+str(max_sim))
    threshold_point = min_sim + 0.2*(max_sim - min_sim)
    print ('threshold_point '+str(threshold_point))
    return threshold_point

# function to summarize video only keep frames that are different, above ssim threshold_point
# framePath should contain sequential numbered images 1.jpg, 2.jpg etc
def cull_frames(framePath, outPath):

    # the total number of image frames in framePath
    total_num_frames = len(glob.glob1(framePath,"*.jpg"))

    # get the similarity point to be used to determine if adjacent images are similar
    similarity_point = similarity_threshold(framePath)

    # starting point with first two adjacent images frames
    x = 1
    y = 2

    # initialize a counter
    counter = 0

    # loop through all adjacent frames and only keep those that are different
    for i in range (1, total_num_frames):

        # calculate adjacent frame similarity
        s = calc_ssim(framePath, x, y)

        # the image to be kept in the outPath folder
        outImage=outPath+str(y)+'.jpg'

        if (s > similarity_point):
            # the images are similar so don't keep y
            y = y + 1
            counter = counter + 1
            if (counter > 10):
                # we have reached the 10th similar image so keep y
                print ('keep 10th frame image '+ str(y))
                # read image into opencv
                currentImage=cv2.imread(framePath+str(y)+'.jpg')
                # add that image to the SSIM summary folder
                cv2.imwrite(outImage, currentImage)
                x = y
                y = y + 1
                counter = 0  # reset the counter
        else:
            # the images are different so keep y
            print ('keep scene change image '+ str(y))
            # read image into opencv
            currentImage=cv2.imread(framePath+str(y)+'.jpg')
            # add that image to the SSIM summary folder
            cv2.imwrite(outImage, currentImage)
            x = y
            y = y + 1
            counter = 0  # reset the counter

def main():
    # framePath is the directory where frames from the FULL video is located, should be ordered 1.jpg, 2.jpg, etc.
    framePath = "/Users/gerrypesavento/Documents/sara/videosummary/localfiles/driveway/frames/"

    # outPath is where the saved frames of the SSIM summary are put, make this folder in advance
    outPath = '/Users/gerrypesavento/Documents/sara/videosummary/localfiles/driveway/summary/ssim/frames/'

    # start
    cull_frames(framePath, outPath)

if __name__=="__main__":
    main()

