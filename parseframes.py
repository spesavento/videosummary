
# take a video from S3, parse it into frames, and store those
# frames in S3.  Start with a fixed example video, but will need
# to take any video file and parse it.

#load python libraries
import cv2  # opencv is used to parse the video
import numpy as np
import os, os.path
import sys

# the video to parse
thevideo = str('https://videosummaryfiles.s3-us-west-1.amazonaws.com/ucdavis/full/ucdavis.mp4')

# read the video into opencv
cvvideo = cv2.VideoCapture(thevideo)

# video number of frames
num_frames = int(cvvideo.get(cv2.CAP_PROP_FRAME_COUNT))

# video frame rate
fps = int(cvvideo.get(cv2.CAP_PROP_FPS))

# video time duration in seconds
video_time = num_frames/fps

# pixel width and height
pixelwidth = int(cvvideo.get(cv2.CAP_PROP_FRAME_WIDTH))
pixelheight = int(cvvideo.get(cv2.CAP_PROP_FRAME_HEIGHT))

# output information about the video
print ("\n analyzing video ... ")
print ('video: ',num_frames,'frames,',fps,'fps,',int(video_time),'seconds,', pixelwidth,'width x',pixelheight,'height')

# extract individual frames
print ('\n extracting frames from videos ... this takes a minute')

# define the folder that will receive the video frame images
s3_frames_folder = "s3://videosummaryfiles/ucdavis/frames/"

# parse video into individual frames 1.jpg, 2.jpg, etc
cap = cv2.VideoCapture(thevideo)
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    # need to write files to S3 folder here
    # cv2.imwrite(s3_frames_folder + '/' + str(i)+'.jpg',frame)
    # temp
    print ("frame parsed " + str(i))
    i+=1
print (i+1, ' image frames extracted from video in ', s3_frames_folder)
cap.release()
cv2.destroyAllWindows()