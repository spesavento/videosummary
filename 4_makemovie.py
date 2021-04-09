
# Make a movie from image frames in a folder pathIn, using OpenCV
# Reference article, this code was modified
# https://www.life2coding.com/convert-image-frames-video-file-using-opencv-python/

# The image frames in pathIn already have the bounding boxes written on them
# Alternative to OpenCV, use ffmpeg (pip3 install ffmpeg)
# ffmpeg -i /Users/folder-of-frames-to-compile/%d.jpg -vcodec mpeg4 bicyle.mp4

import cv2
import numpy as np
import os
from os.path import isfile, join

# function to convert frames to video using OpenCV
def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    # for sorting the file names properly
    # see python reference https://docs.python.org/3/howto/sorting.html
    # the files have to be renamed in order, ex 0010.jpg, 0011.jpg
    # these other versions of sort by key so far don't work
    # files.sort(key = lambda x: int(x[5:-4]))
    # files.sort(key = lambda x: x[5:-4])
    files.sort()

    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        # height, width, layers = img.shape
        # size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)

    # for OSX, mp4 seems to be the best choice for playing videos
    # to go -- automatically read the size, width and height, I had to hardcode 960,540
    # because img.shape did not work
    # Note width, height must be the same as the images
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(pathOut, fourcc, fps, (1280,720))
    # out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

def main():
    # pathIn is the folder were the images to compile to video are located, the must be alphabetically sorted already (ex. 000245.jpg)
    pathIn= '<local path here>/localfiles/driveway/summary/person/frames/'
    # pathOut is where the compiled summary movie is to be located
    pathOut = '<local path here>/localfiles/driveway/summary/person/driveway_summary.mp4'
    # the frame rate should be close to that of the original video
    fps = 29.0
    # call the function to convert to video
    convert_frames_to_video(pathIn, pathOut, fps)

if __name__=="__main__":
    main()




