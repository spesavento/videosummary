
# Makes a movie from image frames in a folder pathIn, using OpenCV
# Reference article
# https://www.life2coding.com/convert-image-frames-video-file-using-opencv-python/

# Alternative method to OpenCV, use ffmpeg (pip3 install ffmpeg)
# ffmpeg -i /Users/folder-of-frames-to-compile/%d.jpg -vcodec mpeg4 bicyle.mp4

import cv2
import numpy as np
import os
from os.path import isfile, join

# function to convert frames to video using OpenCV
def convert_frames_to_video(pathIn,pathOut,fps,frame_width,frame_height):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn,f))]

    # for sorting the file names properly
    # see python reference https://docs.python.org/3/howto/sorting.html
    # the files have to be renamed in order, ex 0010.jpg, 0011.jpg
    # these other versions of sort by key so far don't work
    # files.sort(key = lambda x: int(x[5:-4]))
    # files.sort(key = lambda x: x[5:-4])
    files.sort()

    for i in range(len(files)):
        filename=pathIn+files[i]
        #reading each files
        img = cv2.imread(filename)
        # height, width, layers = img.shape
        # size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)

    # define the parameters for creating the video
    # .mp4 is a good choice for playing videos, works on OSX and Windows
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(pathOut, fourcc, fps, (frame_width,frame_height))

    # create the video from frame array
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

def main():
    # pathIn is the folder were the images to compile to video are located,
    # they must be alphabetically sorted already (ex. 000001.jpg, 000002.jpg, etc)
    pathIn= '/Users/gerrypesavento/Documents/sara/videosummary/localfiles/driveway/summary/ssim/frames/'
    # pathOut is where the compiled summary movie is to be located
    pathOut = '/Users/gerrypesavento/Documents/sara/videosummary/localfiles/driveway/summary/ssim/ssim_summary.mp4'

    # width and height of images can be automatically found using opencv, for now put in correct numbers
    # these numbers have to be correct or video will not compile
    frame_width = 1280
    frame_height = 720
    # the frame rate should be close to that of the original video, play with this number
    fps = 29.0
    # call the function to convert to video
    convert_frames_to_video(pathIn, pathOut, fps, frame_width, frame_height)

if __name__=="__main__":
    main()




