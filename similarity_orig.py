#!/usr/bin/env python

# python -m pip install -U scikit-image
# (also Cython, numpy, scipi, matplotlib, networkx, pillow, imageio, tifffile, PyWavelets)
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import numpy as np # may not be needed
import cv2
import glob
import numpy as np
import os
from os.path import isfile, join

# Creates an array of shot-change frames
def ShotChange(full_frame_path):
    print ('determining shot changes ...')
    # create array of file images and sort them
    files = [f for f in os.listdir(full_frame_path) if isfile(join(full_frame_path,f))]
    files.sort()
    # number of frames in folder
    num = len(files)
    # initialize the shot change array variable
    # keep the very first frame as a scene change
    shot_change = [0]
    # initialize the first 3 frames
    last_frame = 0
    # calculate the ssim on the first 3 frames
    frame_a = cv2.imread(full_frame_path+'frame0.jpg')
    frame_b = cv2.imread(full_frame_path+'frame1.jpg')
    frame_c = cv2.imread(full_frame_path+'frame2.jpg')
    frame_a_bw = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    frame_b_bw = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    frame_c_bw = cv2.cvtColor(frame_c, cv2.COLOR_BGR2GRAY)
    ssim_ab = ssim(frame_a_bw, frame_b_bw)
    ssim_bc = ssim(frame_b_bw, frame_c_bw)
    # now loop through all frames to look for "local minimums"
    for i in range(0, num-3):
        # read in four frames to opencv
        frame_d = cv2.imread(full_frame_path+'frame'+str(i+3)+'.jpg')
        # convert them to grayscale
        frame_d_bw = cv2.cvtColor(frame_d, cv2.COLOR_BGR2GRAY)
        # calculate ssim between adjacent frames
        ssim_cd = ssim(frame_c_bw, frame_d_bw)
        # we have ssim_ab, ssim_bc, ssim_cd ... we want to know if ssim_bc is a "local minimum"
        # for now 0.6 is randomly chosen, seems to work OK
        if (ssim_bc/ssim_ab < 0.6 and ssim_bc/ssim_cd < 0.6 and i-last_frame > 20):
            # print ('shot change frame '+ str(i+2))
            last_frame = i+2
            # build the shot_change array with the frame numbers where change happens
            shot_change.append(i+2)
        # slide the window
        ssim_ab = ssim_bc
        ssim_bc = ssim_cd
        frame_c_bw = frame_d_bw
    print('shot change array is:')
    print(shot_change)
    return shot_change

# Take each Shot section and reduce it 1:6
# This is too simple, just taking every 6th frame, but identifying the shots
def MakeSummaryFrames(full_frame_path,summary_frame_path, shot_change):
    print ('storing summary frames in '+summary_frame_path+'...')
    num_shots = len(shot_change)
    z = 0
    for i in range (0, num_shots-1):
        frame_a = shot_change[i]
        frame_b = shot_change[i+1]
        num_frames_in_shot = frame_b - frame_a
        num_frames_to_keep = int(num_frames_in_shot/6)
        save_frame = frame_a
        for y in range (0, num_frames_to_keep):
            # save every 6th frame
            save_frame = save_frame + 6
            summary_image = full_frame_path+'frame'+str(save_frame)+'.jpg'
            img = cv2.imread(summary_image)
            # write the shot numbers on the summary frames
            cv2.putText(
                img, #numpy image
                str(i), #text
                (10,60), #position
                cv2.FONT_HERSHEY_SIMPLEX, #font
                2, #font size
                (0, 0, 255), #font color red
                4) #font stroke
            alpha_number = str(save_frame).zfill(5)
            summary_frame_img = summary_frame_path+alpha_number+'.jpg'
            cv2.imwrite(summary_frame_img,img)
            z = z+1

# Convert frames folder to video using OpenCV
def FramesToVideo(summary_frame_path,pathOut,fps,frame_width,frame_height):
    frame_array = []
    files = [f for f in os.listdir(summary_frame_path) if isfile(join(summary_frame_path,f))]
    # sort the files
    # see python reference https://docs.python.org/3/howto/sorting.html
    files.sort()
    for i in range(len(files)):
        filename=summary_frame_path+files[i]
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

def FindFaces(img):
    # Load classifier
    face_classifier = cv2.CascadeClassifier('haarcascade_face_classifier.xml')
    # Read the input image
    img = cv2.imread(img)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_classifier.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display the output
    # Note ... can store the output as array variable (shots that have faces)
    cv2.imshow('img', img)
    cv2.waitKey()
    # to exit, while the image box is displaying, hit any key



def main():

    # directory of full video frames - ordered frame1.jpg, frame2.jpg, etc.
    full_frame_path = "../project_files/project_dataset/frames/meridian/"

    # directory for summary frames
    summary_frame_path = "../project_files/summary/meridian/frames/"

    # directory for summary video
    summary_video_path = '../project_files/summary/meridian/video/meridian.mp4'

    # get shot_change array
    # shot_change = ShotChange(full_frame_path)

    # make summary frame folder
    # MakeSummaryFrames(full_frame_path,summary_frame_path,shot_change)

    # make a video from the summary frame folder
    # FramesToVideo(summary_frame_path, summary_video_path, 30, 320, 180)

    FindFaces('/Users/gerrypesavento/Documents/sara/videosummary/project_files/project_dataset/frames/meridian/frame1089.jpg')

if __name__=="__main__":
    main()

