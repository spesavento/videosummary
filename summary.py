#!/usr/bin/env python

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage import io
from sklearn import preprocessing
import numpy as np
import cv2
import numpy as np
import pickle as pl
import struct
import os
from os.path import isfile, join
from matplotlib import pyplot as plt
import moviepy.editor as mpe
import natsort
import wave
from PIL import Image
from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils


def StructuredSimilarity(frames_jpg_path):
    # calculates the "structured similarity index" between adjacent frames
    # ssim() looks at luminance, contrast and structure, it is a scikit-image function
    # we use this for both Shot Change detection, and Action weight
    # create array of file images and sort them
    files = [f for f in os.listdir(frames_jpg_path) if isfile(join(frames_jpg_path,f))]
    files.sort()
    # initialize array
    ssi_array = []
    # number of adjacent frames
    adjframes = len(files)-1
    # loop through all adjacent frames and calculate the ssi
    for i in range (0, adjframes):
        frame_a = cv2.imread(frames_jpg_path+'frame'+str(i)+'.jpg')
        frame_b = cv2.imread(frames_jpg_path+'frame'+str(i+1)+'.jpg')
        frame_a_bw = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        frame_b_bw = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
        ssim_ab = ssim(frame_a_bw, frame_b_bw)
        ssim_ab = round(ssim_ab, 4)
        ssi_array.append(ssim_ab)
    return (ssi_array)

def ShotChange(ssi_array):
    # length of ssi_array, how many adjacent frames
    num = len(ssi_array)
    # initialize the shot_array variable
    shotchange_array = [0]
    last_hit = 0
    for i in range (0, num-3):
        ssim_ab = ssi_array[i]
        ssim_bc = ssi_array[i+1]
        ssim_cd = ssi_array[i+2]
        # 0.6 is chosen because a 60% change in similarity works well for a shot change threshold
        if (ssim_bc/ssim_ab < 0.6 and ssim_bc/ssim_cd < 0.6 and i-last_hit > 20):
            shotchange_array.append(i+2)
            last_hit = i+2
    return (shotchange_array)

def FindAction(shotchange_array, ssi_array):
    # initialize action array
    action_array = []
    for x in range (0, len(shotchange_array)-1):
        frames_in_shot = shotchange_array[x+1] - shotchange_array[x] - 1
        ssi_total = 0
        ssi_average = 0
        for y in range (shotchange_array[x], shotchange_array[x+1]-1):
            ssi_total = ssi_total + ssi_array[y]
        ssi_average = ssi_total / frames_in_shot
        # instead of low is high action, make high is high action
        ssi_average = 1 - ssi_average
        action_array.append(ssi_average)
    # in the action array, a smaller value means more action (less similarity within shot frames)
    # return a normalized weighted array, value 0 to 1
    action_array_normalized = preprocessing.minmax_scale(action_array, feature_range=(0, 1))
    action_array = [round(num, 4) for num in action_array_normalized]
    return(action_array)

def FindFaces(shotchange_array, frames_jpg_path):
    # Load face classifier, using "Haar" classifier, basic but works fine
    face_classifier = cv2.CascadeClassifier('haarcascade_face_classifier.xml')
    # initialize array variable to record faces
    face_array = []
    # loop through the number of shots
    for x in range (0, len(shotchange_array)-1):
        frames_in_shot = shotchange_array[x+1] - shotchange_array[x] - 1
        face_total = 0
        for y in range (shotchange_array[x], shotchange_array[x+1]-1):
            # url of frame image to analyze
            filename=frames_jpg_path+'frame'+str(y)+'.jpg'
            # read it into OpenCV
            img = cv2.imread(filename)
            # convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # detect faces
            # scaleFactor – how much the image size is reduced at each image scale
            # minNeighbors = 4 gives few false positives, but misses a few faces
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for (x, y, w, h) in faces:
                face_total = face_total + 1
        face_array.append(face_total)
    # return a normalized weighted array, value 0 to 1
    face_array_normalized = preprocessing.minmax_scale(face_array, feature_range=(0, 1))
    face_array = [round(num, 4) for num in face_array_normalized]
    return(face_array)

def FindPeople(shotchange_array, frames_jpg_path):
    # OpenCV has a pre-trained person model using Histogram Oriented Gradients (HOG)
    # and Linear SVM
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # initialize array variable to record faces
    people_array = []
    for x in range (0, len(shotchange_array)-1):
        frames_in_shot = shotchange_array[x+1] - shotchange_array[x] - 1
        people_total = 0
        for y in range (shotchange_array[x], shotchange_array[x+1]-1):
            # url of frame image to analyze
            filename=frames_jpg_path+'frame'+str(y)+'.jpg'
            # read it into OpenCV
            image = cv2.imread(filename)
            # resize the image to increase speed (may try this on face detect as well)
            image = imutils.resize(image, width=min(400, image.shape[1]))
            orig = image.copy()
            # detect people in the image
            (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
            for (x, y, w, h) in rects:
                people_total = people_total + 1
        people_array.append(people_total)
    # return a normalized weighted array, value 0 to 1
    people_array_normalized = preprocessing.minmax_scale(people_array, feature_range=(0, 1))
    people_array = [round(num, 4) for num in people_array_normalized]
    return(people_array)


# Take each Shot section and reduce it 1:6
# This is too simple, just taking every 6th frame, but identifying the shots
def ShowShotChange(frames_jpg_path,summary_frame_path, shot_change):
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
            summary_image = frames_jpg_path+'frame'+str(save_frame)+'.jpg'
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



def main():

    # put in the directory of the frames and where the summary video will go

    # directory of full video frames - ordered frame1.jpg, frame2.jpg, etc.
    frames_jpg_path = "../project_files/project_dataset/frames/soccer/"

    # directory for summary frames
    summary_frame_path = "../project_files/summary/soccer/frames/"

    # directory for summary video
    summary_video_path = '../project_files/summary/meridian/video/soccer.mp4'

    # start processing

    # get ssi_array, the structured similarity between adjacent frames
    print ('\nssi_array')
    print ('the similarity between adjacent frames ... takes a minute')
    ssi_array = StructuredSimilarity(frames_jpg_path)
    print(str(ssi_array[0 : 50])+' ... long')

    # get the shotchange_array, which are the shot boundary frames
    print ('\nshotchange_array')
    print ('the frames where shots change')
    shotchange_array = ShotChange(ssi_array)
    print(str(shotchange_array))

    # get action_array, shows the average action weight for each shot
    print ('\naction_array')
    action_array = FindAction(shotchange_array, ssi_array)
    print(str(action_array))

    # get the face array
    print('\nface_array')
    face_array = FindFaces(shotchange_array, frames_jpg_path)
    print(str(face_array))

    # get the people array
    print('\npeople_array')
    people_array = FindPeople(shotchange_array, frames_jpg_path)
    print(str(people_array))


    # make summary frame folder
    # ShowShotChange(frames_jpg_path,summary_frame_path,shotchange_array)

    # make a video from the summary frame folder
    # FramesToVideo(summary_frame_path, summary_video_path, 30, 320, 180)


if __name__=="__main__":
    main()

