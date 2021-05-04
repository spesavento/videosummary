#!/usr/bin/env python

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
from array import *
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage import io
from sklearn import preprocessing
import videoplayer as vp

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures, MidTermFeatures
import matplotlib.pyplot as plt

from moviepy.editor import *
import pygame

def FrameSimilarity(frames_jpg_path):
    # calculates the "structured similarity index" between adjacent frames
    # ssim() looks at luminance, contrast and structure, it is a scikit-image function
    # we use ssim() for both (1) Shot Change detection, and (2) Action weight
    files = [f for f in os.listdir(frames_jpg_path) if isfile(join(frames_jpg_path,f))]
    files.sort()
    # initialize array
    ssi_array = []
    # number of adjacent frames
    numadj = len(files)-2
    # loop through all adjacent frames and calculate the ssi
    for i in range (0, numadj):
    # for i in range (0, 3000):
        frame_a = cv2.imread(frames_jpg_path+'frame'+str(i)+'.jpg')
        frame_b = cv2.imread(frames_jpg_path+'frame'+str(i+1)+'.jpg')
        # crop frame images to center-weight them
        crop_img_a = frame_a[20:160, 50:270] #y1:y2 x1:x2 orginal is 320 w x 180 h
        crop_img_b = frame_b[20:160, 50:270]
        frame_a_bw = cv2.cvtColor(crop_img_a, cv2.COLOR_BGR2GRAY)
        frame_b_bw = cv2.cvtColor(crop_img_b, cv2.COLOR_BGR2GRAY)
        ssim_ab = ssim(frame_a_bw, frame_b_bw)
        ssim_ab = round(ssim_ab, 3)
        ssi_array.append(ssim_ab)
    return (ssi_array)

def FrameChange(ssi_array):
    # this function finds the frames at the shot boundary
    # length of ssi_array, how many adjacent frames
    num = len(ssi_array)
    # initialize the shot_array variable
    framechange_array = [0]
    last_hit = 0
    for i in range (0, num-3):
        ssim_ab = ssi_array[i]
        ssim_bc = ssi_array[i+1]
        ssim_cd = ssi_array[i+2]
        # 0.6 is chosen because a 60% change in similarity works well for a shot change threshold
        if (ssim_bc/ssim_ab < 0.6 and ssim_bc/ssim_cd < 0.6 and i-last_hit > 22):
            framechange_array.append(i+2)
            last_hit = i+2
    # add the last frame to the array to the end if last frame is more than last shot change
    if num-1 > framechange_array[-1] + 4:
        framechange_array.append(num-1)

    return (framechange_array)

def ShotArray(framechange_array):
    # from where the frames change, create an array of the video shots
    shot_array = []
    shot_begin = 0
    shot_end = 0
    for x in range (0, len(framechange_array)-1):
        shot_begin = framechange_array[x]
        shot_end = framechange_array[x+1]-1
        shot_array.append([shot_begin,shot_end])
    return(shot_array)

def FindAction(framechange_array, ssi_array):
    # initialize action array
    action_array = []
    for x in range (0, len(framechange_array)-1):
        frames_in_shot = framechange_array[x+1] - framechange_array[x] - 1
        ssi_total = 0
        ssi_average = 0
        for y in range (framechange_array[x], framechange_array[x+1]-1):
            ssi_total = ssi_total + ssi_array[y]
        ssi_average = ssi_total / frames_in_shot
        # instead of low is high action, make high is high action
        ssi_average = 1 - ssi_average
        action_array.append(ssi_average)
    # in the action array, a smaller value means more action (less similarity within shot frames)
    # return a normalized weighted array, value 0 to 1
    action_array_normalized = preprocessing.minmax_scale(action_array, feature_range=(0, 1))
    action_array = [round(num, 3) for num in action_array_normalized]
    return(action_array)

def FindFaces(framechange_array, frames_jpg_path):
    # Load face classifier, using "Haar" classifier, basic but works fine
    face_classifier = cv2.CascadeClassifier('haarcascade_face_classifier.xml')
    # initialize array variable to record faces
    face_array = []
    # loop through the number of shots
    for x in range (0, len(framechange_array)-1):
        frames_in_shot = framechange_array[x+1] - framechange_array[x] - 1
        face_total = 0
        for y in range (framechange_array[x], framechange_array[x+1]-1, 5):
            # url of frame image to analyze
            filename=frames_jpg_path+'frame'+str(y)+'.jpg'
            # read it into OpenCV
            img = cv2.imread(filename)
            # convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # detect faces
            # scaleFactor - how much the image size is reduced at each image scale
            # minNeighbors = 4 gives few false positives, but misses a few faces
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for (x, y, w, h) in faces:
                face_total = face_total + 1
        face_array.append(face_total)
    # return a normalized weighted array, value 0 to 1
    face_array_normalized = preprocessing.minmax_scale(face_array, feature_range=(0, 1))
    face_array = [round(num, 3) for num in face_array_normalized]
    return(face_array)

def FindPeople(framechange_array, frames_jpg_path):
    # OpenCV has a pre-trained person model using Histogram Oriented Gradients (HOG)
    # and Linear SVM
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # initialize array variable to record faces
    people_array = []
    for x in range (0, len(framechange_array)-1):
        frames_in_shot = framechange_array[x+1] - framechange_array[x] - 1
        people_total = 0
        for y in range (framechange_array[x], framechange_array[x+1]-1, 5):
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
    people_array = [round(num, 3) for num in people_array_normalized]
    return(people_array)

def FindAudioShots(framechange_array, audio_path):
    feature = 1
    [Fs, x] = audioBasicIO.read_audio_file(audio_path)
    x = audioBasicIO.stereo_to_mono(x)
    frame_size = (Fs // 30)
    F, f_names = ShortTermFeatures.feature_extraction(x, Fs, frame_size, frame_size, deltas=False)
    # plt.subplot(2,1,1); plt.plot(F[3,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[3]) 
    # plt.subplot(2,1,2); plt.plot(F[feature,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[feature]); plt.show()

    astd = (np.std(F[feature,:]))
    aave = (np.average(F[feature,:]))

    which_shots = np.zeros(len(F[feature,:])).flatten()
    print(which_shots.shape)

    for i in range(len(F[feature,:])):
        if (abs(F[feature,:][i]-aave) > astd * 4.5):
            which_shots[i] = F[feature,:][i]
    
    audioshotchange_list = []

    prev_val = 0.0
    last_start = 0
    for i in range(len(F[1,:])):
        # print(which_shots[i])
        if (prev_val == 0.0 and which_shots[i] > 0.0):
            last_start = i
        if (prev_val > 0.0 and which_shots[i] == 0.0):
            audioshotchange_list.append([last_start, i, which_shots[last_start]])

        prev_val = which_shots[i]
    
    audio_array = np.zeros(len(framechange_array)-1)

    for x in range (0, len(framechange_array)-1):
        first_frame = framechange_array[x]
        last_frame = framechange_array[x+1]
        for y in range(len(audioshotchange_list)):
            if audioshotchange_list[y][0] >= first_frame and audioshotchange_list[y][0] < last_frame:
                audio_array[x] += audioshotchange_list[y][2]
        audio_array[x] /= (last_frame - first_frame)

    audio_array = preprocessing.minmax_scale(audio_array, feature_range=(0, 1))
    audio_array = [round(num, 3) for num in audio_array]
    return(audio_array)

def TotalWeights(shot_array, action_array, face_array, people_array, audio_array):
    # use numpy to add the weight arrays
    # for now a simple addition of action, face, people weights
    face_array_scaled = [element * 0.5 for element in face_array]
    people_array_scaled = [element * 0.5 for element in people_array]
    audio_array_scaled = [element * 0.5 for element in audio_array]
    arr = []
    arr.append(action_array)
    arr.append(face_array_scaled)
    arr.append(people_array_scaled)
    arr.append(audio_array_scaled)
    np_arr = np.array(arr)
    np_weight = np_arr.sum(axis=0)
    total_weight = list(np.around(np.array(np_weight),3))
    # total_weight = np_weight.tolist()
    for x in range (0, len(shot_array)):
        shot_array[x].append(total_weight[x])
    totalweight_array = shot_array
    # returns a multi-level weighted array [shot start, shot end, total weight]
    return(totalweight_array)

def SaveSummaryFrames(totalweight_array, summary_frame_path, frames_jpg_path):
    # with weighted shots, save the summary frames into summary_frame_path
    # sort the array by weight descending, best shots first
    sorted_array = sorted(totalweight_array, key=lambda x: x[2], reverse=True)
    print('\nsorted_array')
    print('shots ordered by highest weight first')
    print(str(sorted_array))
    frame_count = 0
    summary_array = []
    ordered_array = []
    # first truncated the shots that won't be used
    # do this by counting the top weighted shots until
    # frame count is < 2700 (90 seconds x 30 fps)
    for x in range (0, len(sorted_array)-1):
        start_frame = sorted_array[x][0]
        end_frame = sorted_array[x][1]
        num_frames = end_frame - start_frame
        frame_count = frame_count + num_frames
        # stop if frame_count is 90 sec (90 sec * 30 fps = 2700)
        if (frame_count < 2700):
            summary_array.insert(x, sorted_array[x])
    # ordered array sort by shot start frame number
    ordered_array = sorted(summary_array, key=lambda x: x[0])
    print('\nordered_array')
    print('shots trimmed down to < 2700 frames, ordered by scene number')
    print(str(ordered_array))
    num_shots=len(ordered_array)
    # create a numeric list 0000, 0001, to 9999
    numlist = ["%04d" % x for x in range(10000)]
    count = 0
    # print(str(num_shots))
    for y in range (0,num_shots):
        start = ordered_array[y][0]
        end = ordered_array[y][1]
        # print(str(start))
        for z in range (start, end):
            shot_image = frames_jpg_path+'frame'+str(z)+'.jpg'
            img = cv2.imread(shot_image)
            summary_image = summary_frame_path+str(z)+'.jpg'
            # add shot number to frame
            cv2.putText(
                img, #numpy image
                str(y), #text
                (10,60), #position
                cv2.FONT_HERSHEY_SIMPLEX, #font
                2, #font size
                (0, 0, 255), #font color red
                4) #font stroke
            cv2.imwrite(summary_image,img)
            count = count+1

# Convert frames folder to video using OpenCV
def FramesToVideo(summary_frame_path,pathOut,fps,frame_width,frame_height,audio_path,new_audio_path):
    frame_array = []
    audio_frames = []

    audio_object = wave.open(audio_path, 'r')
    framerate = audio_object.getframerate()

    files = [f for f in os.listdir(summary_frame_path) if isfile(join(summary_frame_path,f))]
    # sort the files
    # see python reference https://docs.python.org/3/howto/sorting.html
    files.sort()
    for i in range(len(files)):
        filename=summary_frame_path+files[i]
        FrameNum = int(os.path.splitext(files[i])[0])
        # Convert to audio frame
        # print(files[i])
        # print(FrameNum)
        AudioFrameNum = ((FrameNum * framerate) // 30)
        # print(AudioFrameNum)
        NumFramesToRead = (framerate // 30)
        # print(NumFramesToRead)

        audio_object.setpos(AudioFrameNum)
        NewAudioFrames = audio_object.readframes(NumFramesToRead)
        audio_frames.append(NewAudioFrames)

        #reading each files
        img = cv2.imread(filename)
        # height, width, layers = img.shape
        # size = (width,height)
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

    # Write new audio file
    sampleRate = framerate # hertz
    duration = len(audio_frames) / framerate # seconds
    obj = wave.open(new_audio_path,'w')
    obj.setnchannels(2) # mono
    obj.setsampwidth(2)
    obj.setframerate(sampleRate)

    for i in range(len(audio_frames)):
        obj.writeframesraw(audio_frames[i])
    obj.close()

def MakeCollage(framechange_array, frames_jpg_path, collage_path):
    # creates a collage of the shots in a video, the collage shows shot # and frame #
    # imporant - the top.jpg must be in the folder path, and it has to be exact width of 2240px
    # take the frame one forward of the shot change
    offset = 1
    i = 0
    # start with a blank image that is the same width (2240px) of 7 frames
    im_v = cv2.imread('top.jpg')
    # make a collage that is 7 frames wide
    for x in range (0, len(framechange_array)-7, 7):
        im_a = cv2.imread(frames_jpg_path+'frame'+str(framechange_array[x]+offset)+'.jpg')
        im_b = cv2.imread(frames_jpg_path+'frame'+str(framechange_array[x+1]+offset)+'.jpg')
        im_c = cv2.imread(frames_jpg_path+'frame'+str(framechange_array[x+2]+offset)+'.jpg')
        im_d = cv2.imread(frames_jpg_path+'frame'+str(framechange_array[x+3]+offset)+'.jpg')
        im_e = cv2.imread(frames_jpg_path+'frame'+str(framechange_array[x+4]+offset)+'.jpg')
        im_f = cv2.imread(frames_jpg_path+'frame'+str(framechange_array[x+5]+offset)+'.jpg')
        im_g = cv2.imread(frames_jpg_path+'frame'+str(framechange_array[x+6]+offset)+'.jpg')
        # add the shot numbers to the collage images
        cv2.putText(im_a, str(x), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.putText(im_b, str(x+1), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.putText(im_c, str(x+2), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.putText(im_d, str(x+3), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.putText(im_e, str(x+4), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.putText(im_f, str(x+5), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.putText(im_g, str(x+6), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        # add the frame numbers to the collage images
        cv2.putText(im_a, str(framechange_array[x]), (120,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
        cv2.putText(im_b, str(framechange_array[x+1]), (120,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
        cv2.putText(im_c, str(framechange_array[x+2]), (120,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
        cv2.putText(im_d, str(framechange_array[x+3]), (120,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
        cv2.putText(im_e, str(framechange_array[x+4]), (120,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
        cv2.putText(im_f, str(framechange_array[x+5]), (120,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
        cv2.putText(im_g, str(framechange_array[x+6]), (120,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
        # build the collage
        im_h = cv2.hconcat([im_a, im_b, im_c, im_d, im_e, im_f, im_g])
        im_v = cv2.vconcat([im_v, im_h])
    cv2.imwrite(collage_path, im_v)

def SyncVideoWithAudio(old_video_name, video_name, audio_path):

    my_clip = mpe.VideoFileClip(old_video_name)
    audio_background = mpe.AudioFileClip(audio_path)
    final_clip = my_clip.set_audio(audio_background)
    final_clip.write_videofile(video_name,fps=30)

    my_clip.close()
    final_clip.close()
    audio_background.close()

def main():

    # name of the video to process
    video_name = 'soccer'

    # jpg video frames to be analyzed - ordered frame0.jpg, frame1.jpg, etc.
    frames_jpg_path = '../project_files/project_dataset/frames/'+video_name+'/'

    # audio to process
    audio_path = '../project_files/project_dataset/audio/'+video_name+'.wav'

    # new audio path
    new_audio_path = "../project_files/summary/" +video_name+ "/sound.wav"

    # directory for summary frames and summary video
    summary_frame_path = '../project_files/summary/'+video_name+'/frames/'
    summary_video_path = '../project_files/summary/'+video_name+'/summary.mp4'
    summary_video_audio_path = '../project_files/summary/'+video_name+'/summary_with_audio.mp4'
    collage_path = '../project_files/summary/'+video_name+'/collage.jpg'

    # empty the summary folders and summary results
    print ('\nremoving all previous summary files in summary/shot folders')
    filesToRemove = [os.path.join(summary_frame_path,f) for f in os.listdir(summary_frame_path)]
    for f in filesToRemove:
        os.remove(f)
    if os.path.exists(summary_video_path):
        os.remove(summary_video_path)
    if os.path.exists(collage_path):
        os.remove(collage_path)

    # get ssi_array, the structured similarity between adjacent frames
    print ('\nssi_array')
    print ('the similarity between adjacent frames ... takes a long minute')
    ssi_array = FrameSimilarity(frames_jpg_path)
    print(str(ssi_array[0 : 50])+' ... more')

    # get the framechange_array, which are the shot boundary frames
    print ('\nframechange_array')
    print ('these are the frames where the shot changed')
    framechange_array = FrameChange(ssi_array)
    print(str(framechange_array))

    # get the shot_array, showing the shot sequences start, end
    print ('\nshot_array')
    shot_array = ShotArray(framechange_array)
    print (str(len(shot_array))+' shots in the video')
    print(str(shot_array))

    # get the audio array
    print('\naudio_array')
    audio_array = FindAudioShots(framechange_array, audio_path)
    print('there are '+str(len(audio_array))+' audio weights')
    print(str(audio_array))

    # get action_array, shows the average action weight for each shot
    print ('\naction_array')
    action_array = FindAction(framechange_array, ssi_array)
    print(str(len(action_array))+' action weights')
    print(str(action_array))

    # get the face array
    print('\nface_array')
    face_array = FindFaces(framechange_array, frames_jpg_path)
    print(str(len(face_array))+' face weights')
    print(str(face_array))

    # get the people array
    print('\npeople_array')
    people_array = FindPeople(framechange_array, frames_jpg_path)
    print('there are '+str(len(people_array))+' people weights')
    print(str(people_array))

    # total the weights
    print('\ntotalweight_array')
    print('[shot start, shot end, total weight]')
    totalweight_array = TotalWeights(shot_array, action_array, face_array, people_array, audio_array)
    print(str(totalweight_array))

    # create summary frames in a folder
    SaveSummaryFrames(totalweight_array,summary_frame_path, frames_jpg_path)

    # create summary video
    print('\nfrom the summary frames, creating a summary video')
    FramesToVideo(summary_frame_path, summary_video_path, 30, 320, 180, audio_path, new_audio_path)
    print('the summary video is stored as '+summary_video_path)

    # Adding audio to video
    SyncVideoWithAudio(summary_video_path, summary_video_audio_path, new_audio_path)

    # # optional - make a photo collage of the shots
    # print('\nbonus: photo collage of scenes saved as collage.jpg in the root folder')
    # MakeCollage(framechange_array, frames_jpg_path, collage_path)

    vp.PlayVideo(summary_video_audio_path)

if __name__=="__main__":
    main()
