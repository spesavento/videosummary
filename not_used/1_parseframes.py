
# Takes a video already stored on S3, parses it into frames,
# and store those frames in S3 in frames/ folder.
# This script takes several minutes (thousands of frames uploaded to S3)

#load python libraries
import cv2  # opencv
import numpy as np
import os, os.path
import sys
import boto3 # for AWS S3

def getVideodata(videoPath):

    # read the video into opencv as cvvideo
    cvvideo = cv2.VideoCapture(videoPath)

    # video number of frames
    num_frames = int(cvvideo.get(cv2.CAP_PROP_FRAME_COUNT))
    # video frame rate and time length in seconds
    fps = int(cvvideo.get(cv2.CAP_PROP_FPS))
    video_time = num_frames/fps
    # video pixel width and height
    pixelwidth = int(cvvideo.get(cv2.CAP_PROP_FRAME_WIDTH))
    pixelheight = int(cvvideo.get(cv2.CAP_PROP_FRAME_HEIGHT))

    data = 'video: ',num_frames,'frames,',fps,'fps,',int(video_time),'seconds,', pixelwidth,'width x',pixelheight,'height'
    return data

    cvvideo.release()
    cv2.destroyAllWindows()

# create a local folder localfiles/<videoname>/frames/ just outside the git repo folder
# this local folder will be used to get the parsed images on your
# local machine. This parses video into individual frames 1.jpg, 2.jpg, etc,
cap is the video opened in opencv
cap = cv2.VideoCapture(videoPath)
i = 1
while(cap.isOpened()):
    ret,frame = cap.read()
    if ret==True:
        # temporarily store the frame in a local folder ../temp
        # there is probably a way to directly upload it to S3 ?
        cv2.imwrite('../<localpath>/frames/'+str(i)+'.jpg',frame)
        print('stored locally '+str(i))
        i = i + 1
    else:
        print('can not open video')
        break
# end the opencv session
cap.release()
cv2.destroyAllWindows()
print (i+1, ' image frames extracted in localfiles')

# # NEVER upload the key and sec_key to GitHub
key="<key here>"
sec_key= "<secret key here>"
region_name = "us-west-1"

# un-comment code below to store frames on AWS S3

# create an s3 sessioin using boto3, allows access to S3
s3_client = boto3.client('s3', aws_access_key_id=key, aws_secret_access_key=sec_key, region_name=region_name)

# upload the frames to S3
cap = cv2.VideoCapture(videoPath)
x=1
while x < num_frames:
    # upload the image frame to S3
    s3_client.upload_file('<local path here>/localfiles/driveway/frames/'+str(x)+'.jpg', 'videosummaryfiles', videotoparse+'/frames/'+str(x)+'.jpg')
    print ('uploaded to S3 '+str(x))
    x = x+1
# end of openCV session
cap.release()
cv2.destroyAllWindows()
print ('completed frame uploads to S3')


def main():
    # pathIn is the folder were the images to compile to video are located,
    # they must be alphabetically sorted already (ex. 000001.jpg, 000002.jpg, etc)
    framePath= '../project_files/project_dataset/frames/meridian/'
    # pathOut is where the compiled summary movie is to be located
    pathOut = '../project_files/summary_videos/meridian_summary.mp4'

    # width and height of images can be automatically found using opencv, for now put in correct numbers
    # these numbers have to be correct or video will not compile
    frame_width = 320
    frame_height = 180
    # the frame rate should be close to that of the original video, play with this number
    fps = 30.0
    # call the function to convert to video
    convert_frames_to_video(pathIn, pathOut, fps, frame_width, frame_height)

if __name__=="__main__":
    main()
