
# Takes a video already stored on S3, parses it into frames,
# and store those frames in S3 in frames/ folder.
# This script takes several minutes (thousands of frames uploaded to S3)

#load python libraries
import cv2  # opencv
import numpy as np
import os, os.path
import sys
import boto3 # for AWS S3

# name of the video to parse - for now hard-coded
videotoparse = 'ucdavis'

# the video to parse
thevideo = str('https://videosummaryfiles.s3-us-west-1.amazonaws.com/ucdavis/full/'+videotoparse+'.mp4')

# use OpenCV to analyze the video and get the number of frames,
# frame_rate, time duration, width and height

# read the video into opencv as cvvideo
cvvideo = cv2.VideoCapture(thevideo)

# video number of frames
num_frames = int(cvvideo.get(cv2.CAP_PROP_FRAME_COUNT))

# video frame rate and time length in seconds
fps = int(cvvideo.get(cv2.CAP_PROP_FPS))
video_time = num_frames/fps

# video pixel width and height
pixelwidth = int(cvvideo.get(cv2.CAP_PROP_FRAME_WIDTH))
pixelheight = int(cvvideo.get(cv2.CAP_PROP_FRAME_HEIGHT))

# output information about the video
print ("\nanalyzing video ... ")
print ('video: ',num_frames,'frames,',fps,'fps,',int(video_time),'seconds,', pixelwidth,'width x',pixelheight,'height')

cvvideo.release()
cv2.destroyAllWindows()

# create a local folder localfiles/<videoname>/frames/ just outside the git repo folder
# this local folder will be used to get the parsed images on your
# local machine. This parses video into individual frames 1.jpg, 2.jpg, etc,
# cap is the video opened in opencv
cap = cv2.VideoCapture(thevideo)
i = 0
while(cap.isOpened()):
    ret,frame = cap.read()
    if ret==True:
        # temporarily store the frame in a local folder ../temp
        # there is probably a way to directly upload it to S3 ?
        cv2.imwrite('../localfiles/'+videotoparse+'/frames/'+str(i)+'.jpg',frame)
        print('stored locally '+stri(i))
        i = i + 1
    else:
        print('can not open video')
        break
# end the opencv session
cap.release()
cv2.destroyAllWindows()
print (i+1, ' image frames extracted in localfiles')

# NEVER upload the key and sec_key to GitHub
key="<hide>"
sec_key= "<hide>"
region_name = "us-west-1"

# create an s3 sessioin using boto3, allows access to S3

# s3_client = boto3.client('s3', aws_access_key_id=key, aws_secret_access_key=sec_key, region_name=region_name)

# upload the frames to S3

# cap = cv2.VideoCapture(thevideo)
# x=1
# while x < num_frames:
#     # upload the image frame to S3
#     s3_client.upload_file('../temp_frames/'+str(x)+'.jpg', 'videosummaryfiles', videotoparse+'/frames/'+str(x)+'.jpg')
#     print ('uploaded to S3 '+str(x))
#     # delete the image frame from local folder
#     os.remove('../temp_frames/'+str(x)+'.jpg')
#     print ('deleted file locally '+str(x))
#     x = x+1
# # end of openCV session
# cap.release()
# cv2.destroyAllWindows()
# print ('completed frame uploads to S3')


