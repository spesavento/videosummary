#!/usr/bin/env python3

# 1. finds images that have a certain object,
# 2. draws bounding boxes around the objects and
# 3. stores the image frames with bounding boxes into a new folder

import cv2
import sys
import boto3
import json
import pymysql # pip3 install PyMySQL
import os
from pathlib import Path

# for now, manually input the object and video variables here
# Specify object to find, and video to use
objecttofind = 'Person'
# 3 is driveway.mp4, this is the video_id in the mysql table
videotouse = 3
imageWidth=1280
imageHeight=720

# establish a connection to AWS RDS MySQL database
# need to import the database password (on Google Doc)
conn = pymysql.connect(
    database="videometadata",
    host="videometadata.cqcc6zskeglo.us-west-1.rds.amazonaws.com",
    user="admin",
    password="<password here>",
    port=3306,
    )
cur = conn.cursor()

# get images that have the object objecttofind
# select rows that match the object detected
sql = "SELECT * FROM boundingbox WHERE object_detected = %s and video_id = %s ORDER BY frame_number ASC"
val = (objecttofind, videotouse)
cur.execute(sql, val)
result = cur.fetchall()
conn.commit()
# store the object bounding box information that was in the mysql table
for row in result:
    frame_number = row[2]
    bb_width = row[5]
    bb_height = row[6]
    bb_left = row[7]
    bb_top = row[8]

    # calculate the (x,y) coordinates of the bounding box
    # (0,0) is considered the top left corner of the image
    upperleft_x = int(bb_left*imageWidth)
    upperleft_y = int(bb_top*imageHeight)
    lowerright_x = int(upperleft_x + (bb_width * imageWidth))
    lowerright_y = int(upperleft_y + (bb_height * imageHeight))

    # using local file
    inImage='/Users/gerrypesavento/Documents/sara/videosummary/localfiles/driveway/frames/'+str(frame_number)+'.jpg'
    outImage='/Users/gerrypesavento/Documents/sara/videosummary/localfiles/driveway/summary/person/frames/'+str(frame_number)+'.jpg'

    # if a summary image already has a bounding box, that becomes the input image to add a 2nd (etc) bounding box
    my_file = Path(outImage)
    if my_file.is_file():
        inImage='/Users/gerrypesavento/Documents/sara/videosummary/localfiles/driveway/summary/person/frames/'+str(frame_number)+'.jpg'
        print ('found existing'+str(frame_number))

    imgcv = cv2.imread(inImage) # opencv numby array format

    color = (255,0,0) # blue box, BGR
    line_width = 2

    # write a bounding box over the image
    # cv2.rectangle(image, start_point, end_point, color, thickness)
    k=cv2.rectangle(imgcv, (upperleft_x, upperleft_y), (lowerright_x, lowerright_y), color, line_width)
    cv2.imwrite(outImage, k)
    print ("wrote image "+ str(frame_number))


