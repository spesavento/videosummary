#!/usr/bin/env python

# drawing bounding box around objects

import cv2
import sys
import boto3
import json
import pymysql # pip3 install PyMySQL
import os
from pathlib import Path

# establish a connection to AWS MySQL database
conn = pymysql.connect(
    database="videometadata",
    host="videometadata.cqcc6zskeglo.us-west-1.rds.amazonaws.com",
    user="admin",
    password="Twister123!",
    port=3306,
    )
cur = conn.cursor()

objecttofind = 'Bicycle'
videotouse = 1

sql = "SELECT * FROM boundingbox WHERE object_detected = %s and video_id = %s ORDER BY frame_number ASC"
val = (objecttofind, videotouse)
# cursor.execute(sql, ('webmaster@python.org',))
cur.execute(sql, val)
result = cur.fetchall()
conn.commit()

for row in result:
    frame_number = row[2]
    bb_width = row[5]
    bb_height = row[6]
    bb_left = row[7]
    bb_top = row[8]

    imageWidth=960
    imageHeight=540

    # calculate the (x,y) coordinates of the bounding box
    # (0,0) is considered the top left corner of the image
    upperleft_x = int(bb_left*imageWidth)
    upperleft_y = int(bb_top*imageHeight)
    lowerright_x = int(upperleft_x + (bb_width * imageWidth))
    lowerright_y = int(upperleft_y + (bb_height * imageHeight))

    inImage='/Users/gerrypesavento/Documents/sara/videosummary/localfiles/ucdavis/frames/'+str(frame_number)+'.jpg'
    outImage='/Users/gerrypesavento/Documents/sara/videosummary/localfiles/ucdavis/summary/bicycle/frames/'+str(frame_number)+'.jpg'

    my_file = Path(outImage)
    if my_file.is_file():
        inImage='/Users/gerrypesavento/Documents/sara/videosummary/localfiles/ucdavis/summary/bicycle/frames/'+str(frame_number)+'.jpg'
        print ('found existing'+str(frame_number))

    imgcv = cv2.imread(inImage) # opencv numby array format

    color = (0,255,0) # green box

    # write a bounding box over the image
    # cv2.rectangle(image, start_point, end_point, color, thickness)
    k=cv2.rectangle(imgcv, (upperleft_x, upperleft_y), (lowerright_x, lowerright_y),(0, 255, 0), 2)
    cv2.imwrite(outImage, k)
    print ("wrote image "+ str(frame_number))


