
# This uses Amazon Rekognition to get object labels on frame images
# Finds people, text and objects
# Uploads object recognition and bounding boxes to AWS RDS in boundingbox MySQL table

import boto3
import json
import sys
import pymysql # pip3 install PyMySQL
import os

# function to detect object labels in frame image
def detect_labels(bucket, key, max_labels=10, min_confidence=90, region="us-west-1"):
	rekognition = boto3.client("rekognition", region)
	response = rekognition.detect_labels(
		Image={
			"S3Object": {
				"Bucket": bucket,
				"Name": key,
			}
		},
		MaxLabels=max_labels,
		MinConfidence=min_confidence,
	)
	return response['Labels']

# establish a connection to AWS MySQL database
conn = pymysql.connect(
    database="videometadata",
    host="videometadata.cqcc6zskeglo.us-west-1.rds.amazonaws.com",
    user="admin",
    password="Twister123!",
    port=3306,
    )
cur = conn.cursor()

# use the ucdavis video for now
video_id=1

for i in range (1, 3857):

    theimage = 'ucdavis/frames/'+str(i)+'.jpg'

    # print out the object labels and locations
    for label in detect_labels('videosummaryfiles', theimage):
        name = label['Name']
        confidence = label['Confidence']
        for instance in label['Instances']:
            width = instance['BoundingBox']['Width']
            height = instance['BoundingBox']['Height']
            left = instance['BoundingBox']['Left']
            top = instance['BoundingBox']['Top']

            sql = "INSERT INTO boundingbox (video_id,frame_number,object_detected,confidence,bb_width,bb_height,bb_left,bb_top) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            val = (video_id, i, name, confidence, width, height, left, top)
            cur.execute(sql, val)
            conn.commit()

            print (str(i), name, confidence, width, height, left, top)




