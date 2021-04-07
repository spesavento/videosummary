
# This uses Amazon Rekognition to get object labels on frame images
# Finds people, text and objects in a local image KEY

import boto3
import json
import sys

BUCKET = "videosummaryfiles"
KEY = "ucdavis/frames/371.jpg"

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

# print out the object labels and locations
for label in detect_labels(BUCKET, KEY):
	name = label['Name']
	confidence = label['Confidence']
	for instance in label['Instances']:
		width = instance['BoundingBox']['Width']
		height = instance['BoundingBox']['Height']
		left = instance['BoundingBox']['Left']
		top = instance['BoundingBox']['Top']
		print (name, confidence, width, height, left, top)
