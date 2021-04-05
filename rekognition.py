# Amazon Rekognition to get object labels on frame images
# Find people, text and objects

import boto3
import json
import sys

BUCKET = "videosummaryfiles"
KEY = "ucdavis/frames/1.jpg"

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

for label in detect_labels(BUCKET, KEY):
	print (label)