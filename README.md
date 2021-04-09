
# Video Summary Project

### Summarize video files - to extract the most useful information in a video.

### [View the Demo](https://videosummaryfiles.s3-us-west-1.amazonaws.com/index.html)

step 1. parseframes.py --> parse video into frames, store on S3<br>
step 2. getstoreboxes.py --> get object lables from frames, store metadata<br>
step 3. boundingbox.py --> add bounding boxes to frames identifying object<br>
step 4. makemovie.py --> make a summary video
