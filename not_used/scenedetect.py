
# https://github.com/Breakthrough/PySceneDetect
# openCV method
# pip3 install scenedetect[opencv]
# pip3 install ffmpeg
# pip3 install pymkv

# Standard PySceneDetect imports:
from scenedetect import VideoManager
from scenedetect import SceneManager
# For content-aware scene detection:
from scenedetect.detectors import ContentDetector

# def find_scenes(video_path, threshold=30.0):
#     # Create our video & scene managers, then add the detector.
#     video_manager = VideoManager([video_path])
#     scene_manager = SceneManager()
#     scene_manager.add_detector(
#         ContentDetector(threshold=threshold))

#     # Improve processing speed by downscaling before processing.
#     video_manager.set_downscale_factor()

#     # Start the video manager and perform the scene detection.
#     video_manager.start()
#     scene_manager.detect_scenes(frame_source=video_manager)

#     # Each returned scene is a tuple of the (start, end) timecode.
#     return scene_manager.get_scene_list()

# # the video to parse
# thevideo = '/Users/gerrypesavento/Documents/sara/videosummary/localfiles/ucdavis/video/ucdavis.mp4'
# scenes = find_scenes(thevideo)
# print(scenes)