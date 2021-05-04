# importing pyglet module
import pyglet
  
def PlayVideo(video_name):
    # width of window 
    width = 320
        
    # height of window 
    height = 180
        
    # caption i.e title of the window 
    title = "CS576 Video Player"
        
    # creating a window 
    window = pyglet.window.Window(width, height, title) 
    
    
    # video path
    vidPath = video_name
    
    # creating a media player object
    player = pyglet.media.Player()
    
    # creating a source object
    source = pyglet.media.StreamingSource()
    
    # load the media from the source
    MediaLoad = pyglet.media.load(vidPath)
    
    # add this media in the queue
    player.queue(MediaLoad)
    
    # play the video
    player.play()
    
    # on draw event
    @window.event
    def on_draw():
        
        # clea the window
        window.clear()
        
        # if player sorce exist
        # and video format exist
        if player.source and player.source.video_format:
            
            # get the texture of video and
            # make surface to display on the screen
            player.get_texture().blit(0, 0)
            
            
    # key press event     
    @window.event 
    def on_key_press(symbol, modifier): 
        
        # key "p" get press 
        if symbol == pyglet.window.key.P: 
            
            # printng the message
            print("Key : P is pressed")
            
            # pause the video
            player.pause()
            
            # printing message
            print("Video is paused")
            
            
        # key "r" get press 
        if symbol == pyglet.window.key.R: 
            
            # printng the message
            print("Key : R is pressed")
            
            # resume the video
            player.play()
            
            # printing message
            print("Video is resumed")
    
    # run the pyglet application
    pyglet.app.run()
