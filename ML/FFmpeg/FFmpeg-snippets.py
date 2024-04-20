# FFmpeg Python library

#Video processing
# ff=ffmpy.Ffmpeg(inputs= {‘video.mp4’: None}, outputs= {‘out_frames/%06d.jpg’: ‘-vf fps=29.97/2,scale=320:-1 –q:v 5’})
# ff.cmd – Display the Ffmpeg command that will run to check it for errors
# ff.run() – Run the command
# Use skimage and Matplotlib libraries to perform video frames processing as images.
