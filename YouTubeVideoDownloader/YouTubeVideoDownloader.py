import pytube
from pytube.cli import on_progress
import ffmpeg
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
video = pytube.YouTube('https://www.youtube.com/watch?v=', on_progress_callback=on_progress)

for stream in video.streams:
    #if 'video' in str(stream) or 'mp4' in str(stream):
    print(stream)
    fileSize = stream.filesize/(1024*1024)
    print(f'Stream size in MB: {fileSize}')
    bitrate = stream.bitrate/(1024*8)
    print(f'Bitrate in KB/s: {bitrate}')

stream = video.streams.get_by_itag('313')
stream.download()

#stream = video.streams.filter(file_extension='mp4').get_highest_resolution()

video_stream = ffmpeg.input('why.webm')
audio_stream = ffmpeg.input('why.mp4')
ffmpeg.output(audio_stream, video_stream, 'out.mp4').run()

