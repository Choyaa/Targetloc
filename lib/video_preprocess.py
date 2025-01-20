from moviepy.editor import VideoFileClip

# 视频文件路径
video_path = '/mnt/sda/feicuiwan/DJI_202305121654_002/DJI_20230512171647_0326_W.MP4'

# 截取视频的起始时刻和结束时刻（以秒为单位）
start_time = 0  # 从第10秒开始
end_time = 10    # 到第20秒结束

# 降低分辨率的目标尺寸（宽度x高度）
new_resolution = (960, 540)

# 加载视频
video = VideoFileClip(video_path)

# 截取视频片段
clip = video.subclip(start_time, end_time)

# 降低视频分辨率
clip = clip.resize(new_resolution)

# 输出文件路径
output_path = '/home/ubuntu/Documents/code/github/Target2loc/datasets/翡翠湾/output_video.mp4'

# 写入文件
clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

# 释放资源
clip.close()
video.close()