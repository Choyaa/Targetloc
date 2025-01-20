import numpy as np
import os
import glob
import pandas as pd
import pyproj
from pyproj import Transformer
from pyproj import CRS
from datetime import datetime
from PIL import Image
import argparse
import exifread
# from pyexiv2 import Image
import shutil
import pyexif
# def get_dji_json(exif_dict):
#     # 获取GPS对应的标签
#     gps_latitude_tag = "gps_latitude_tag"
#     gps_latitude_ref_tag = "GPS GPSLatitudeRef"
#     gps_longitude_tag = "GPS GPSLongitude"
#     gps_longitude_ref_tag = "GPS GPSLongitudeRef"
#     gps_altitude_tag = "GPS GPSAltitude"
#     yaw_tag = "Xmp.drone-dji.GimbalYawDegree"
#     pitch_tag = "Xmp.drone-dji.GimbalPitchDegree"
#     roll_tag = "Xmp.drone-dji.GimbalRollDegree"
#     focal_tag = "EXIF FocalLength"
#     Width_tag = "EXIF ExifImageWidth"
#     Height_tag = "EXIF ExifImageLength"

#     create_time_tag = "EXIF DateTimeOriginal"
#     if "gps_latitude_tag" in exif_dict and "gps_latitude_ref_tag" in exif_dict \
#         and "gps_longitude_tag" in exif_dict and "gps_longitude_ref_tag" in exif_dict:
    
#         # 获取GPS纬度和经度的分数值和方向值
#         gps_latitude_value = exif_dict["gps_latitude_tag"]
#         gps_latitude_ref_value = exif_dict["gps_latitude_ref_tag"]
#         gps_longitude_value = exif_dict["gps_longitude_tag"]
#         gps_longitude_ref_value = exif_dict["gps_longitude_ref_tag"]
#         gps_altitude_value = exif_dict["gps_altitude_tag"]
#         # 将GPS纬度和经度的分数值转换为浮点数值
#         gps_latitude = (float(gps_latitude_value[0].num) / float(gps_latitude_value[0].den) +
#                         (float(gps_latitude_value[1].num) / float(gps_latitude_value[1].den)) / 60.0 +
#                         (float(gps_latitude_value[2].num) / float(gps_latitude_value[2].den)) / 3600.0)
#         gps_longitude = (float(gps_longitude_value[0].num) / float(gps_longitude_value[0].den) +
#                             (float(gps_longitude_value[1].num) / float(gps_longitude_value[1].den)) / 60.0 +
#                             (float(gps_longitude_value[2].num) / float(gps_longitude_value[2].den)) / 3600.0)
#         gps_altitude = eval(str(gps_altitude_value[0]).split('/')[0]) / eval(str(gps_altitude_value[0]).split('/')[1])
        
        
#         roll_value = eval(all_info_xmp[roll_tag])
#         pitch_value = eval(all_info_xmp[pitch_tag])
#         yaw_value = eval(all_info_xmp[yaw_tag])
#         # create_time
#         create_time = exif_data[create_time_tag].values
#         # intrinsic  focal_value
#         focal_value = float(exif_data["EXIF FocalLength"].values[0])
#         width = float(exif_data[Width_tag].values[0])
#         height = float(exif_data[Height_tag].values[0])
#         # 根据GPS纬度和经度的方向值，判断正负号
#         if gps_latitude_ref_value != "N":
#             gps_latitude = -gps_latitude
#         if gps_longitude_ref_value != "E":
#             gps_longitude = -gps_longitude
#         # 返回这些值
#         return roll_value, pitch_value, yaw_value, gps_latitude, gps_longitude, gps_altitude
        
#         # return roll_value, pitch_value, yaw_value, gps_latitude, gps_longitude, gps_altitude, focal_value, width, height, img_type, create_time
#     else:
#         # 如果不存在这些标签，返回None
#         return None
def os_mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
# def get_dji_exif(exif_file, return_focal_lens = False):
#     img = pyexif.ExifEditor(exif_file)
#     img.getDictTags()
    
    
def get_dji_exif(exif_file, return_focal_lens = False):
    # 打开exif文件
    with open(exif_file, "rb") as f:
        # 读取exif信息
        exif_data = exifread.process_file(f)
        img = pyexif.ExifEditor(exif_file)
        all_info_xmp = img.getDictTags()
        # img = Image(exif_file)
        # all_info_xmp = img.read_xmp()
        
        # 获取GPS对应的标签
        gps_latitude_tag = "GPS GPSLatitude"
        gps_latitude_ref_tag = "GPS GPSLatitudeRef"
        gps_longitude_tag = "GPS GPSLongitude"
        gps_longitude_ref_tag = "GPS GPSLongitudeRef"
        gps_altitude_tag = "GPS GPSAltitude"
        # yaw_tag = "Xmp.drone-dji.GimbalYawDegree"
        # pitch_tag = "Xmp.drone-dji.GimbalPitchDegree"
        # roll_tag = "Xmp.drone-dji.GimbalRollDegree"
        yaw_tag = "GimbalYawDegree"
        pitch_tag = "GimbalPitchDegree"
        roll_tag = "GimbalRollDegree"
        focal_tag = "EXIF FocalLength"
        Width_tag = "EXIF ExifImageWidth"
        Height_tag = "EXIF ExifImageLength"
        img_type = exif_file.split('/')[-1][-5]
        create_time_tag = "EXIF DateTimeOriginal"
        if gps_latitude_tag in exif_data and gps_latitude_ref_tag in exif_data and gps_longitude_tag in exif_data and gps_longitude_ref_tag in exif_data:
        
            # 获取GPS纬度和经度的分数值和方向值
            gps_latitude_value = exif_data[gps_latitude_tag].values
            gps_latitude_ref_value = exif_data[gps_latitude_ref_tag].values
            gps_longitude_value = exif_data[gps_longitude_tag].values
            gps_longitude_ref_value = exif_data[gps_longitude_ref_tag].values
            gps_altitude_value = exif_data[gps_altitude_tag].values
            # 将GPS纬度和经度的分数值转换为浮点数值
            gps_latitude = (float(gps_latitude_value[0].num) / float(gps_latitude_value[0].den) +
                            (float(gps_latitude_value[1].num) / float(gps_latitude_value[1].den)) / 60.0 +
                            (float(gps_latitude_value[2].num) / float(gps_latitude_value[2].den)) / 3600.0)
            gps_longitude = (float(gps_longitude_value[0].num) / float(gps_longitude_value[0].den) +
                             (float(gps_longitude_value[1].num) / float(gps_longitude_value[1].den)) / 60.0 +
                             (float(gps_longitude_value[2].num) / float(gps_longitude_value[2].den)) / 3600.0)
            gps_altitude = eval(str(gps_altitude_value[0]).split('/')[0]) / eval(str(gps_altitude_value[0]).split('/')[1])
            
            # print(all_info_xmp[roll_tag])
            # pitch_value = eval(all_info_xmp[pitch_tag])
            
            roll_value = float(all_info_xmp[roll_tag])
            pitch_value = all_info_xmp[pitch_tag]
            yaw_value = float(all_info_xmp[yaw_tag])
            # create_time
            create_time = exif_data[create_time_tag].values
            # intrinsic  focal_value
            
            width = float(exif_data[Width_tag].values[0])
            height = float(exif_data[Height_tag].values[0])
            # 根据GPS纬度和经度的方向值，判断正负号
            if gps_latitude_ref_value != "N":
                gps_latitude = -gps_latitude
            if gps_longitude_ref_value != "E":
                gps_longitude = -gps_longitude
                
            if return_focal_lens:
                focal_value = float(exif_data[focal_tag].values[0])
                return roll_value, pitch_value, yaw_value, gps_latitude, gps_longitude, gps_altitude, focal_value
            # 返回这些值
            return roll_value, pitch_value, yaw_value, gps_latitude, gps_longitude, gps_altitude
            
            # return roll_value, pitch_value, yaw_value, gps_latitude, gps_longitude, gps_altitude, focal_value, width, height, img_type, create_time
        else:
            # 如果不存在这些标签，返回None
            return None


def read_exif_data(folder_path):
    """
    read EXIF information of each raw image and return as a Dict.

    """
    exif_dict = {}
    img_list = glob.glob(folder_path + "/*.JPG")
    img_list = np.sort(img_list)
    for filename in img_list:
        # if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".JPG"):
        if "DJI_20230512161418_0051_W" in filename:
            print("1")
        # 打开图像文件并获取exif信息
        image_path = os.path.join(folder_path, filename)
        result = get_dji_exif(image_path)     

        # 保存exif信息到字典中
        if result:
            exif_dict[filename] = result
    return exif_dict



def main(image_path):


    exif_dict = read_exif_data(image_path)

   
    print('---------------read exif file end---------------------')

            
       


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="write EXIF information (name qw qx qy qz x y z) in txt file")
    parser.add_argument("--input_EXIF_photo", default="/mnt/sda/CityofStars/Queries/W/Seq1/image")
    parser.add_argument("--txt_pose", default="/mnt/sda/CityofStars/Queries/W/Seq1/")


    args = parser.parse_args()
    main(args.input_EXIF_photo)