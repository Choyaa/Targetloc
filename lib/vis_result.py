import PIL.Image as Image

base_path = '/mnt/sda/liujiachong/Production_3/render_images'
img1_path = base_path + '/0.jpg'
img2_path = base_path + '/微信图片_20241102140815.png'
save_path = base_path + '/compare1.png'

# 打开图像
img1 = Image.open(img1_path)
img2 = Image.open(img2_path)

# 确保两张图像具有相同的大小
# 假设我们想要的输出图像大小是img1的大小
target_size = img1.size

# 调整img2的大小以匹配img1的大小
img2 = img2.resize(target_size, Image.ANTIALIAS)

# 创建一个新图像，其大小和模式与img1相同
img = Image.new(img1.mode, target_size)

# 将调整大小后的图像粘贴到新图像上
# 将第一张图片的左上角和右下角复制到新图片的对应位置
img.paste(img1.crop((0, 0, target_size[0] // 2, target_size[1] // 2)), (0, 0))
img.paste(img1.crop((target_size[0] // 2, target_size[1] // 2, target_size[0], target_size[1])), (target_size[0] // 2, target_size[1] // 2))

# 将第二张图片的右上角和左下角复制到新图片的对应位置
img.paste(img2.crop((target_size[0] // 2, 0, target_size[0], target_size[1] // 2)), (target_size[0] // 2, 0))
img.paste(img2.crop((0, target_size[1] // 2, target_size[0] // 2, target_size[1])), (0, target_size[1] // 2))

# 保存合成后的图片
path = save_path
img.save(path)