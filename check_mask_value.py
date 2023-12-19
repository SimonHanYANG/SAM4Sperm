from PIL import Image
import numpy as np

# 替换为你的图片文件路径
image_path = 'combined_mask.png'

# 读取图片
image = Image.open(image_path)

# 如果是RGBA，转换为RGB
if image.mode == 'RGBA':
    image = image.convert("RGB")

# 转换图片为numpy数组
image_data = np.array(image)

# 设置非黑色的像素为白色
# 我们假设黑色的像素值为(0, 0, 0)
white = [255, 255, 255]
black = [0, 0, 0]
mask = np.all(image_data == black, axis=-1)
# 用三个通道分别设置颜色
image_data[~mask] = white

# 创建一个新的Pillow图像
new_image = Image.fromarray(image_data)

# 检查像素值是否为0或255
unique_values = np.unique(image_data)
is_binary = np.all(np.isin(unique_values, [0, 255]))

# 打印结果
if is_binary:
    print("The pixel value of the image contains only 0 or 255.")
else:
    non_binary_values = unique_values[~np.isin(unique_values, [0, 255])]
    print("The image contains pixel values OTHER than 0 and 255: {}.".format(non_binary_values))
    

# 可选择保存新的图片
new_image.save('combined_mask_0255.png')