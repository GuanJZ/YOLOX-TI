import os
import subprocess

def convert_images(input_dir, output_dir):
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录下的所有jpg文件
    filenames = os.listdir(input_dir)
    image_id = 1
    for filename in filenames:
        if filename.endswith(".jpg"):
            # 构建输入和输出文件的完整路径
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace("jpg", "yuv"))

            # 使用ffmpeg将jpg图像转换为nv12 yuv图像
            subprocess.run(['ffmpeg', '-y', '-i', input_path, '-pix_fmt', 'nv12', output_path])
            image_id += 1

# 在命令行中接收输入和输出目录的参数
input_dir = "datasets/Rope3D/val"
output_dir = "datasets/Rope3D/yuv/val"

# 调用函数进行转换
convert_images(input_dir, output_dir)