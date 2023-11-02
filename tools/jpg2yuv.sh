#!/bin/bash

# 输入目录
input_dir=$1
# 输出目录
output_dir=$2

# 检查输出目录是否存在，如果不存在则创建
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
    echo "Created output directory: $output_dir"
fi

# 遍历输入目录中的所有jpg文件
for file in "$input_dir"/*.jpg; do
    # 获取文件名（不包含路径和扩展名）
    filename=$(basename "$file" .jpg)
    # 构建输出文件路径
    output_file="$output_dir/$filename.yuv"

    # 使用ffmpeg进行转换
    ffmpeg -i "$file" -c:v rawvideo -pix_fmt nv12 "$output_file"

    echo "Converted: $file -> $output_file"
done