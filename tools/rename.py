import os

# folder_path = 'datasets/Rope3D/preds_kitti_MONO_2.5D_int8/val'  # 替换为你的文件夹路径
folder_path = 'datasets/Rope3D/extrinsics/val'  # 替换为你的文件夹路径

# 获取文件夹中的所有文件名
file_names = sorted(os.listdir(folder_path))

# 重命名文件
for i, file_name in enumerate(file_names):
    file_path = os.path.join(folder_path, file_name)
    new_file_name = f"{i+1:010d}.yaml"
    new_file_path = os.path.join(folder_path, new_file_name)
    os.rename(file_path, new_file_path)
    print(f"已将文件 {file_name} 重命名为 {new_file_name}")