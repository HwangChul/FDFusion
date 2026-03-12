import os
import shutil
import re


def save_code_files(source_file, destination_folder):
    """
    保存代码文件到指定的目标文件夹。

    参数:
        source_file (str): 源文件路径（通常是当前的训练脚本文件）。
        destination_folder (str): 目标文件夹路径，用于保存相关文件。
    """
    # 确保目标文件夹存在
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)  # 如果目标文件夹不存在，则创建

    # 读取源文件内容
    with open(source_file, 'r', encoding="utf-8") as file:
        content = file.read()  # 读取源文件内容

    # 使用正则表达式查找模型文件的名称
    match = re.search(r'from net\.(\w+) import Net', content)
    if match:  # 如果匹配到 `from net.<model_name> import Net` 的语句
        model_name = match.group(1)  # 提取模型文件的名称
        model_file_path = os.path.join('net', f'{model_name}.py')  # 构造模型文件的路径

    # 构造目标训练文件路径
    dest_train_file_path = os.path.join(destination_folder, os.path.basename(__file__))

    # 复制训练文件到目标文件夹
    shutil.copyfile(source_file, dest_train_file_path)

    # 复制模型文件到目标文件夹
    shutil.copyfile(model_file_path, os.path.join(destination_folder, f'{model_name}.py'))
