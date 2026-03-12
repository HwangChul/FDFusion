import os
import logging
import time
import shutil
import json

class Logger():
    def __init__(self,rootpath=r'.',timestamp=False):
        super(Logger, self).__init__()
        self.change_path(rootpath,timestamp)

    def init_logger(self):
        if self.timestamp:
            self.logpath = os.path.join(self.rootpath,time.strftime("%y_%m_%d_%H_%M", time.localtime()))
        else:
            self.logpath=self.rootpath
        print("output: "+self.logpath)
        if not os.path.exists(self.logpath):
            os.makedirs(self.logpath)
        self.txtpath=os.path.join(self.logpath,'log.txt')

    def change_path(self,rootpath,timestamp):
        self.rootpath=rootpath
        self.timestamp=timestamp
        self.init_logger()

    def log(self,logmessage):
        file = open(self.txtpath,'a')
        file.write(logmessage+'\n')
        file.close()

    def log_and_print(self,logmessage):
        self.log(logmessage)
        print(logmessage)

    def save_param(self,para_dic):
        f = open(os.path.join(self.logpath, 'param.json'), 'w')
        f.write(json.dumps(para_dic))
        f.close()

    def new_subfolder(self,foldername):
        folderpath=os.path.join(self.logpath,foldername)
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)



class Logger1():
    def __init__(self, rootpath=r'.', timestamp=False):
        """
        初始化 Logger1 类。
        参数:
            rootpath (str): 日志文件存储的根路径，默认当前目录。
            timestamp (bool): 是否在日志路径中添加时间戳文件夹，默认 False。
        """
        super(Logger1, self).__init__()
        self.change_path(rootpath, timestamp)  # 初始化日志路径

    def init_logger(self):
        """
        初始化日志路径和文件。
        如果 timestamp 为 True，则在根路径下创建一个带时间戳的文件夹作为日志路径。
        如果路径不存在，则创建路径。
        """
        if self.timestamp:
            # 生成当前时间戳文件夹名，例如 "24_01_26_15_30"
            self.timestamp_folder_name = time.strftime("%y_%m_%d_%H_%M", time.localtime())
            self.logpath = os.path.join(self.rootpath, self.timestamp_folder_name)
        else:
            self.logpath = self.rootpath  # 不使用时间戳时直接使用根路径

        print("output: " + self.logpath)  # 打印日志输出路径

        # 如果路径不存在，则创建路径
        if not os.path.exists(self.logpath):
            os.makedirs(self.logpath)

        # 设置日志文件路径
        self.txtpath = os.path.join(self.logpath, 'log.txt')

    def change_path(self, rootpath, timestamp):
        """
        更改日志存储路径。
        参数:
            rootpath (str): 新的根路径。
            timestamp (bool): 是否在路径中添加时间戳文件夹。
        """
        self.rootpath = rootpath
        self.timestamp = timestamp
        self.init_logger()  # 重新初始化日志路径

    def log(self, logmessage):
        """
        记录日志消息到日志文件。
        参数:
            logmessage (str): 要记录的日志内容。
        """
        file = open(self.txtpath, 'a')  # 以追加模式打开日志文件
        file.write(logmessage + '\n')  # 写入日志内容
        file.close()  # 关闭文件

    def log_and_print(self, logmessage):
        """
        记录日志消息到日志文件并打印到控制台。
        参数:
            logmessage (str): 要记录的日志内容。
        """
        self.log(logmessage)  # 记录日志
        print(logmessage)  # 打印日志内容到控制台

    def save_param(self, para_dic):
        """
        保存参数字典为 JSON 文件。
        参数:
            para_dic (dict): 参数字典。
        """
        f = open(os.path.join(self.logpath, 'param.json'), 'w')  # 打开或创建 param.json 文件
        f.write(json.dumps(para_dic))  # 将参数字典写入 JSON 文件
        f.close()  # 关闭文件

    def new_subfolder(self, foldername):
        """
        在日志路径下创建新的子文件夹。
        参数:
            foldername (str): 子文件夹名称。
        """
        folderpath = os.path.join(self.logpath, foldername)  # 拼接子文件夹路径
        if not os.path.exists(folderpath):  # 如果文件夹不存在，则创建
            os.makedirs(folderpath)

    def get_timestamp_folder_name(self):
        """
        获取当前时间戳文件夹名称。
        如果 timestamp 为 False，则返回 None。
        返回:
            str 或 None: 时间戳文件夹名称或 None。
        """
        return self.timestamp_folder_name if self.timestamp else None

    


    
if __name__ == '__main__':
    logger=Logger1("bhw_log", timestamp=True)
    time = logger.get_timestamp_folder_name()
    print(time)