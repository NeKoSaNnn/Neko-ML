import multiprocessing
import os
import os.path as osp
import shutil
import time


class neko_utils(object):
    def __init__(self, log_path="./log"):
        self.mkdir_nf(log_path)
        self.log_file = open(osp.join(log_path, "{}.log".format(self.get_now_day())), "a+")

    def __del__(self):
        self.log_file.close()

    def log(self, log_type, dict_val, is_dynamic=False):
        assert dict_val is not None and isinstance(dict_val, dict)
        Log = "{} -- ".format(log_type) + " , ".join(f"{k} : {v}" for k, v in dict_val.items())
        self.log_file.write(self.get_now_time() + " -- " + Log + "\n")
        if is_dynamic:
            print("\r" + Log, end="", flush=True)
        else:
            print("\r" + Log, end="\n")

    def divide_line(self, divide_line_str, total_len=100, is_log=False):
        divide_line_str = (" " + divide_line_str + " ").center(total_len, "=")
        print(divide_line_str)
        if is_log:
            self.log_file.write(divide_line_str + "\n")

    def mkdir_f(self, dir_path):
        # 强制mkdir
        try:
            shutil.rmtree(dir_path)
        except:
            pass
        os.makedirs(dir_path)

    def mkdir_nf(self, dir_path):
        # 不强制mkdir
        if not osp.exists(dir_path):
            os.makedirs(dir_path)

    def get_now_day(self):
        return time.strftime("%Y_%m_%d", time.localtime())

    def get_now_time(self):
        return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    def get_num_workers(self):
        return multiprocessing.cpu_count()
