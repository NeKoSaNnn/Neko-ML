import os
import os.path as osp
import shutil
import time


class neko_utils(object):
    def __init__(self):
        self.log_file = open("log.txt", "a+")

    def __del__(self):
        self.log_file.close()

    def log(self, log_type, now_epoch, now_iter, dict_loss, is_dynamic=True):
        assert dict_loss is not None
        Log = "{} -- epoch {} iter {} : ".format(log_type, now_epoch, now_iter) + " , ".join(
            f"{loss_name} = {loss_val:.6f}" for
            loss_name, loss_val in
            dict_loss.items())
        self.log_file.writelines(self.get_now_time() + Log + "\n")
        print(Log, flush=is_dynamic)

    def divide_line(self, divide_line_str, total_len=60):
        divide_line_len = (total_len - len(divide_line_str) - 2) // 2
        line = "=" * divide_line_len + " " + divide_line_str + " " + "=" * divide_line_len
        print(line)

    def mkdir_f(self, dir_path):
        # 强制mkdir
        try:
            shutil.rmtree(dir_path)
        except:
            pass
        os.mkdir(dir_path)

    def mkdir_nf(self, dir_path):
        # 不强制mkdir
        if not osp.exists(dir_path):
            os.mkdir(dir_path)

    def get_now_time(self):
        return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
