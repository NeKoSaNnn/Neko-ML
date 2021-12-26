import os
import os.path as osp
import shutil
import sys
import time

sys.path.append(osp.dirname(sys.path[0]))


def log(log_type, now_epoch, now_iter, dict_loss):
    assert dict_loss is not None
    Log = "{} -- epoch {} iter {} : ".format(log_type, now_epoch, now_iter)
    for loss_name, loss_val in dict_loss.items():
        Log += "{} = {:.6f} ".format(loss_name, loss_val)
    print(Log)


def divide_line(divide_line_str, total_len=60):
    divide_line_len = (total_len - len(divide_line_str) - 2) // 2
    line = "=" * divide_line_len + " " + divide_line_str + " " + "=" * divide_line_len
    print(line)


def mkdir_f(dir_path):
    # 强制mkdir
    try:
        shutil.rmtree(dir_path)
    except:
        pass
    os.mkdir(dir_path)


def mkdir_nf(dir_path):
    # 不强制mkdir
    if not osp.exists(dir_path):
        os.mkdir(dir_path)


def get_now_time():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
