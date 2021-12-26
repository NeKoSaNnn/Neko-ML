import os
import os.path as osp
import shutil
import time


def log(log_type, now_epoch, now_iter, dict_loss):
    assert dict_loss is not None
    Log = "{} -- epoch {} iter {} : ".format(log_type, now_epoch, now_iter)
    for loss_name, loss_val in dict_loss.items():
        Log += "{} = {:.6f} ".format(loss_name, loss_val)
    print(Log)


def divid_line(divid_line_str, totoal_len=60):
    divid_line_len = (totoal_len - len(divid_line_str) - 2) // 2
    line = "=" * divid_line_len + " " + divid_line_str + " " + "=" * divid_line_len
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
