#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import os
import os.path as osp

import argparse
import sys

root_dir_name = osp.dirname(sys.path[0])  # ...Neko-ML/
now_dir_name = sys.path[0]  # ...DRSegFL/
sys.path.append(root_dir_name)

from DRSegFL import utils, constants, logger


def kill(pid, logger):
    if os.name == "nt":
        # Windows
        cmd = "taskkill /pid " + str(pid) + " /f"
        try:
            os.system(cmd)
            logger.info("killed {}".format(pid))
        except Exception as e:
            logger.error(e)
    elif os.name == "posix":
        # Linux
        cmd = "kill -9" + str(pid)
        try:
            os.system(cmd)
            logger.info("killed {}".format(pid))
        except Exception as e:
            logger.error(e)
    else:
        logger.error("Undefined os.name")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="path to the config of the process to be stopped")

    args = parser.parse_args()
    assert osp.exists(args.config_path), "{} not exist".format(args.config_path)

    logger = logger.Logger()
    try:
        config = utils.load_json(args.config_path)
        if constants.PID in config:
            pid = config[constants.PID]
            kill(pid, logger)
        else:
            logger.info("config haven't run")
    except Exception as e:
        logger.error(e)
