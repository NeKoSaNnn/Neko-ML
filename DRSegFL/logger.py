#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""


class Logger(object):
    def __init__(self, logger):
        self.logger = logger

    def debug(self, msg):
        if self.logger is not None:
            self.logger.debug(msg)
        else:
            print("[Debug]:{}".format(msg))

    def info(self, msg):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print("[Info]:{}".format(msg))

    def warn(self, msg):
        if self.logger is not None:
            self.logger.warn(msg)
        else:
            print("[Warn]:{}".format(msg))

    def warning(self, msg):
        if self.logger is not None:
            self.logger.warning(msg)
        else:
            print("[Warn]:{}".format(msg))

    def error(self, msg):
        if self.logger is not None:
            self.logger.error(msg)
        else:
            print("[Error]:{}".format(msg))
