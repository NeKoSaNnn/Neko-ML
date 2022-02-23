#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:mjx
"""
import os.path as osp
import sys

sys.path.append(osp.dirname(sys.path[0]))
from neko import neko_args


class args(neko_args.neko_args):
    def __init__(self):
        super(args, self).__init__()

        # federated arguments
        self.parser.add_argument("--iid", action="store_true", help="i.i.d")
        self.parser.add_argument("--non_iid", action="store_true", help="non i.i.d")
        self.parser.add_argument("--num_users", type=int, default=100, help="number of users: K")
        self.parser.add_argument("--client_frac", type=float, default=0.1, help="the fraction of clients: C")
        self.parser.add_argument("--local_ep", type=int, default=5, help="the number of local epochs: E")
        self.parser.add_argument("--local_bs", type=int, default=10, help="the number of local batch size: B")
        self.parser.add_argument("--all_clients", action="store_true", help="aggregation over all clients")
