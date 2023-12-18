#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :
# Copyright 2019 John Lees

"""Wrapper for running python executable without installing"""
import os
os.environ["LD_PRELOAD"] = "/lib/x86_64-linux-gnu/libSegFault.so"


from pp_sketch.__main__ import main

if __name__ == '__main__':
    main()
