#!/usr/bin/env python
# Copyright 2018-2020 John Lees and Nick Croucher

"""Tests for PopPUNK"""

import subprocess
import os
import sys
import shutil
import argparse


parser = argparse.ArgumentParser(description='Test pp-sketchlib',
                                 prog='run_test')
parser.add_argument('--no-cpp',
                    action='store_true',
                    default=False,
                    help='Do not run C++ tests')
args = parser.parse_args()

if not os.path.isfile("12754_4#89.contigs_velvet.fa"):
    sys.stderr.write("Extracting example dataset\n")
    subprocess.run("tar xf example_set.tar.bz2", shell=True, check=True)

# python tests
# create sketches
sys.stderr.write("Testing sketching via python\n")
subprocess.run("poppunk_sketch --sketch --rfile references.txt --ref-db test_db --sketch-size 10000 --min-k 15 --k-step 4", shell=True, check=True)
# calculate distances
sys.stderr.write("Testing distances via python\n")
subprocess.run("poppunk_sketch --query --ref-db test_db.h5 --query-db test_db.h5", shell=True, check=True)

# C++ test
if not args.no_cpp:
    sys.stderr.write("Testing functions via C++\n")
    subprocess.run("sketch_test 12673_8#43 12673_8#43.contigs_velvet.fa 19183_4#69 19183_4#69.contigs_velvet.fa references.txt", shell=True, check=True)

sys.stderr.write("Tests completed\n")

