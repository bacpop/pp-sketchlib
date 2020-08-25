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
subprocess.run("poppunk_sketch --sketch --rfile references.txt --ref-db test_db_phased --codon-phased --sketch-size 10000 --min-k 15 --k-step 4", shell=True, check=True)
# calculate distances
sys.stderr.write("Testing distances via python\n")
subprocess.run("poppunk_sketch --query --ref-db test_db --query-db test_db --min-k 15 --k-step 4", shell=True, check=True) # checks if can be run
subprocess.run("python test-dists.py --ref-db test_db --results ppsketch_ref", shell=True, check=True) # checks results match
subprocess.run("poppunk_sketch --query --ref-db test_db_phased --query-db test_db_phased --min-k 15 --k-step 4", shell=True, check=True) # checks if can be run
subprocess.run("python test-dists.py --ref-db test_db_phased --results ppsketch_ref_phased", shell=True, check=True) # checks results match
subprocess.run("poppunk_sketch --query --ref-db test_db --query-db test_db --min-k 15 --k-step 4 --sparse --threshold 0.01", shell=True, check=True) # checks if can be run
# Joining
sys.stderr.write("Testing join via python\n")
subprocess.run("poppunk_sketch --sketch --rfile db1_refs.txt --ref-db db1 --sketch-size 10000 --min-k 15 --k-step 4", shell=True, check=True)
subprocess.run("poppunk_sketch --sketch --rfile db2_refs.txt --ref-db db2 --sketch-size 10000 --min-k 15 --k-step 4", shell=True, check=True)
subprocess.run("poppunk_sketch --join --ref-db db1 --query-db db2 --output joined", shell=True, check=True)
# Matrix
sys.stderr.write("Testing matrix ops via python\n")
subprocess.run("python test-matrix.py", shell=True, check=True)

# C++ test
if not args.no_cpp:
    sys.stderr.write("Testing functions via C++\n")
    subprocess.run("sketch_test 12673_8#43 12673_8#43.contigs_velvet.fa 19183_4#69 19183_4#69.contigs_velvet.fa references.txt", shell=True, check=True)

sys.stderr.write("Tests completed\n")

