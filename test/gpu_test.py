#!/usr/bin/env python
# Copyright 2018-2020 John Lees and Nick Croucher

"""Tests for pp-sketchlib GPU"""

# Currently unsupported on Azure - run manually

import subprocess
import os
import sys
import shutil
import argparse

warn_diff = 0.01
warn_diff_fraction = 0.02
max_diff = 0.001
max_diff_fraction = 0.05

def compare_dists(d1, d2):
    fail = False
    if d1 != 0 and d2 != 0:
        diff = d1 - d2
        diff_fraction = 2*(diff)/(d1 + d2)
        if (abs(diff) > warn_diff and abs(diff_fraction) > warn_diff_fraction):
            sys.stderr.write("expected: " + str(d1) + "; calculated: " + str(d2) + "\n")
        if (abs(diff) > max_diff and abs(diff_fraction) > max_diff_fraction):
            sys.stderr.write("Difference outside tolerance")
            fail = True
    return(fail)

def compare_dists_files(cpu_file, gpu_file):
    fail = False
    with open(cpu_file, 'r') as cpu_dists, open(gpu_file, 'r') as gpu_dists:
        cpu_header = cpu_dists.readline()
        gpu_header = gpu_dists.readline()
        if (cpu_header != gpu_header):
            sys.stderr.write("Headers mismatch")
            fail = True
        for cpu_line, gpu_line in zip(cpu_dists, gpu_dists):
            rname, qname, core_dist, acc_dist = cpu_line.rstrip().split("\t")
            g_rname, g_qname, g_core_dist, g_acc_dist = cpu_line.rstrip().split("\t")
            if (rname != g_rname or qname != g_qname):
                sys.stderr.write("Sample order mismatch")
                fail = True
            fail = fail or compare_dists(float(core_dist), float(g_core_dist))
            fail = fail or compare_dists(float(acc_dist), float(g_acc_dist))
    return(fail)

# calculate distances
sys.stderr.write("Testing self distances on CPU\n")
subprocess.run("python ../pp_sketch-runner.py --query --ref-db small_mass --query-db small_mass --read-k --print", shell=True, check=True)
sys.stderr.write("Testing self distances on GPU\n")
subprocess.run("python ../pp_sketch-runner.py --query --ref-db small_mass --query-db small_mass --read-k --print --use-gpu", shell=True, check=True)

sys.stderr.write("Testing self distances match\n")
subprocess.run("python ../pp_sketch-runner.py --query --ref-db small_mass --query-db small_mass --read-k --print > tmp.cpu.self.dists.txt", shell=True, check=True)
subprocess.run("python ../pp_sketch-runner.py --query --ref-db small_mass --query-db small_mass --read-k --print --use-gpu > tmp.gpu.self.dists.txt", shell=True, check=True)
fail = compare_dists_files("tmp.cpu.self.dists.txt", "tmp.gpu.self.dists.txt")

sys.stderr.write("Testing query distances on CPU\n")
subprocess.run("python ../pp_sketch-runner.py --query --ref-db small_mass --query-db small_mass_plus1 --read-k --print", shell=True, check=True)
sys.stderr.write("Testing query distances on GPU\n")
subprocess.run("python ../pp_sketch-runner.py --query --ref-db small_mass --query-db small_mass_plus1 --read-k --print --use-gpu", shell=True, check=True)

sys.stderr.write("Testing query distances match\n")
subprocess.run("python ../pp_sketch-runner.py --query --ref-db small_mass --query-db small_mass_plus1 --read-k --print > tmp.cpu.query.dists.txt", shell=True, check=True)
subprocess.run("python ../pp_sketch-runner.py --query --ref-db small_mass --query-db small_mass_plus1 --read-k --print --use-gpu > tmp.gpu.query.dists.txt", shell=True, check=True)
fail = fail or compare_dists_files("tmp.cpu.query.dists.txt", "tmp.gpu.query.dists.txt")

sys.stderr.write("Tests completed\n")
sys.exit(fail)
