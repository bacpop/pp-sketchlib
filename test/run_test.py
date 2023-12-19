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

if not os.path.isfile("12754_4#89.contigs_velvet.fa"):
    sys.stderr.write("Extracting example dataset\n")
    subprocess.run("tar xf example_set.tar.bz2", shell=True, check=True)

# python tests

# module attributes
import pp_sketchlib
print(pp_sketchlib.version)
print(pp_sketchlib.sketchVersion)

# create sketches
sys.stderr.write("Sketch smoke test\n")
subprocess.run("python ../sketchlib-runner.py sketch -l references.txt -o test_db -s 10000 -k 15,29,4 --cpus 1", shell=True, check=True)
#os.remove("test_db.h5")
#subprocess.run("python ../sketchlib-runner.py sketch -l references.txt -o test_db -s 10000 -k 15,29,4 --cpus 2", shell=True, check=True)
subprocess.run("python ../sketchlib-runner.py sketch -l references.txt -o test_db_phased --codon-phased --cpus 1", shell=True, check=True)
subprocess.run("python ../sketchlib-runner.py sketch 12673_8#24.contigs_velvet.fa 12673_8#34.contigs_velvet.fa -o test_db_small -s 1000 --kmer 14", shell=True, check=True)
subprocess.run("python ../sketchlib-runner.py add random test_db --cpus 2", shell=True, check=True)
# calculate distances
sys.stderr.write("Distance integration test\n")
subprocess.run("python ../sketchlib-runner.py query dist test_db --cpus 1", shell=True, check=True) # checks if can be run
subprocess.run("python ../sketchlib-runner.py query dist test_db -o ppsketch --cpus 1", shell=True, check=True) # checks if can be run
subprocess.run("python ../sketchlib-runner.py query jaccard test_db_small --cpus 1", shell=True, check=True) # checks if can be run
subprocess.run("python test-dists.py --ref-db test_db --results ppsketch_ref", shell=True, check=True) # checks results match
subprocess.run("python ../sketchlib-runner.py query dist test_db_phased --cpus 1", shell=True, check=True) # checks if can be run
subprocess.run("python test-dists.py --ref-db test_db_phased --results ppsketch_ref_phased", shell=True, check=True) # checks results match

sys.stderr.write("Sparse distance smoke test\n")
subprocess.run("python ../sketchlib-runner.py query sparse test_db --kNN 2", shell=True, check=True) # checks if can be run
subprocess.run("python ../sketchlib-runner.py query sparse test_db -o sparse_query --kNN 2", shell=True, check=True) # checks if can be run
subprocess.run("python ../sketchlib-runner.py query sparse test_db --threshold 0.01", shell=True, check=True) # checks if can be run
subprocess.run("python ../sketchlib-runner.py query sparse jaccard test_db --kNN 2 --kmer 19", shell=True, check=True) # checks if can be run

sys.stderr.write("Ref v query distance smoke test\n")
subprocess.run("python ../sketchlib-runner.py sketch -l rlist.txt -o r_db --cpus 1", shell=True, check=True)
subprocess.run("python ../sketchlib-runner.py sketch -l qlist.txt -o q_db --cpus 1", shell=True, check=True)
subprocess.run("python ../sketchlib-runner.py query dist r_db q_db.h5", shell=True, check=True) # checks if can be run
subprocess.run("python ../sketchlib-runner.py query jaccard r_db q_db", shell=True, check=True) # checks if can be run

# Joining
sys.stderr.write("Join smoke test\n")
subprocess.run("python ../sketchlib-runner.py sketch -l db1_refs.txt -o db1 --cpus 1", shell=True, check=True)
subprocess.run("python ../sketchlib-runner.py sketch -l db2_refs.txt -o db2 --cpus 1", shell=True, check=True)
subprocess.run("python ../sketchlib-runner.py join db1.h5 db2.h5 -o joined", shell=True, check=True)
# Random
sys.stderr.write("Random test\n")
subprocess.run("python ../sketchlib-runner.py remove random test_db --cpus 1", shell=True, check=True)
# Matrix
sys.stderr.write("Matrix integration test\n")
subprocess.run("python test-matrix.py", shell=True, check=True)

sys.stderr.write("Tests completed\n")

