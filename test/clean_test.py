#!/usr/bin/env python
# Copyright 2018 John Lees and Nick Croucher

"""Clean test files"""

import os
import sys
import shutil

def deleteDir(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)    

sys.stderr.write("Cleaning up tests\n")
refs = []
with open("references.txt", 'r') as ref_file:
    for line in ref_file:
        refs.append(line.rstrip().split("\t")[1])

# clean up
outputFiles = [
    "test_db.h5",
    "listeria.h5",
    "sample.h5",
    "full.h5"
]
for delFile in outputFiles:
    os.remove(delFile)

for ref in refs:
    os.remove(ref)

