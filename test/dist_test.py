import timeit, functools

def dist_test():
    pp_sketchlib.queryDatabase("listeria", "listeria", names, names, kmers, 1)

setup = """
import sys

sys.path.insert(0, "build/lib.macosx-10.9-x86_64-3.7")

import pp_sketchlib
"""
#import numpy as np
#
#from __main__ import dist_test
#
#kmers = np.arange(15, 30, 3)
#
#names = []
#sequences = []
#with open("rfiles.txt", 'r') as refFile:
#    for refLine in refFile:
#        refFields = refLine.rstrip().split("\t")
#        names.append(refFields[0])
#        sequences.append(list(refFields[1:]))
#"""


if __name__ == '__main__':
    import numpy as np
    import sys

    sys.path.insert(0, "build/lib.macosx-10.9-x86_64-3.7")
    import pp_sketchlib

    #from __main__ import dist_test

    kmers = np.arange(15, 30, 3)

    names = []
    sequences = []
    with open("rfiles.txt", 'r') as refFile:
        for refLine in refFile:
           refFields = refLine.rstrip().split("\t")
           names.append(refFields[0])
           sequences.append(list(refFields[1:]))

    t = timeit.Timer(functools.partial(pp_sketchlib.queryDatabase, "listeria", "listeria", names, names, kmers, 1), setup=setup)
    print(t.timeit(100))
