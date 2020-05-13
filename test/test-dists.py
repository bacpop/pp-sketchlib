import sys
import pp_sketchlib
import numpy as np
import h5py
import pickle

ref_db = "test_db"
old_results = "ppsketch_ref"
max_diff = 0.01

def iterDistRows(refSeqs, querySeqs, self=True):
    if self:
        if refSeqs != querySeqs:
            raise RuntimeError('refSeqs must equal querySeqs for db building (self = true)')
        for i, ref in enumerate(refSeqs):
            for j in range(i + 1, len(refSeqs)):
                yield(refSeqs[j], ref)
    else:
        for query in querySeqs:
            for ref in refSeqs:
                yield(ref, query)

def check_res(res, expected):
    if (not np.all(res == expected)):
        print(res)
        print(expected)
        raise RuntimeError("Results don't match")

# Generate distances
rList = []
ref = h5py.File(ref_db + ".h5", 'r')
for sample_name in list(ref['sketches'].keys()):
    rList.append(sample_name)

db_kmers = ref['sketches/' + rList[0]].attrs['kmers']

distMat = pp_sketchlib.queryDatabase(ref_db, ref_db, rList, rList, db_kmers)

# Read old distances
with open(old_results + ".pkl", 'rb') as pickle_file:
    rList_old, qList_old, self = pickle.load(pickle_file)
oldDistMat = np.load(old_results + ".npy")

# Check both match
assert(rList_old == rList)
names = iterDistRows(rList, rList, True)
for i, (ref, query) in enumerate(names):
    for j, (dist) in enumerate(['core', 'accessory']):
        if oldDistMat[i, j] != 0 and distMat[i, j] != 0:
            diff = 2*(distMat[i, j] - oldDistMat[i, j])/(oldDistMat[i, j] + distMat[i, j]) 
            if (abs(diff) > max_diff):
                sys.stderr.write(dist + " mismatches for " + ref + "," + query + "\n")
                sys.stderr.write("expected: " + str(oldDistMat[i, j]) + "; calculated: " + str(distMat[i, j]) + "\n")
                sys.exit(1)


