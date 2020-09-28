import sys
import pp_sketchlib
import numpy as np
import h5py
import pickle
import argparse

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


description = 'Run poppunk sketching/distances'
parser = argparse.ArgumentParser(description=description,
                                    prog='poppunk_sketch')

parser.add_argument('--ref-db', help='Prefix of db to test')
parser.add_argument('--results', help='Prefix of results to compare against')

parser.add_argument('--warn-diff', type=float, default = 0.0005,
                    help='Absolute difference to WARN for')
parser.add_argument('--warn-diff-frac', type=float, default = 0.02,
                    help='Percentage difference to WARN for')
parser.add_argument('--error-diff', type=float, default = 0.002,
                    help='Absolute difference to ERROR for')
parser.add_argument('--error-diff-frac', type=float, default = 0.05,
                    help='Percentage difference to ERROR for')

args = parser.parse_args()

# Generate distances
rList = []
ref = h5py.File(args.ref_db + ".h5", 'r')
for sample_name in list(ref['sketches'].keys()):
    rList.append(sample_name)

db_kmers = ref['sketches/' + rList[0]].attrs['kmers']
ref.close()

distMat = pp_sketchlib.queryDatabase(args.ref_db, args.ref_db, rList, rList, db_kmers)
jaccard_dists = pp_sketchlib.queryDatabase(args.ref_db, args.ref_db, rList, rList,
                                           db_kmers, jaccard = True)
jaccard_dists_raw = pp_sketchlib.queryDatabase(args.ref_db, args.ref_db, rList, rList,
                                           db_kmers, jaccard = True,
                                           random_correct = False)
distMat = np.hstack((distMat, jaccard_dists, jaccard_dists_raw))

# Read old distances
with open(args.results + ".pkl", 'rb') as pickle_file:
    rList_old, qList_old, self = pickle.load(pickle_file)
oldDistMat = np.load(args.results + ".npy")
oldJaccardDistMat = np.load(args.results + "_jaccard.npy")
oldRawJaccardDistMat = np.load(args.results + "_raw_jaccard.npy")
oldDistMat = np.hstack((oldDistMat, oldJaccardDistMat, oldRawJaccardDistMat))

# Check both match
if (rList_old != rList):
    sys.stderr.write("Genome order mismatching\n")
    print(rList)
    print(rList_old)
    sys.exit(1)

names = iterDistRows(rList, rList, True)
for i, (ref, query) in enumerate(names):
    for j, (dist) in enumerate(['core', 'accessory'] + [str(x) for x in db_kmers]):
        if oldDistMat[i, j] != 0 and distMat[i, j] != 0:
            diff = distMat[i, j] - oldDistMat[i, j]
            diff_fraction = 2*(diff)/(oldDistMat[i, j] + distMat[i, j])
            if (abs(diff) > args.warn_diff and abs(diff_fraction) > args.warn_diff_frac):
                sys.stderr.write(dist + " mismatches for " + ref + "," + query + "\n")
                sys.stderr.write("expected: " + str(oldDistMat[i, j]) + "; calculated: " + str(distMat[i, j]) + "\n")
            if (abs(diff) > args.error_diff and abs(diff_fraction) > args.error_diff_frac):
                sys.stderr.write("Difference outside tolerance")
                sys.exit(1)

sys.exit(0)
