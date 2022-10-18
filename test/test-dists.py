import sys
#sys.path.insert(0, '../build/lib.macosx-10.9-x86_64-3.10')
import pp_sketchlib
import numpy as np
import h5py
import pickle
import argparse

def dist_in_tolerance(d1, d2, dist_name, sample_name, tols):
    if d1 != 0 and d2 != 0:
            diff = d1 - d2
            diff_fraction = 2*(diff)/(d1 + d2)
            if (abs(diff) > tols['warn_diff'] and abs(diff_fraction) > tols['warn_diff_frac']):
                sys.stderr.write(dist_name + " mismatches for " + sample_name + "\n")
                sys.stderr.write("expected: " + str(d1) + "; calculated: " + str(d2) + "\n")
            if (abs(diff) > tols['err_diff'] and abs(diff_fraction) > tols['err_diff_frac']):
                sys.stderr.write("Difference outside tolerance")
                sys.exit(1)

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

def get_kNN_sparse_tuple(square_core_mat,kNN):
    i_vec = []
    j_vec = []
    dist_vec = []
    for i in range(0,square_core_mat.shape[0]):
        sorted_indices = np.argsort(square_core_mat[i,:])
        j_index = 0
        neighbour_count = 0
        while neighbour_count < kNN:
            if (sorted_indices[j_index] != i):
                i_vec.append(i)
                j_vec.append(sorted_indices[j_index])
                dist_vec.append(square_core_mat[i,sorted_indices[j_index]])
                neighbour_count = neighbour_count + 1
            j_index = j_index + 1
    sparse_knn = (i_vec,j_vec,dist_vec)
    return sparse_knn

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

tols = {"warn_diff": args.warn_diff,
        "warn_diff_frac": args.warn_diff_frac,
        "err_diff": args.error_diff,
        "err_diff_frac": args.error_diff_frac}

# Generate distances
rList = []
ref = h5py.File(args.ref_db + ".h5", 'r')
for sample_name in list(ref['sketches'].keys()):
    rList.append(sample_name)

db_kmers = ref['sketches/' + rList[0]].attrs['kmers']
ref.close()

distMat = pp_sketchlib.queryDatabase(ref_db_name=args.ref_db,
                                     query_db_name=args.ref_db,
                                     rList=rList,
                                     qList=rList,
                                     klist=db_kmers)
jaccard_dists = pp_sketchlib.queryDatabase(ref_db_name=args.ref_db,
                                           query_db_name=args.ref_db,
                                           rList=rList,
                                           qList=rList,
                                           klist=db_kmers,
                                           jaccard = True)
jaccard_dists_raw = pp_sketchlib.queryDatabase(ref_db_name=args.ref_db,
                                               query_db_name=args.ref_db,
                                               rList=rList,
                                               qList=rList,
                                               klist=db_kmers,
                                               jaccard = True,
                                               random_correct = False)
distMat_all = np.hstack((distMat, jaccard_dists, jaccard_dists_raw))

# Read old distances
with open(args.results + ".pkl", 'rb') as pickle_file:
    rList_old, qList_old, self = pickle.load(pickle_file)
oldDistMat = np.load(args.results + ".npy")
oldJaccardDistMat = np.load(args.results + "_jaccard.npy")
oldRawJaccardDistMat = np.load(args.results + "_raw_jaccard.npy")
oldDistMat_all = np.hstack((oldDistMat, oldJaccardDistMat, oldRawJaccardDistMat))

# Check both match
if (rList_old != rList):
    sys.stderr.write("Genome order mismatching\n")
    print(rList)
    print(rList_old)
    sys.exit(1)

names = iterDistRows(rList, rList, True)
for i, (ref, query) in enumerate(names):
    for j, (dist) in enumerate(['core', 'accessory'] + [str(x) for x in db_kmers]):
        dist_in_tolerance(oldDistMat_all[i, j], distMat_all[i, j], dist, ref + "," + query, tols)

# Test sparse queries
square_core_mat = pp_sketchlib.longToSquare(distVec=distMat[:, [0]])

kNN=3
sparseDistMat = pp_sketchlib.querySelfSparse(ref_db_name=args.ref_db,
                                             rList=rList,
                                             klist=db_kmers,
                                             kNN=kNN)
sparse_knn = get_kNN_sparse_tuple(square_core_mat,kNN)
if (sparseDistMat[0] != sparse_knn[0] or
    sparseDistMat[1] != sparse_knn[1]):
    sys.stderr.write("Sparse distances (kNN) mismatching\n")
    print(sparseDistMat)
    print(sparse_knn)
    sys.exit(1)

for idx, (d1, d2) in enumerate(zip(sparseDistMat[2], sparseDistMat[2])):
    dist_in_tolerance(d1, d2, "sparse distances (kNN)", str(idx), tols)

cutoff=0.01
sparseDistMat = pp_sketchlib.querySelfSparse(ref_db_name=args.ref_db,
                                             rList=rList,
                                             klist=db_kmers,
                                             dist_cutoff=cutoff)
sparse_threshold = pp_sketchlib.sparsifyDistsByThreshold(distMat=square_core_mat, distCutoff=cutoff)
if (sparseDistMat[0] != sparse_threshold[0] or
    sparseDistMat[1] != sparse_threshold[1]):
    sys.stderr.write("Sparse distances (cutoff) mismatching\n")
    print(sparseDistMat)
    print(sparse_threshold)
    sys.exit(1)

for idx, (d1, d2) in enumerate(zip(sparseDistMat[2], sparseDistMat[2])):
    dist_in_tolerance(d1, d2, "sparse distances (cutoff)", str(idx), tols)

sys.exit(0)
