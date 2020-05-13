# Copyright 2019 John Lees

'''Wrapper around sketch functions'''

import os, sys

import numpy as np
import pickle
import h5py

import pp_sketchlib

from .__init__ import __version__

def iterDistRows(refSeqs, querySeqs, self=True):
    """Gets the ref and query ID for each row of the distance matrix

    Returns an iterable with ref and query ID pairs by row.

    Args:
        refSeqs (list)
            List of reference sequence names.
        querySeqs (list)
            List of query sequence names.
        self (bool)
            Whether a self-comparison, used when constructing a database.

            Requires refSeqs == querySeqs

            Default is True
    Returns:
        ref, query (str, str)
            Iterable of tuples with ref and query names for each distMat row.
    """
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

def storePickle(rlist, qlist, self, X, pklName):
    """Saves core and accessory distances in a .npy file, names in a .pkl

    Args:
        rlist (list)
            List of reference sequence names
        qlist (list)
            List of query sequence names
        self (bool)
            Whether an all-vs-all self DB
        X (numpy.array)
            n x 2 array of core and accessory distances
        pklName (str)
            Prefix for output files
    """
    with open(pklName + ".pkl", 'wb') as pickle_file:
        pickle.dump([rlist, qlist, self], pickle_file)
    np.save(pklName + ".npy", X)

def get_options():
    import argparse

    description = 'Run poppunk sketching/distances'
    parser = argparse.ArgumentParser(description=description,
                                     prog='pp_sketch')

    modeGroup = parser.add_argument_group('Mode of operation')
    mode = modeGroup.add_mutually_exclusive_group(required=True)
    mode.add_argument('--sketch',
                        action='store_true',
                        default=False,
                        help='Create a database of sketches')
    mode.add_argument('--join',
                        action='store_true',
                        default=False,
                        help='Combine two sketch databases '
                             '--ref-db and --query-db')
    mode.add_argument('--query',
                        action='store_true',
                        default=False,
                        help='Find distances between two sketch databases')

    io = parser.add_argument_group('Input/output')
    io.add_argument('--rfile',
                    help='Samples to sketch')
    io.add_argument('--ref-db',
                    help='Prefix of reference database file')
    io.add_argument('--query-db',
                    help='Prefix of query database file')
    io.add_argument('--subset',
                    help='List of names to include in query',
                    default=None)
    io.add_argument('--output',
                    default='ppsketch',
                    help="Output prefix [default = 'ppsketch']")
    io.add_argument('--print',
                    default=False,
                    action='store_true',
                    help='Print results to stdout instead of file')

    kmerGroup = parser.add_argument_group('Kmer comparison options')
    kmerGroup.add_argument('--read-k', default = False, action='store_true',
                            help='Use k-mer lengths found in query databases (query mode only)')
    kmerGroup.add_argument('--min-k', default = 13, type=int, help='Minimum kmer length [default = 13]')
    kmerGroup.add_argument('--max-k', default = 29, type=int, help='Maximum kmer length [default = 29]')
    kmerGroup.add_argument('--k-step', default = 4, type=int, help='K-mer step size [default = 4]')
    kmerGroup.add_argument('--sketch-size', default=10000, type=int, help='Kmer sketch size [default = 10000]')
    kmerGroup.add_argument('--jaccard', default=False, action='store_true', 
                            help='Output adjusted Jaccard distances, not core and accessory distances')
    kmerGroup.add_argument('--strand', default=True, action='store_false', help='Set to ignore complementary strand sequence '
                                                                                'e.g. for RNA viruses with preserved strand')
    kmerGroup.add_argument('--min-count', default=20, type=int, help='Minimum k-mer count from reads [default = 20]')
    kmerGroup.add_argument('--exact-counter', default=False, action='store_true', 
                            help='Use an exact rather than approximate k-mer counter '
                                  'when using reads as input (increases memory use) '
                                  '[default = False]')

    optimisation = parser.add_argument_group('Optimisation options')
    optimisation.add_argument('--cpus',
                              type=int,
                              default=1,
                              help='Number of CPUs to use '
                                   '[default = 1]')
    optimisation.add_argument('--use-gpu', default=False, action='store_true',
                              help='Use GPU code to calculate distances, '
                                   'if available [default = False]')
    optimisation.add_argument('--gpu-id',
                              type=int,
                              default=0,
                              help='ID of GPU to use '
                                   '[default = 0]')

    other = parser.add_argument_group('Other')
    other.add_argument('--version', action='version',
                       version='%(prog)s '+__version__)

    return parser.parse_args()


def main():
    args = get_options()

    if args.min_k >= args.max_k or args.min_k < 3 or args.max_k > 101 or args.k_step < 1:
        sys.stderr.write("Minimum kmer size " + str(args.min_k) + " must be smaller than maximum kmer size " +
                         str(args.max_k) + "; range must be between 3 and 101, step must be at least one\n")
        sys.exit(1)
    kmers = np.arange(args.min_k, args.max_k + 1, args.k_step)

    if args.sketch:
        names = []
        sequences = []
        
        with open(args.rfile, 'rU') as refFile:
            for refLine in refFile:
                refFields = refLine.rstrip().split("\t")
                names.append(refFields[0])
                sequences.append(list(refFields[1:]))


        if len(set(names)) != len(names):
            sys.stderr.write("Input contains duplicate names! All names must be unique\n")
            sys.exit(1)

        pp_sketchlib.constructDatabase(args.ref_db, names, sequences, kmers, 
                                       int(round(args.sketch_size/64)), args.strand, 
                                       args.min_count, args.exact_counter, args.cpus)

    elif args.join:
        join_name = args.output + ".h5"
        db1_name = args.ref_db + ".h5"
        db2_name = args.query_db + ".h5"

        hdf1 = h5py.File(db1_name, 'r')
        hdf2 = h5py.File(db2_name, 'r')

        try:
            v1 = hdf1['sketches'].attrs['sketch_version']
            v2 = hdf2['sketches'].attrs['sketch_version']
            if (v1 != v2):
                sys.stderr.write("Databases have been written with different sketch versions, "
                                 "joining not recommended\n")
        except RuntimeError as e:
            sys.stderr.write("Unable to check sketch version\n")

        hdf_join = h5py.File(join_name + ".tmp", 'w') # add .tmp in case join_name exists

        # Can only copy into new group, so for second file these are appended one at a time
        try:
            hdf1.copy('sketches', hdf_join)
            join_grp = hdf_join['sketches']
            read_grp = hdf2['sketches']
            for dataset in read_grp:
                join_grp.copy(read_grp[dataset], dataset)
        except RuntimeError as e:
            sys.stderr.write("ERROR: " + str(e) + "\n")
            sys.stderr.write("Joining sketches failed\n")
            sys.exit(1)

        # Clean up
        hdf1.close()
        hdf2.close()
        hdf_join.close()
        os.rename(join_name + ".tmp", join_name)
    
    elif args.query:
        rList = []
        ref = h5py.File(args.ref_db + ".h5", 'r')
        for sample_name in list(ref['sketches'].keys()):
            rList.append(sample_name)

        qList = []
        query = h5py.File(args.query_db + ".h5", 'r')
        for sample_name in list(query['sketches'].keys()):
            qList.append(sample_name)

        if args.subset != None:
            subset = []
            with open(args.subset, 'r') as subset_file:
                for line in subset_file:
                    sample_name = line.rstrip().split("\t")[0]
                    subset.append(sample_name)
            rList = list(set(rList).intersection(subset))
            qList = list(set(qList).intersection(subset))
            if (len(rList) == 0 or len(qList) == 0):
                sys.stderr.write("Subset has removed all samples\n")
                sys.exit(1)

        # Check inputs overlap
        db_kmers = set(ref['sketches/' + rList[0]].attrs['kmers']).intersection(
           query['sketches/' + rList[0]].attrs['kmers'] 
        )
        if args.read_k:
            query_kmers = sorted(db_kmers)
        else:
            query_kmers = sorted(set(kmers).intersection(db_kmers))
            if (len(kmers) == 0):
                sys.stderr.write("No requested k-mer lengths found in DB\n")
                sys.exit(1)
            elif (len(query_kmers) < len(query_kmers)):
                sys.stderr.write("Some requested k-mer lengths not found in DB\n")

        distMat = pp_sketchlib.queryDatabase(args.ref_db, args.query_db, rList, qList, query_kmers, 
                                             args.jaccard, args.cpus, args.use_gpu, args.gpu_id)
        
        # get names order
        if args.print:
            names = iterDistRows(rList, qList, rList == qList)
            if not args.jaccard:
                sys.stdout.write("\t".join(['Query', 'Reference', 'Core', 'Accessory']) + "\n")
                for i, (ref, query) in enumerate(names):
                    sys.stdout.write("\t".join([query, ref, str(distMat[i,0]), str(distMat[i,1])]) + "\n")
            else:
                sys.stdout.write("\t".join(['Query', 'Reference'] + [str(i) for i in query_kmers]) + "\n")
                for i, (ref, query) in enumerate(names):
                    sys.stdout.write("\t".join([query, ref] + [str(k) for k in distMat[i,]]) + "\n") 
        else:
            storePickle(rList, qList, rList == qList, distMat, args.output)

    sys.exit(0)

if __name__ == "__main__":
    main()