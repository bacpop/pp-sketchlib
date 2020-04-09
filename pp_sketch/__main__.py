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
    io.add_argument('--output',
                    default='ppsketch',
                    help="Output prefix [default = 'ppsketch']")
    io.add_argument('--print',
                    default=False,
                    action='store_true',
                    help='Print results to stdout instead of file')

    kmerGroup = parser.add_argument_group('Kmer comparison options')
    kmerGroup.add_argument('--min-k', default = 13, type=int, help='Minimum kmer length [default = 13]')
    kmerGroup.add_argument('--max-k', default = 29, type=int, help='Maximum kmer length [default = 29]')
    kmerGroup.add_argument('--k-step', default = 4, type=int, help='K-mer step size [default = 4]')
    kmerGroup.add_argument('--sketch-size', default=10000, type=int, help='Kmer sketch size [default = 10000]')
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

    if args.min_k >= args.max_k or args.min_k < 3 or args.max_k > 101 or args.k_step < 2:
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

    elif args.query:
        # TODO: add option to get names from HDF5 files
        rList = []
        ref = h5py.File(args.ref_db + ".h5", 'r')
        for sample_name in list(ref['sketches'].keys()):
            rList.append(sample_name)

        qList = []
        query = h5py.File(args.query_db + ".h5", 'r')
        for sample_name in list(query['sketches'].keys()):
            qList.append(sample_name)

        distMat = pp_sketchlib.queryDatabase(args.ref_db, args.query_db, rList, qList, kmers, 
                                             args.cpus, args.use_gpu, args.gpu_id)
        
        # get names order
        if args.print:
            names = iterDistRows(rList, qList, rList == qList)
            sys.stdout.write("\t".join(['Query', 'Reference', 'Core', 'Accessory']) + "\n")
            for i, (ref, query) in enumerate(names):
                sys.stdout.write("\t".join([query, ref, str(distMat[i,0]), str(distMat[i,1])]) + "\n")
        else:
            storePickle(rList, qList, rList == qList, distMat, args.output)

    sys.exit(0)

if __name__ == "__main__":
    main()