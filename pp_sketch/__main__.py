# Copyright 2019-2021 John Lees
"""sketchlib: fast sketching and core/accessory distance estimation from assemblies and reads

Usage:
  sketchlib sketch <files>... -o <output> [-k <kseq>|--kmer <k>] [-s <size>] [--single-strand] [--codon-phased] [--min-count <count>] [--exact-counter] [--cpus <cpus>] [--gpu <gpu>]
  sketchlib sketch -l <file-list> -o <output> [-k <kseq>|--kmer <k>] [-s <size>] [--single-strand] [--codon-phased] [--min-count <count>] [--exact-counter] [--cpus <cpus>] [--gpu <gpu>]
  sketchlib query dist <db1> [<db2>] [-o <output>] [--adj-random] [--cpus <cpus>] [--gpu <gpu>]
  sketchlib query jaccard <db1> [<db2>] [-o <output>] [--kmer <k>] [--adj-random] [--subset <file>] [--cpus <cpus>] [--gpu <gpu>]
  sketchlib query sparse <db1> [<db2>] (--kNN <k>|--threshold <max>) [-o <output>] [--accessory] [--adj-random] [--subset <file>] [--cpus <cpus>] [--gpu <gpu>]
  sketchlib join <db1> <db2> -o <output>
  sketchlib (add|remove) random <db1> [--cpus <cpus>]
  sketchlib (-h | --help)
  sketchlib (--version)

Options:
  -h --help     Show this help.
  --version     Show version.

  -o <output>    Output prefix.
  -l <file-list> File with a list of input files.
  --cpus <cpus>  Number of CPU threads to use [default: 1].
  --gpu <gpu>    Use GPU with specified device ID [default: -1].

  -k <kseq>     Sequence of k-mers to sketch (min,max,step) [default: 15,31,4].
  --kmer <k>    Sketch at a single k-mer length k.
  -s <size>     Sketch size [default: 10000].
  --single-strand  Ignore the reverse complement (e.g. in RNA viruses).
  --codon-phased  Use codon phased seeds X--X--X
  --min-count <count>  Minimum coverage count for k-mers from reads to be sketched [default: 20].
  --exact-counter  Use an exact k-mer count filter for reads (for genomes >10Mb)

  --adj-random  Adjust query matches for their chance of occurring at random
  --subset <file>  Only query samples matching names in file

  --kNN <k>  Use k nearest neighbours to sparsify
  --threshold <max>  Remove distances over max to sparsify
  --accessory  Use accessory distances rather than core to sparsify
"""

import os, sys
import re

import numpy as np
from scipy.sparse import save_npz
import pickle
import h5py

#sys.path.insert(0, '/Users/jlees/Documents/Imperial/pp-sketchlib/build/lib.macosx-10.9-x86_64-3.8')
import pp_sketchlib

from .matrix import ijv_to_coo

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

def getSampleNames(db_prefix):
    rList = []
    ref = h5py.File(db_prefix + ".h5", 'r')
    for sample_name in list(ref['sketches'].keys()):
        rList.append(sample_name)
    return rList

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
    if isinstance(X, np.ndarray):
        np.save(pklName + ".npy", X)
    else:
        save_npz(pklName + ".npz", X)

def get_options():
    from docopt import docopt
    arguments = docopt(__doc__, version="pp-sketchlib v"+__version__)

    # .h5 is removed from the end of DB names due to sketchlib API
    if arguments["<db1>"]:
        arguments["db1"] = re.sub(r"\.h5$", "", arguments["<db1>"])
    if arguments["<db2>"]:
        arguments["db2"] = re.sub(r"\.h5$", "", arguments["<db2>"])

    if arguments['--kmer']:
        arguments['kmers'] = [int(arguments['--kmer'])]
    else:
        try:
            (min_k, max_k, k_step) = [int(x) for x in arguments['-k'].split(",")]
            if min_k >= max_k or min_k < 3 or max_k > 101 or k_step < 1:
                raise RuntimeError("Invalid k-mer sizes")
            arguments['kmers'] = np.arange(int(min_k), int(max_k) + 1, int(k_step))
        except:
            sys.stderr.write("Minimum kmer size must be smaller than maximum kmer size; " +
                             "range must be between 3 and 101, step must be at least one\n")
            sys.exit(1)

    arguments['-s'] = int(round(float(arguments['-s'])/64))
    arguments['--min-count'] = int(arguments['--min-count'])

    if arguments['sparse']:
        if arguments['--kNN']:
            arguments['--kNN'] = int(arguments['--kNN'])
            arguments['--threshold'] = 0
        else:
            arguments['--kNN'] = 0
            arguments['--threshold'] = float(arguments['--threshold'])

    arguments['--cpus'] = int(arguments['--cpus'])
    arguments['--gpu'] = int(arguments['--gpu'])
    if int(arguments['--gpu']) >= 0:
        arguments['--use-gpu'] = True
    else:
        arguments['--use-gpu'] = False

    return arguments

def main():
    args = get_options()

    #
    # Create a database (sketch input)
    #
    if args['sketch']:
        names = []
        sequences = []

        if args['-l']:
            with open(args['-l'], 'rU') as refFile:
                for refLine in refFile:
                    refFields = refLine.rstrip().split("\t")
                    names.append(refFields[0])
                    sequences.append(list(refFields[1:]))
        else:
            for file in args['<files>']:
                name = re.sub("/", "_", file)
                names.append(name)
                sequences.append([file])


        if len(set(names)) != len(names):
            sys.stderr.write("Input contains duplicate names. "
                             "All names must be unique\n")
            sys.exit(1)

        pp_sketchlib.constructDatabase(args['-o'],
                                       names,
                                       sequences,
                                       args['kmers'],
                                       args['-s'],
                                       args['--codon-phased'],
                                       False,
                                       not args['--single-strand'],
                                       args['--min-count'],
                                       args['--exact-counter'],
                                       args['--cpus'],
                                       args['--use-gpu'],
                                       args['--gpu'])

    #
    # Join two databases
    #
    elif args['join']:
        join_name = args['-o'] + ".h5"
        db1_name = args['db1'] + ".h5"
        db2_name = args['db2'] + ".h5"

        hdf1 = h5py.File(db1_name, 'r')
        hdf2 = h5py.File(db2_name, 'r')

        try:
            v1 = hdf1['sketches'].attrs['sketch_version']
            v2 = hdf2['sketches'].attrs['sketch_version']
            if (v1 != v2):
                sys.stderr.write("Databases have been written with different sketch versions, "
                                 "joining not recommended (but proceeding anyway)\n")
            p1 = hdf1['sketches'].attrs['codon_phased']
            p2 = hdf2['sketches'].attrs['codon_phased']
            if (p1 != p2):
                sys.stderr.write("One database uses codon-phased seeds - cannot join "
                                 "with a standard seed database\n")
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

            if 'random' in hdf1 or 'random' in hdf2:
                sys.stderr.write("Random matches found in one database, which will not be copied\n"
                                 "Use --add-random to recalculate for the joined DB\n")
        except RuntimeError as e:
            hdf1.close()
            hdf2.close()
            hdf_join.close()
            sys.stderr.write("ERROR: " + str(e) + "\n")
            sys.stderr.write("Joining sketches failed\n")
            sys.exit(1)

        # Clean up
        hdf1.close()
        hdf2.close()
        hdf_join.close()
        os.rename(join_name + ".tmp", join_name)

    #
    # Query a database (calculate distances)
    #
    elif args['query']:
        rList = getSampleNames(args['db1'])

        if 'db2' not in args:
            args['db2'] = args['db1']
            qList = rList
        else:
            qList = getSampleNames(args['db2'])

        if args['--subset'] != None:
            subset = []
            with open(args['--subset'], 'r') as subset_file:
                for line in subset_file:
                    sample_name = line.rstrip().split("\t")[0]
                    subset.append(sample_name)
            rList = list(set(rList).intersection(subset))
            qList = list(set(qList).intersection(subset))
            if (len(rList) == 0 or len(qList) == 0):
                sys.stderr.write("Subset has removed all samples\n")
                sys.exit(1)

        # Check inputs overlap
        ref = h5py.File(args['db1'] + ".h5", 'r')
        query = h5py.File(args['db2'] + ".h5", 'r')
        db_kmers = set(ref['sketches/' + rList[0]].attrs['kmers']).intersection(
           query['sketches/' + qList[0]].attrs['kmers']
        )
        query_kmers = sorted(db_kmers)
        if args['jaccard'] and len(args['kmers']) == 1:
            if args['kmers'][0] not in query_kmers:
                sys.stderr.write("Selected --kmer is not in both query databases\n")
                sys.exit(1)
            else:
                query_kmers = args['kmers']
        ref.close()
        query.close()

        if args['sparse']:
            sparseIdx = pp_sketchlib.queryDatabaseSparse(args['db1'],
                                                         args['db2'],
                                                         rList,
                                                         qList,
                                                         query_kmers,
                                                         args['--adj-random'],
                                                         args['--threshold'],
                                                         args['--kNN'],
                                                         not args['--accessory'],
                                                         args['--cpus'],
                                                         args['--use-gpu'],
                                                         args['--gpu'])
            if not args['-o']:
                if args['--accessory']:
                    distName = 'Accessory'
                else:
                    distName = 'Core'
                sys.stdout.write("\t".join(['Query', 'Reference', distName]) + "\n")

                (i_vec, j_vec, dist_vec) = sparseIdx
                for (i, j, dist) in zip(i_vec, j_vec, dist_vec):
                    sys.stdout.write("\t".join([rList[i], qList[j], str(dist)]) + "\n")

            else:
                coo_matrix = ijv_to_coo(sparseIdx, (len(rList), len(qList)), np.float32)
                storePickle(rList, qList, rList == qList, coo_matrix, args['-o'])

        else:
            distMat = pp_sketchlib.queryDatabase(args['db1'],
                                                 args['db2'],
                                                 rList,
                                                 qList,
                                                 query_kmers,
                                                 args['--adj-random'],
                                                 args['jaccard'],
                                                 args['--cpus'],
                                                 args['--use-gpu'],
                                                 args['--gpu'])

            # get names order
            if not args['-o']:
                names = iterDistRows(rList, qList, rList == qList)
                if not args['jaccard']:
                    sys.stdout.write("\t".join(['Query', 'Reference', 'Core', 'Accessory']) + "\n")
                    for i, (ref, query) in enumerate(names):
                        sys.stdout.write("\t".join([query, ref, str(distMat[i,0]), str(distMat[i,1])]) + "\n")
                else:
                    sys.stdout.write("\t".join(['Query', 'Reference'] + [str(i) for i in query_kmers]) + "\n")
                    for i, (ref, query) in enumerate(names):
                        sys.stdout.write("\t".join([query, ref] + [str(k) for k in distMat[i,]]) + "\n")
            else:
                storePickle(rList, qList, rList == qList, distMat, args['-o'])

    #
    # Add random match chances to an older database
    #
    elif args['random']:
        ref = h5py.File(args['db1'] + ".h5", 'r+')
        if args['add']:
            rList = getSampleNames(args['db1'])
            db_kmers = ref['sketches/' + rList[0]].attrs['kmers']
            ref.close()

            pp_sketchlib.addRandom(args['db1'],
                                   rList,
                                   db_kmers,
                                   not args['--single-strand'],
                                   args['--cpus'])
        elif args['remove']:
            if 'random' in ref:
                del ref['random']
            else:
                sys.stderr.write(args['db1'] + \
                  " doesn't contain random match chances\n")
            ref.close()

    sys.exit(0)

if __name__ == "__main__":
    main()