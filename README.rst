pp-sketchlib
------------
|Build status| |Anaconda package|

.. |Build status| image:: https://dev.azure.com/jlees/pp-sketchlib/_apis/build/status/johnlees.pp-sketchlib?branchName=master
   :target: https://dev.azure.com/jlees/pp-sketchlib/_build/latest?definitionId=1&branchName=master

.. |Anaconda package| image:: https://anaconda.org/bioconda/pp-sketchlib/badges/version.svg
   :target: https://anaconda.org/bioconda/pp-sketchlib

Library of sketching functions used by `PopPUNK <https://www.poppunk.net>`__.

Installation
============
Install using conda (recommended)::

    conda -c bioconda install pp-sketchlib

.. note::
    If you are getting ``UnsatisfiableError`` or similar version conflicts try following the 
    `tips on conda-forge <https://conda-forge.org/docs/user/tipsandtricks.html#using-multiple-channels>`__. 
    It may also help if you downgrade your version of conda (to 4.5).

Or install locally::

    python setup.py install

For this option you will need a C++14 compiler (``GCC >=7.2.0`` or ``Clang``), 
``pybind11``, ``hdf5`` and ``CMake (>=3.12)``.

Usage
=====
Create a set of sketches and save these as a database::

    poppunk_sketch --sketch --rfile rfiles.txt --ref-db listeria --sketch-size 156 --cpus 4 --min-k 15 --k-step 2

The input file ``rfiles.txt`` has one sequence per line. The first column is the sample name, subsequent tab-separated
columns are files containing associated sequences, which may be assemblies or reads, and may be gzipped. For example::

    sample1    sample1.fa
    sample2    sample2.fa
    sample3    sample3_1.fq.gz     sample3_2.fq.gz

Calculate core and accessory distances between databases with ``--query``. If all-vs-all, only the upper triangle is calculated,
for example::

    poppunk_sketch --query --ref_db listeria.h5 --query_db listeria.h5 --cpus 4 > distances.txt

API
===

python
^^^^^^

Import the package and call commands. See ``pp_sketch/__main__.py``::

    import pp_sketchlib

    pp_sketchlib.constructDatabase(ref_db, names, sequences, kmers, int(round(sketch_size/64)), min_count, cpus)
    distMat = pp_sketchlib.queryDatabase(ref_db, ref_db, rList, qList, kmers, cpus)

    print(distMat)


C++
^^^

See ``main.cpp`` for examples::


    #include <fstream>
    #include <iostream>

    #include "reference.hpp"
    #include "database.hpp"
    #include "api.hpp"

    // Set k-mer lengths
    std::vector<size_t> kmer_lengths {15, 17, 19, 21, 23, 25, 27, 29};
    
    // Create a two sketches
    Reference ref(argv[1], {argv[2]}, kmer_lengths, 156, 0);
    Reference query(argv[3], {argv[4]}, kmer_lengths, 156, 0);

    // Output some distances at a single k-mer length
    std::cout << ref.jaccard_dist(query, 15) << std::endl;
    std::cout << ref.jaccard_dist(query, 29) << std::endl;

    // Calculate core and accessory distances between two sketches
    auto core_acc = ref.core_acc_dist(query); 
    std::cout << std::get<0>(core_acc) << "\t" << std::get<1>(core_acc) << std::endl;

    // Save sketches to file
    Database sketch_db("sketch.h5");
    sketch_db.add_sketch(ref);
    sketch_db.add_sketch(query);

    // Read sketches from file
    Reference ref_read = sketch_db.load_sketch(argv[1]);
    Reference query_read = sketch_db.load_sketch(argv[3]);

    // Create sketches using multiple threads, saving to file
    std::vector<Reference> ref_sketches = create_sketches("full",
                               {argv[1], argv[3]}, 
                               {{argv[2]}, {argv[4]}}, 
                               kmer_lengths,
                               32,
                               0,
                               2);

    // Calculate distances between sketches using multiple threads
    MatrixXf dists = query_db(ref_sketches,
                              ref_sketches,
                              kmer_lengths,
                              2);

    std::cout << dists << std::endl;

    
    // Read sketches from an existing database, using random access
    HighFive::File h5_db("listeria.h5");
    Database listeria_db(h5_db);
    std::vector<Reference> listeria_sketches;
    for (auto name_it = names.cbegin(); name_it != names.cend(); name_it++)
    {
        listeria_sketches.push_back(listeria_db.load_sketch(*name_it));
    }


Citations
=========
The overall method was described in the following paper:

Lees JA, Harris SR, Tonkin-Hill G, Gladstone RA, Lo SW, Weiser JN, Corander J, Bentley SD, Croucher NJ. Fast and flexible
bacterial genomic epidemiology with PopPUNK. *Genome Research* **29**:1-13 (2019).
doi:`10.1101/gr.241455.118 <https://dx.doi.org/10.1101/gr.241455.118>`__

This extension uses parts of the following methods, and in some cases their code:

| *bindash* (written by XiaoFei Zhao):
| Zhao, X. BinDash, software for fast genome distance estimation on a typical personal laptop. 
*Bioinformatics* **35**:671–673 (2019). `doi:10.1093/bioinformatics/bty651 <https://dx.doi.org/10.1093/bioinformatics/bty651>`__

| *ntHash* (written by Hamid Mohamadi):
| Mohamadi, H., Chu, J., Vandervalk, B. P. & Birol, I. ntHash: recursive nucleotide hashing. 
*Bioinformatics* **32**:3492–3494 (2016). `doi:10.1093/bioinformatics/btw397 <https://dx.doi.org/10.1093/bioinformatics/btw397>`__

| *countmin* (similar to that used in the khmer library, written by the Lab for Data Intensive Biology at UC Davis):
| Zhang, Q., Pell, J., Canino-Koning, R., Howe, A. C. & Brown, C. T. 
These are not the k-mers you are looking for: efficient online k-mer counting using a probabilistic data structure. 
PLoS One 9, e101271 (2014). `doi:10.1371/journal.pone.0101271 <https://doi.org/10.1371/journal.pone.0101271>`__