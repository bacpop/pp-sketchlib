# pp-sketchlib <img src='sketchlib_logo.png' align="right" height="139" />

<!-- badges: start -->
[![Build status](https://dev.azure.com/jlees/pp-sketchlib/_apis/build/status/johnlees.pp-sketchlib?branchName=master)](https://dev.azure.com/jlees/pp-sketchlib/_build/latest?definitionId=1&branchName=master)
[![Anaconda package](https://anaconda.org/conda-forge/pp-sketchlib/badges/version.svg)](https://anaconda.org/conda-forge/pp-sketchlib)
<!-- badges: end -->


Library of sketching functions used by [PopPUNK](https://www.poppunk.net>).

## Installation
Install using conda (recommended):

```
conda install -c conda-forge pp-sketchlib
```

**NOTE**
    If you are getting `UnsatisfiableError` or similar version conflicts try following the
    [tips on conda-forge](https://conda-forge.org/docs/user/tipsandtricks.html#using-multiple-channels>).
    It may also help if you downgrade your version of conda (to 4.5). Installing into
    a new environment is recommended.

Or install locally:

```
python setup.py install
```

For this option you will need (all available through conda):

- a C++14 compiler (`GCC >=7.2.0` or `Clang`)
- `CMake (>=3.18)`
- `pybind11`
- `hdf5`
- `highfive`
- `Eigen` (>=v3.0)
- `armadillo`

If you wish to compile the GPU code you will also need the CUDA toolkit
installed (tested on 10.2 and 11.0).

Or install through pip

You need to have suitable system dependencies installed.  On ubuntu, this suffices:

```
apt-get update && apt-get install -y --no-install-recommends \
  cmake gfortran libarmadillo-dev libeigen3-dev libopenblas-dev
```

For reasons that are not yet clear, make sure that you have a copy of pybind11 available:

```
pip3 install --user pybind11
```

At which point you can install pp-sketchlib:

```
pip3 install --user pp-sketchlib
```

## Usage
Create a set of sketches and save these as a database:

```
poppunk_sketch --sketch --rfile rfiles.txt --ref-db listeria --sketch-size 10000 --cpus 4 --min-k 15 --k-step 2
```

The input file `rfiles.txt` has one sequence per line. The first column is the sample name, subsequent tab-separated
columns are files containing associated sequences, which may be assemblies or reads, and may be gzipped. For example:

```
sample1    sample1.fa
sample2    sample2.fa
sample3    sample3_1.fq.gz     sample3_2.fq.gz
```

Calculate core and accessory distances between databases with `--query`. If all-vs-all, only the upper triangle is calculated,
for example:

```
poppunk_sketch --query --ref_db listeria --query_db listeria --cpus 4
```

This will save output files as a database for use with PopPUNK. If you wish to output the
distances add the `--print` option:

```
poppunk_sketch --query --ref_db listeria --query_db listeria --cpus 4 --print > distances.txt
```

### Other options

Sketching:

- `--strand` ignores reverse complement k-mers, if input is all in the same sense
- `--min-count` minimum k-mer count to include when using reads
- `--exact-counter` uses a hash table to count k-mers, which is recommended for non-bacterial datasets.

Query:

- To only use some of the samples in the sketch database, you can add the `--subset` option with a file which lists the required sample names.
- `--jaccard` will output the Jaccard distances, rather than core and accessory distances.

### Large datasets

When working with large datasets, you can increase the `--cpus` to high numbers and get
a roughly proportional performance increase.

For calculating sketches of read datasets, or large numbers of distances, and you have a CUDA compatible GPU,
you can calculate distances on your graphics device even more quickly. Add the `--use-gpu` option:

```
poppunk_sketch --sketch --rfile rfiles.txt --ref-db listeria --cpus 4 --use-gpu
poppunk_sketch --query --ref-db listeria --query-db listeria --use-gpu
```

Both CPU parallelism and the GPU will be used, so be sure to add
both `--cpus` and `--use-gpu` for maximum speed. This is particularly efficient
when sketching.

You can set the `--gpu-id` if you have more than one device, which may be necessary on
cluster systems. This mode can also benefit from having multiple CPU cores available too.

### Benchmarks

Sketching 31610 ~3Mb *L. monocytogenes* genomes takes around 20 minutes.
Calculating all core/accessory distances (500M pairs) takes a further 14 minutes
on a CPU node, or 2 minutes on a GPU. Assigning new queries is twice as fast.

| Mode        | Parallelisation | Speed                           |
|-------------|-----------------|---------------------------------|
| Sketching   | CPU             | 26 genomes per second           |
| Read sketch | CPU             | 1.2 genomes per minute          |
|             | CPU & GPU       | 49 genomes per minute           |
| Distances   | CPU             | 170k-1600k distances per second |
|             | GPU             | 6000k distances per second      |

CPU tested using 16 cores on a Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz.
GPU tested using an NVIDIA RTX 2080 Ti GPU (4352 CUDA cores @ 1.35GHz).

NB: The distance speeds can be increased (linearly) by decreasing number of
k-mers or number of bins. The values above are for eight k-mer lengths
and 10000 bins.

**NOTE**
    If your results are slower than this you may wish to compile the library
    yourself. The version on conda is optimised for portability over speed,
    and we have observed greater speeds compiling with `--march=native`,
    which will automatically be used with `python setup.py install`.

## API

### python

Import the package and call commands. See `pp_sketch/__main__.py`

```
import pp_sketchlib

pp_sketchlib.constructDatabase(ref_db, names, sequences, kmers, int(round(sketch_size/64)),
strand_preserved, min_count, use_exact, cpus)
distMat = pp_sketchlib.queryDatabase(ref_db, ref_db, rList, qList, kmers,
jaccard, cpus, use_gpu, deviceid)

print(distMat)
```

Available functions:

- `constructDatabase()` - sketch genomes and save to a HDF5 database file (returns nothing).
- `queryDatabase()` - calculate distances between two database files (returns numpy array).
- `queryDatabaseSparse()` - as `queryDatabase()`, but only return distances exceeding a
  threshold, or nearest neighbours (return is a sparse COO matrix).
- `addRandomToDb()` - add a random match calculation to a database (returns nothing).
- `jaccardDist()` - Calculate a single jaccard distance between two samples in the same database
  (returns a floating point number).
- `squareToLong()` - Convert a square distance matrix to long form (returns numpy vector).
- `longToSquare()` - Convert a long form distance matrix to a symmetric square distance matrix (returns numpy array).
- `longToSquareMulti()` - Converts three long form distance matrices from ref-ref, query-query, query-ref comparisons
  into a single square distance matrix (returns a numpy array).
- `sparsifyDists()` - Convert a square distance matrix into a sparse matrix, by applying a
  distance threshold or number of nearest neighbours (returns a sparse COO matrix).

### hdf5 database files

Run `h5ls` on a database to see what groups it contains. Databases should always
contain `sketch` and may contain `random`. Run `h5dump` to see the full contents.
Contents are programmatically accessible with any HDF5 API. See `__main__.py` for an
example in python.

#### sketch

Attributes:

- `sketch_version` - version of sketching code used to create the database.
  The SHA1 hash of relevant code files (doesn't change with every commit).

Contains a group for each sample, within each has attributes:

- `base_freq` - frequency of A, C, G, T within the input sequence.
- `bbits` - bin bits as in bindash (hard-coded as 14).
- `k-mers` - k-mer lengths the sketch is at.
- `length` - sequence length. Exact if from an assembly, estimated using minhash
  if from reads.
- `missing_bases` - count of Ns.
- `sketchsize64` - number of bins/64, as in bindash.

And a dataset for each k-mer length, named as the k-mer length. Each dataset also
has the k-mer length stored as an attribute.

#### random

Attributes:

- `k_max` - maximum k-mer length (above this random match chance = 0).
- `k_min` - minimum k-mer length (below this will error).
- `use_rc` - using both strands?

Datasets:

- `centroids` - k-means centroids of base frequency clusters.
- `matches_keys` - k-mer lengths at which random match chances were calculated.
- `matches_values` - random match chances. Flattened matrices in the same order
  as the k-mer keys, and row-major across centroid pairs.
- `table_keys` - sample order of `table_values`.
- `table_values` - centroid ID assigned to each sample.

C++
---
I have yet to set up a proper namespace for this, but you can include this
code (`api.hpp` will do most functions) and use the parts you need. If you
are interested in this becoming more functional, please raise an issue.

See `main.cpp` for examples:

```
#include <fstream>
#include <iostream>

#include "reference.hpp"
#include "database.hpp"
#include "random_match.hpp"
#include "api.hpp"

// Set k-mer lengths
std::vector<size_t> kmer_lengths {15, 17, 19, 21, 23, 25, 27, 29};

// Create a two sketches
Reference ref(argv[1], {argv[2]}, kmer_lengths, 156, true, 0, false);
Reference query(argv[3], {argv[4]}, kmer_lengths, 156, true, 0, false);

// Use default random match chances
RandomMC random(true);

// Output some distances at a single k-mer length
std::cout << ref.jaccard_dist(query, 15, random) << std::endl;
std::cout << ref.jaccard_dist(query, 29, random) << std::endl;

// Calculate core and accessory distances between two sketches
auto core_acc = ref.core_acc_dist<RandomMC>(query, random);
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
                           156,
                           true,
                           0,
                           false,
                           2);
// Calculate distances between sketches using multiple threads
MatrixXf dists = query_db(ref_sketches,
                          ref_sketches,
                          kmer_lengths,
                          random,
                          false,
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
```

## Algorithms

### Sketching

1. Read in a sequence to memory. Whether a sequence is reads or not is determinedby the presence of quality scores. Count base composition and number of Ns.
2. Divide the range `[0, 2^64)` into equally sized bins (number of bins must be a multiple of 64).
3. If assemblies, roll through k-mers at each requested length using ntHash, producing
   64-bit hashes.
4. If reads, roll through k-mers as above, but also count occurences and only
   pass through those over the minimum count.
5. For each hash, assign it to the appropriate bin, and only store it there if lower than
   the current bin value.
6. After completing hashing, keep only the 14 least significant bits in each bin.
7. Apply the optimal densification function, taking values from adjacent bins
   iff any bins were not filled.
8. Take blocks of 64 bins, and transpose them into 14 64-bit integers.
9. The array of 64-bit integers is the sketch.

### Distances

1. For each k-mer length, iterate over the two arrays of 64-bit integers being compared.
2. Start with a mask of 64 ON bits.
3. Compute the XOR between the first two 64-bit integers (whether the first of the 14
   bin bits of the first 64 bins is identical).
4. Compute the AND between this and the mask, update this as the mask. The mask
   thereby keeps track of whether all bin bits in each bin were indentical.
5. After looping over 14 arrays, use POPCNT on the mask to calculate how many of
   the first 64 bins were identical.
6. The Jaccard distance is the proportion of identical bins over the total number
   of bins.
7. The core and accessory distance is calculated using simple linear regression of log(jaccard)
   on k-mer lengths. Core distance is `exp(gradient)`, accessory is `exp(intercept)`.

### Random match chance

To create the random match chances:

1. Take the base composition of all samples, and cluster using k-means.
2. For each cluster centroid, create five random genomes using repeated Bernoulli draws
   from the base frequencies at the centroid.
3. Choose maximum and minimum k-mer length based on where a Jaccard distance of 0 and 1
   would be expected with equal base frequencies.
4. For each k-mer length, at each pairwise combination of centroids (including self),
   sketch the random genomes and calculate the jaccard distances.

To adjust for random match chance:

1. Assign all samples to their closest k-means centroid by base-composition.
2. Find the pre-calculated random match chance between those two centroids.
3. Downweight the observed Jaccard distance using `|obs - random| / (1 - random)`

If pre-calculated random match chances have not been computed, the formula of
Blais & Blanchette is used (formula 6 in the paper cited below).

## Notes

- All matrix/array structures are row-major, for compatibility with numpy.
- GPU sketching is only supported for reads. If a mix of reads and assemblies,
  sketch each separately and join the databases.
- GPU sketching filters out any read containing an N, which may give slightly
  different results from the CPU code.
- GPU sketching with variable read lengths is untested, but theoretically supported.
- GPU distances use lower precision than the CPU code, so slightly different results
  are expected.

## Citations
The overall method was described in the following paper:

Lees JA, Harris SR, Tonkin-Hill G, Gladstone RA, Lo SW, Weiser JN, Corander J, Bentley SD, Croucher NJ. Fast and flexible
bacterial genomic epidemiology with PopPUNK. *Genome Research* **29**:1-13 (2019).
doi:[10.1101/gr.241455.118](https://dx.doi.org/10.1101/gr.241455.118)

This extension uses parts of the following methods, and in some cases their code (license included where required):

*bindash* (written by XiaoFei Zhao):\
Zhao, X. BinDash, software for fast genome distance estimation on a typical personal laptop.\
*Bioinformatics* **35**:671–673 (2019).\
doi:[10.1093/bioinformatics/bty651](https://dx.doi.org/10.1093/bioinformatics/bty651>)

*ntHash* (written by Hamid Mohamadi):\
Mohamadi, H., Chu, J., Vandervalk, B. P. & Birol, I. ntHash: recursive nucleotide hashing.\
*Bioinformatics* **32**:3492–3494 (2016).\
doi:[10.1093/bioinformatics/btw397](https://dx.doi.org/10.1093/bioinformatics/btw397>)

*countmin* (similar to that used in the khmer library, written by the Lab for Data Intensive Biology at UC Davis):\
Zhang, Q., Pell, J., Canino-Koning, R., Howe, A. C. & Brown, C. T.\
These are not the k-mers you are looking for: efficient online k-mer counting using a probabilistic data structure.\
PLoS One 9, e101271 (2014).\
doi:[10.1371/journal.pone.0101271](https://doi.org/10.1371/journal.pone.0101271>)

*CSRS*\
Blais, E. & Blanchette, M.\
Common Substrings in Random Strings.\
Combinatorial Pattern Matching 129–140 (2006).\
doi:[10.1007/11780441_13](https://doi.org/10.1007/11780441_13>)

## Building and testing notes (for developers)
Run `python setup.py build --debug` to build with debug flags on

You can set an environment variable `SKETCHLIB_INSTALL` to affect `python setup.py`:

- Empty: uses cmake
- `conda`: sets library location to the conda environment, and uses `src/Makefile` (used to be used in conda-forge recipe)
- `azure`: Uses `src/Makefile`

### cmake
Now requires v3.19. If nvcc version is 11.0 or higher, sm8.6 with device link time optimisation will be used.
Otherwise, code is generated for sm7.0 and sm7.5.

### make
See `src/Makefile`. Targets are:

- `all` (default): builds test executables `sketch_test`, `matrix_test`, `read_test` and `gpu_dist_test`
- `python`: builds the python extension, same as cmake
- `web`: builds the webassembly (requires `emcc` installed and activated)
- `install`: installs executables (don't use this)
- `python_install`: installs python extension
- `clean`: removes all intermediate build files and executables

Modifiers:

- `DEBUG=1` runs with debug flags
- `PROFILE=1` runs with profiler flags for `ncu` and `nsys`
- `GPU=1` also build CUDA code (assumes `/usr/local/cuda-11.1/` and SM v8.6)

### Test that Python can build an installable package

Build a python source package and install it into an empty docker container with vanilla python 3. If this works, then there's a good chance that the version uploaded to pypi will work

```
rm -rf dist
python3 setup.py sdist
docker run --rm -it -v "${PWD}:/src:ro" python:3 /src/docker/install
```

See [this PR](https://github.com/bacpop/pp-sketchlib/pull/70) for the sorts of things we're trying to work around here.

### Publish to pypi

If things are being weird, the test index can be useful:

```
python3 setup.py sdist
twine upload --repository testpypi dist/*
```

You can test installing this into an empty docker container with

```
docker run --rm -it --entrypoint bash python:3
apt-get update && apt-get install -y --no-install-recommends \
  cmake gfortran libarmadillo-dev libeigen3-dev libopenblas-dev
pip install -i https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  pp-sketchlib
```

It can take a few minutes for the new version to become available so you may want to do

```
pip install -i https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  pp-sketchlib==1.7.5.3
```

updated with your current version to force installation of the new one.

Once satisfied that pip/twine haven't uploaded a completely broken package (and typically once the PR is merged) upload to the main pypi.

```
twine upload dist/*
```
