

Demos for a few HyperVec functionalities
=====================================


demo_auto_tune.py
-----------------

Demonstrates the auto-tuning functionality of HyperVec


demo_ondisk_ivf.py
------------------

Shows how to construct a HyperVec index that stores the inverted file
data on disk, eg. when it does not fit in RAM. The script works on a
small dataset (sift1M) for demonstration and proceeds in stages:

0: train on the dataset

1-4: build 4 indexes, each containing 1/4 of the dataset. This can be
done in parallel on several machines

5: merge the 4 indexes into one that is written directly to disk
(needs not to fit in RAM)

6: load and test the index
