# Commands to run from the current directory for various results. Uncomment and call script to run.
cd ./scripts
# DCORR SIMULATIONS to validate distribution under the null is uniform in relevant 2-sample, k-sample cases.
# python3 ./scripts/gradient_ksample_test.py --test DCORR --simulate --label gaussian --n-perms 1000 --n-datasets 500 --sim-dim 100
# python3 ./scripts/gradient_ksample_test.py --test DCORR --simulate --label 6-sample --n-perms 10000= --k-sample 6 --n-datasets 500 --sim-dim 100
# python3 ./scripts/gradient_ksample_test.py --test DCORR --simulate --label 3-sample-novices --n-perms 1000 --k-sample 3N --n-datasets 500 --sim-dim 100
# python3 ./scripts/gradient_ksample_test.py --test DCORR --simulate --label 3-sample-experts --n-perms 1000 --k-sample 3E --n-datasets 500 --sim-dim 100

# DCORR REAL 2-sample, k-sample All
# python3 ./scripts/gradient_ksample_test.py --test DCORR --label 6-sample --n-perms 10000 --k-sample 6
# python3 ./scripts/gradient_ksample_test.py --test DCORR --label 3-sample-novices --n-perms 10000 --k-sample 3N
# python3 ./scripts/gradient_ksample_test.py --test DCORR --label 3-sample-experts --n-perms 10000 --k-sample 3E

# DCORR REAL 2-sample, k-sample exlude subject 073
# python3 ./gradient_ksample_test.py --test DCORR --label 2-sample-073_exclude --n-perms 10000 --exclude-ids 073

# DCORR 
python3 ./gradient_ksample_test.py --test DCORR --label dmap_2-sample-073_exclude --n-perms 10000 --exclude-ids 073


# 
cd ../