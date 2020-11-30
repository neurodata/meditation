# Commands to run from the current directory for various results. Uncomment and call script to run.
cd ./scripts
# DCORR SIMULATIONS to validate distribution under the null is uniform in relevant 2-sample, k-sample cases.
# python3 ./gradient_ksample_test.py --test DCORR --simulate --label gaussian --n-perms 1000 --n-datasets 500 --sim-dim 18715
# python3 ./gradient_ksample_test.py --test DCORR --simulate --label 6-sample --n-perms 10000 --k-sample 6 --n-datasets 500 --sim-dim 100
# python3 ./gradient_ksample_test.py --test DCORR --simulate --label 3-sample-novices --n-perms 1000 --k-sample 3N --n-datasets 500 --sim-dim 100
# python3 ./gradient_ksample_test.py --test DCORR --simulate --label 3-sample-experts --n-perms 1000 --k-sample 3E --n-datasets 500 --sim-dim 100

# DCORR REAL 2-sample, k-sample All
# python3 ./gradient_ksample_test.py --test DCORR --label 2-sample --n-perms 10000
# python3 ./gradient_ksample_test.py --test DCORR --label 6-sample --n-perms 10000 --k-sample 6
# python3 ./gradient_ksample_test.py --test DCORR --label 3-sample-novices --n-perms 10000 --k-sample 3N
# python3 ./gradient_ksample_test.py --test DCORR --label 3-sample-experts --n-perms 10000 --k-sample 3E

# DCORR REAL diffusion embedding
# python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 2-sample --n-perms 10000 #--exclude-ids 073
# python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 6-sample --n-perms 10000 --k-sample 6 #--exclude-ids 073
# python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 3-sample-novices --n-perms 10000 --k-sample 3N #--exclude-ids 073
# python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 3-sample-experts --n-perms 10000 --k-sample 3E #--exclude-ids 073

# DCORR REAL diffusion map separate aligns, 073 excluded from multi-align directory
# python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 2-sample_multi-align_norm-False --n-perms 10000 --align
# python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 6-sample --n-perms 10000 --k-sample 6 #--exclude-ids 073
# python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 3-sample-novices_multi-align --n-perms 10000 --k-sample 3N --align
# python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 3-sample-experts_multi-align --n-perms 10000 --k-sample 3E --align

# DCORR REAL diffusion map separate aligns, 073 excluded from multi-align directory, no norm
# python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 2-sample_multi-align_norm-False --n-perms 10000 --align
# python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 6-sample_norm-False --n-perms 10000 --k-sample 6 --norm-off
# python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 3-sample-novices_multi-align_norm-False --n-perms 10000 --k-sample 3N --align --norm-off
# python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 3-sample-experts_multi-align_norm-False --n-perms 10000 --k-sample 3E --align --norm-off

# Current Testing
# python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 2-sample_normed --n-perms 1000
# python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 2-sample_aligned --n-perms 2000
# python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 2-sample_multi-align_norm-False --n-perms 1000 --align --norm-off

# dcorr k-sample real data
# python3 ./gradient_ksample_test.py --test DCORR --label 6-sample --n-perms 10000 --k-sample 6
# python3 ./gradient_ksample_test.py --test DCORR --label 6-sample_multiway --n-perms 10000 --k-sample 6 --multiway

# Manova real data, computationally intractable i think due to covariance matrix size
# python3 ./gradient_ksample_test.py --test Manova --label 2-sample --n-perms 10000
# python3 ./gradient_ksample_test.py --test Manova --label 6-sample --n-perms 10000 --k-sample 6
# python3 ./gradient_ksample_test.py --test Manova --label 3-sample-novices --n-perms 10000 --k-sample 3N
# python3 ./gradient_ksample_test.py --test Manova --label 3-sample-experts --n-perms 10000 --k-sample 3E

# DCORR REAL diffusion map, 073 excluded, normed
# python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 6-sample_normed --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 3-sample-novices_normed --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 3-sample-experts_normed --n-perms 10000 --k-sample 3E --norm
python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 2-sample_normed --n-perms 10000 --norm

# DCORR REAL diffusion map with means aligned, 073 excluded, no norm
# python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 6-sample_mean-align --n-perms 1000 --k-sample 6
# python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 3-sample-novices_mean-align --n-perms 10000 --k-sample 3N
# python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 3-sample-experts_mean-align --n-perms 10000 --k-sample 3E
# python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 2-sample_mean-align --n-perms 10000

# DCORR REAL diffusion map with means aligned, 073 excluded, norm
# python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 6-sample_mean-align_normed --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 3-sample-novices_mean-align_normed --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 3-sample-experts_mean-align_normed --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 2-sample_mean-align_normed --n-perms 10000 --norm

# DCORR MASE embeddings, 073 excluded
# python3 ./gradient_ksample_test.py --test DCORR --data mase --label 6-sample_dmap_normed --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --test DCORR --data mase --label 3-sample-novices_dmap__normed --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --test DCORR --data mase --label 3-sample-experts_dmap__normed --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --test DCORR --data mase --label 2-sample_dmap__normed --n-perms 10000 --norm

# DCORR GCCA dmap embeddings, 073 excluded
# python3 ./gradient_ksample_test.py --test DCORR --data mase --label 6-sample_dmap_normed --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --test DCORR --data mase --label 3-sample-novices_dmap__normed --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --test DCORR --data mase --label 3-sample-experts_dmap__normed --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --test DCORR --data mase --label 2-sample_dmap__normed --n-perms 10000 --norm

# DCORR GCCA dmap embeddings, 073 excluded
# python3 ./gradient_ksample_test.py --test DCORR --data gcca --label 6-sample_dmap --n-perms 1000 --k-sample 6
# python3 ./gradient_ksample_test.py --test DCORR --data gcca --label 3-sample-novices_dmap --n-perms 10000 --k-sample 3N
# python3 ./gradient_ksample_test.py --test DCORR --data gcca --label 3-sample-experts_dmap --n-perms 10000 --k-sample 3E
# python3 ./gradient_ksample_test.py --test DCORR --data gcca --label 2-sample_dmap --n-perms 10000

cd ../