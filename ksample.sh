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
# python3 ./gradient_ksample_test.py --test DCORR --data dmap --label 2-sample_normed --n-perms 10000 --norm

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

# DCORR SVD raw embeddings, 073 excluded, normed
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/svd_raw_aligned_12-02 --test DCORR --data svd --label raw_aligned --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/svd_raw_aligned_12-02 --test DCORR --data svd --label raw_aligned --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/svd_raw_aligned_12-02 --test DCORR --data svd --label raw_aligned --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/svd_raw_aligned_12-02 --test DCORR --data svd --label raw_aligned --n-perms 10000 --norm

# DCORR SVD dmap embeddings, 073 excluded, normed
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/svd_dmap_aligned_12-02 --test DCORR --data svd --label 6-sample_dmap --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/svd_dmap_aligned_12-02 --test DCORR --data svd --label 3-sample-novices_damp --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/svd_damp_aligned_12-02 --test DCORR --data svd --label 3-sample-experts_damp --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/svd_damp_aligned_12-02 --test DCORR --data svd --label 2-sample_damp --n-perms 10000 --norm

#####################################################################################################################################################################################################
########### Main Comparisons ########################################################################################################################################################################

# DCORR gcca raw (ZG 3, min rank)
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/gcca_09-22-21:18_min_rank-ZG3_exclude-073 --test DCORR --data gcca --label raw_ZG3-min_normed --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/gcca_09-22-21:18_min_rank-ZG3_exclude-073 --test DCORR --data gcca --label raw_ZG3-min_normed --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/gcca_09-22-21:18_min_rank-ZG3_exclude-073 --test DCORR --data gcca --label raw_ZG3-min_normed --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/gcca_09-22-21:18_min_rank-ZG3_exclude-073 --test DCORR --data gcca --label raw_ZG3-min_normed --n-perms 10000 --norm
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_gcca_raw_ZG3-min_normed/ --save /home/rflperry/meditation/data/DCORR_gcca_raw_ZG3-min_normed/DCORR_gcca_raw_ZG3-min_normed.pdf


# DCORR gcca dmap (ZG 2, min rank)
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/gcca_dmap_ZG2_12-04 --test DCORR --data gcca --label dmap_ZG2_normed --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/gcca_dmap_ZG2_12-04 --test DCORR --data gcca --label dmap_ZG2_normed --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/gcca_dmap_ZG2_12-04 --test DCORR --data gcca --label dmap_ZG2_normed --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/gcca_dmap_ZG2_12-04 --test DCORR --data gcca --label dmap_ZG2_normed --n-perms 10000 --norm
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_gcca_dmap_ZG2-min_normed/ --save /home/rflperry/meditation/data/DCORR_gcca_dmap_ZG2-min_normed/DCORR_gcca_dmap_ZG2-min_normed.pdf


# DCORR gcca dmap (ZG 2, min rank, grads 1:4)
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/gcca_dmap_ZG2_12-04 --test DCORR --data gcca --label dmap_ZG2_grads=1-4_normed --n-perms 1000 --k-sample 6 --norm --start-grad 1
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/gcca_dmap_ZG2_12-04 --test DCORR --data gcca --label dmap_ZG2_grads=1-4_normed --n-perms 10000 --k-sample 3N --norm --start-grad 1
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/gcca_dmap_ZG2_12-04 --test DCORR --data gcca --label dmap_ZG2_grads=1-4_normed --n-perms 10000 --k-sample 3E --norm --start-grad 1
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/gcca_dmap_ZG2_12-04 --test DCORR --data gcca --label dmap_ZG2_grads=1-4_normed --n-perms 10000 --norm --start-grad 1
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_gcca_dmap_ZG2_grads=1-4_normed/ --save /home/rflperry/meditation/data/DCORR_gcca_dmap_ZG2_grads=1-4_normed/DCORR_gcca_dmap_ZG2_grads=1-4_normed.pdf

# Mapalign, aligned
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_antoinne_aligned_onepass_09-04 --test DCORR --data dmap --label antoinne_aligned_onepass_normed --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_antoinne_aligned_onepass_09-04 --test DCORR --data dmap --label antoinne_aligned_onepass_normed --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_antoinne_aligned_onepass_09-04 --test DCORR --data dmap --label antoinne_aligned_onepass_normed --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_antoinne_aligned_onepass_09-04 --test DCORR --data dmap --label antoinne_aligned_onepass_normed --n-perms 10000 --norm
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_dmap_antoinne_aligned_onepass_normed/ --save /home/rflperry/meditation/data/DCORR_dmap_antoinne_aligned_onepass_normed/DCORR_dmap_antoinne_aligned_onepass_normed.pdf

# SVD raw
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/svd_raw_aligned_12-07 --test DCORR --data svd --label raw_aligned_normed --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/svd_raw_aligned_12-07 --test DCORR --data svd --label raw_aligned_normed --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/svd_raw_aligned_12-07 --test DCORR --data svd --label raw_aligned_normed --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/svd_raw_aligned_12-07 --test DCORR --data svd --label raw_aligned_normed --n-perms 10000 --norm
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_svd_raw_aligned_normed/ --save /home/rflperry/meditation/data/DCORR_svd_raw_aligned_normed/DCORR_svd_raw_aligned_normed.pdf

# SVD raw, mean-aligned
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/svd_raw_mean-aligned_12-07 --test DCORR --data svd --label raw_mean-aligned_normed --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/svd_raw_mean-aligned_12-07 --test DCORR --data svd --label raw_mean-aligned_normed --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/svd_raw_mean-aligned_12-07 --test DCORR --data svd --label raw_mean-aligned_normed --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/svd_raw_mean-aligned_12-07 --test DCORR --data svd --label raw_mean-aligned_normed --n-perms 10000 --norm
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_svd_raw_mean-aligned_normed/ --save /home/rflperry/meditation/data/DCORR_svd_raw_mean-aligned_normed/DCORR_svd_raw_mean-aligned_normed.pdf

# SVD dmap (1st component, constant, not saved)
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/svd_dmap_aligned_12-02 --test DCORR --data svd --label dmap_aligned_normed --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/svd_dmap_aligned_12-02 --test DCORR --data svd --label dmap_aligned_normed --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/svd_dmap_aligned_12-02 --test DCORR --data svd --label dmap_aligned_normed --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/svd_dmap_aligned_12-02 --test DCORR --data svd --label dmap_aligned_normed --n-perms 10000 --norm
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_svd_dmap_aligned_normed/ --save /home/rflperry/meditation/data/DCORR_svd_dmap_aligned_normed/DCORR_svd_dmap_aligned_normed.pdf

# Mapalign, mean aligned
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_mean-aligned_09-04 --test DCORR --data dmap --label mapalign_mean-aligned_normed --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_mean-aligned_09-04 --test DCORR --data dmap --label mapalign_mean-aligned_normed --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_mean-aligned_09-04 --test DCORR --data dmap --label mapalign_mean-aligned_normed --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_mean-aligned_09-04 --test DCORR --data dmap --label mapalign_mean-aligned_normed --n-perms 10000 --norm
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_dmap_mapalign_mean-aligned_normed/ --save /home/rflperry/meditation/data/DCORR_dmap_mapalign_mean-aligned_normed/DCORR_dmap_mapalign_mean-aligned_normed.pdf

# Mapalign replicated, mean aligned
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication_mean-aligned_12-04 --test DCORR --data dmap --label mapalign_replication_mean-aligned_normed --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication_mean-aligned_12-04 --test DCORR --data dmap --label mapalign_replication_mean-aligned_normed --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication_mean-aligned_12-04 --test DCORR --data dmap --label mapalign_replication_mean-aligned_normed --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication_mean-aligned_12-04 --test DCORR --data dmap --label mapalign_replication_mean-aligned_normed --n-perms 10000 --norm
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_dmap_mapalign_replication_mean-aligned_normed/ --save /home/rflperry/meditation/data/DCORR_dmap_mapalign_replication_mean-aligned_normed/DCORR_dmap_mapalign_replication_mean-aligned_normed.pdf

# Mapalign replicated, aligned, row centered
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication_aligned_onepass_12-04 --test DCORR --data dmap --label mapalign_replication_aligned_onepass_normed --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication_aligned_onepass_12-04 --test DCORR --data dmap --label mapalign_replication_aligned_onepass_normed --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication_aligned_onepass_12-04 --test DCORR --data dmap --label mapalign_replication_aligned_onepass_normed --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication_aligned_onepass_12-04 --test DCORR --data dmap --label mapalign_replication_aligned_onepass_normed --n-perms 10000 --norm
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_dmap_mapalign_replication_aligned_onepass_normed/ --save /home/rflperry/meditation/data/DCORR_dmap_mapalign_replication_aligned_onepass_normed/DCORR_dmap_mapalign_replication_aligned_onepass_normed.pdf

# Dmap, group-svd
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_antoine_group-svd_09-04 --test DCORR --data dmap --label antoine_group-svd_normed --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_antoine_group-svd_09-04 --test DCORR --data dmap --label antoine_group-svd_normed --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_antoine_group-svd_09-04 --test DCORR --data dmap --label antoine_group-svd_normed --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_antoine_group-svd_09-04 --test DCORR --data dmap --label antoine_group-svd_normed --n-perms 10000 --norm
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_dmap_antoine_group-svd_normed/ --save /home/rflperry/meditation/data/DCORR_dmap_antoine_group-svd_normed/DCORR_dmap_antoine_group-svd_normed.pdf

# mapalign joint embedding normed
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/joint_mapalign_01-04/ --test DCORR --data joint --label mapalign_normed --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/joint_mapalign_01-04/ --test DCORR --data joint --label mapalign_normed --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/joint_mapalign_01-04/ --test DCORR --data joint --label mapalign_normed --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/joint_mapalign_01-04/ --test DCORR --data joint --label mapalign_normed --n-perms 10000 --norm
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_joint_mapalign_normed/ --save /home/rflperry/meditation/data/DCORR_joint_mapalign_normed/DCORR_joint_mapalign_normed.pdf

# Mapalign replicated, aligned
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication_align-100_01-04 --test DCORR --data dmap --label mapalign_replication_align-100_normed --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication_align-100_01-04 --test DCORR --data dmap --label mapalign_replication_align-100_normed --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication_align-100_01-04 --test DCORR --data dmap --label mapalign_replication_align-100_normed --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication_align-100_01-04 --test DCORR --data dmap --label mapalign_replication_align-100_normed --n-perms 10000 --norm
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_dmap_mapalign_replication_align-100_normed/ --save /home/rflperry/meditation/data/DCORR_dmap_mapalign_replication_align-100_normed/DCORR_dmap_mapalign_replication_align-100_normed.pdf

# MASE (scaled) o aff
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/mase_aff_scaled_01-07 --test DCORR --data mase --label aff_scaled_normed --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/mase_aff_scaled_01-07 --test DCORR --data mase --label aff_scaled_normed --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/mase_aff_scaled_01-07 --test DCORR --data mase --label aff_scaled_normed --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/mase_aff_scaled_01-07 --test DCORR --data mase --label aff_scaled_normed --n-perms 10000 --norm
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_mase_aff_scaled_normed/ --save /home/rflperry/meditation/data/DCORR_mase_aff_scaled_normed/DCORR_mase_aff_scaled_normed.pdf

# Mapalign replicated, aligned
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_aligned_12-04 --test DCORR --data dmap --label mapalign_replication-csv_aligned_normed --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_aligned_12-04 --test DCORR --data dmap --label mapalign_replication-csv-aligned_normed --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_aligned_12-04 --test DCORR --data dmap --label mapalign_replication-csv-aligned_normed --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_aligned_12-04 --test DCORR --data dmap --label mapalign_replication-csv-aligned_normed --n-perms 10000 --norm
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_dmap_mapalign_replication-csv_aligned_normed/ --save /home/rflperry/meditation/data/DCORR_dmap_mapalign_replication-csv_aligned_normed/DCORR_dmap_mapalign_replication-csv_aligned_normed.pdf

# Mapalign replicated from csv, aligned
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_align-100_12-04 --test DCORR --data dmap --label mapalign_replication-csv_align-100_normed --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_align-100_12-04 --test DCORR --data dmap --label mapalign_replication-csv_align-100_normed --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_align-100_12-04 --test DCORR --data dmap --label mapalign_replication-csv_align-100_normed --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_align-100_12-04 --test DCORR --data dmap --label mapalign_replication-csv_align-100_normed --n-perms 10000 --norm
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_dmap_mapalign_replication-csv_align-100_normed/ --save /home/rflperry/meditation/data/DCORR_dmap_mapalign_replication-csv_align-100_normed/DCORR_dmap_mapalign_replication_align-100_normed.pdf

# MASE o aff
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/mase_aff_01-05 --test DCORR --data mase --label aff_normed --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/mase_aff_01-05 --test DCORR --data mase --label aff_normed --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/mase_aff_01-05 --test DCORR --data mase --label aff_normed --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/mase_aff_01-05 --test DCORR --data mase --label aff_normed --n-perms 10000 --norm
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_mase_aff_normed/ --save /home/rflperry/meditation/data/DCORR_mase_aff_normed/DCORR_mase_aff_normed.pdf

# MASE o dmap
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/mase_dmap_01-08 --test DCORR --data mase --label dmap_normed --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/mase_dmap_01-08 --test DCORR --data mase --label dmap_normed --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/mase_dmap_01-08 --test DCORR --data mase --label dmap_normed --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/mase_dmap_01-08 --test DCORR --data mase --label dmap_normed --n-perms 10000 --norm
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_mase_dmap_normed/ --save /home/rflperry/meditation/data/DCORR_mase_dmap_normed/DCORR_mase_dmap_normed.pdf

# Mapalign replicated from csv, aligned-100 v2
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_align-100_01-09 --test DCORR --data dmap --label mapalign_replication-csv_align-100_01-09_normed --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_align-100_01-09 --test DCORR --data dmap --label mapalign_replication-csv_align-100_01-09_normed --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_align-100_01-09 --test DCORR --data dmap --label mapalign_replication-csv_align-100_01-09_normed --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_align-100_01-09 --test DCORR --data dmap --label mapalign_replication-csv_align-100_01-09_normed --n-perms 10000 --norm
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_dmap_mapalign_replication-csv_align-100_01-09_normed/ --save /home/rflperry/meditation/data/DCORR_dmap_mapalign_replication-csv_align-100_01-09_normed/DCORR_dmap_mapalign_replication-csv_align-100_01-09_normed.pdf

# Mapalign replicated from csv, aligned-5
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_align-5_01-09 --test DCORR --data dmap --label mapalign_replication-csv_align-5_01-09 --n-perms 1000 --k-sample 6
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_align-5_01-09 --test DCORR --data dmap --label mapalign_replication-csv_align-5_01-09 --n-perms 10000 --k-sample 3N
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_align-5_01-09 --test DCORR --data dmap --label mapalign_replication-csv_align-5_01-09 --n-perms 10000 --k-sample 3E
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_align-5_01-09 --test DCORR --data dmap --label mapalign_replication-csv_align-5_01-09 --n-perms 10000
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_dmap_mapalign_replication-csv_align-5_01-09/ --save /home/rflperry/meditation/data/DCORR_dmap_mapalign_replication-csv_align-5_01-09/DCORR_dmap_mapalign_replication_align-5_01-09.pdf

# Eig of dmap (non normed eig), aligned
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/svd_dmap-eigs_aligned_01-13 --test DCORR --data svd --label dmap-eigs_01-13_normed --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/svd_dmap-eigs_aligned_01-13 --test DCORR --data svd --label dmap-eigs_01-13_normed --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/svd_dmap-eigs_aligned_01-13 --test DCORR --data svd --label dmap-eigs_01-13_normed --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/svd_dmap-eigs_aligned_01-13 --test DCORR --data svd --label dmap-eigs_01-13_normed --n-perms 10000 --norm
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_svd_dmap-eigs_01-13_normed/ --save /home/rflperry/meditation/data/DCORR_svd_dmap-eigs_01-13_normed/DCORR_svd_dmap-eigs_01-13_normed.pdf

# mapalign joint embedding
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/joint_mapalign_01-04/ --test DCORR --data joint --label mapalign --n-perms 1000 --k-sample 6
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/joint_mapalign_01-04/ --test DCORR --data joint --label mapalign --n-perms 10000 --k-sample 3N
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/joint_mapalign_01-04/ --test DCORR --data joint --label mapalign --n-perms 10000 --k-sample 3E
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/joint_mapalign_01-04/ --test DCORR --data joint --label mapalign --n-perms 10000
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_joint_mapalign/ --save /home/rflperry/meditation/data/DCORR_joint_mapalign/DCORR_joint_mapalign.pdf

# Mapalign replicated from csv, aligned-5
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_mean-align-5_01-09 --test DCORR --data dmap --label mapalign_replication-csv_mean-align-5_01-09 --n-perms 1000 --k-sample 6
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_mean-align-5_01-09 --test DCORR --data dmap --label mapalign_replication-csv_mean-align-5_01-09 --n-perms 10000 --k-sample 3N
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_mean-align-5_01-09 --test DCORR --data dmap --label mapalign_replication-csv_mean-align-5_01-09 --n-perms 10000 --k-sample 3E
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_mean-align-5_01-09 --test DCORR --data dmap --label mapalign_replication-csv_mean-align-5_01-09 --n-perms 10000
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_dmap_mapalign_replication-csv_mean-align-5_01-09/ --save /home/rflperry/meditation/data/DCORR_dmap_mapalign_replication-csv_mean-align-5_01-09/DCORR_dmap_mapalign_replication_mean-align-5_01-09.pdf

# Mapalign replicated svd align
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_svd-align_01-09 --test DCORR --data dmap --label mapalign_replication-csv_svd-align_01-09 --n-perms 1000 --k-sample 6
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_svd-align_01-09 --test DCORR --data dmap --label mapalign_replication-csv_svd-align_01-09 --n-perms 10000 --k-sample 3N
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_svd-align_01-09 --test DCORR --data dmap --label mapalign_replication-csv_svd-align_01-09 --n-perms 10000 --k-sample 3E
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_svd-align_01-09 --test DCORR --data dmap --label mapalign_replication-csv_svd-align_01-09 --n-perms 10000
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_dmap_mapalign_replication-csv_svd-align_01-09/ --save /home/rflperry/meditation/data/DCORR_dmap_mapalign_replication-csv_svd-align_01-09/DCORR_dmap_mapalign_replication_svd-align_01-09.pdf

# Mapalign replicated svd align, normed
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_svd-align_01-09 --test DCORR --data dmap --label mapalign_replication-csv_svd-align_normed_01-09 --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_svd-align_01-09 --test DCORR --data dmap --label mapalign_replication-csv_svd-align_normed_01-09 --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_svd-align_01-09 --test DCORR --data dmap --label mapalign_replication-csv_svd-align_normed_01-09 --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/dmap_replication-csv_svd-align_01-09 --test DCORR --data dmap --label mapalign_replication-csv_svd-align_normed_01-09 --n-perms 10000 --norm
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_dmap_mapalign_replication-csv_svd-align_normed_01-09/ --save /home/rflperry/meditation/data/DCORR_dmap_mapalign_replication-csv_svd-align_normed_01-09/DCORR_dmap_mapalign_replication_svd-align_normed_01-09.pdf

# grouppca, normed
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/grouppca_raw_02-25 --test DCORR --data grouppca --label raw_normed_02-25 --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/grouppca_raw_02-25 --test DCORR --data grouppca --label raw_normed_02-25 --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/grouppca_raw_02-25 --test DCORR --data grouppca --label raw_normed_02-25 --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/grouppca_raw_02-25 --test DCORR --data grouppca --label raw_normed_02-25 --n-perms 10000 --norm
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_grouppca_raw_normed_02-25/ --save /home/rflperry/meditation/data/DCORR_grouppca_raw_normed_02-25/DCORR_grouppca_raw_normed_02-25.pdf

# grouppca, magnitudes
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/grouppca_raw_02-25 --test DCORR --data grouppca --label raw_02-25 --n-perms 1000 --k-sample 6
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/grouppca_raw_02-25 --test DCORR --data grouppca --label raw_02-25 --n-perms 10000 --k-sample 3N
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/grouppca_raw_02-25 --test DCORR --data grouppca --label raw_02-25 --n-perms 10000 --k-sample 3E
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/grouppca_raw_02-25 --test DCORR --data grouppca --label raw_02-25 --n-perms 10000
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_grouppca_raw_02-25/ --save /home/rflperry/meditation/data/DCORR_grouppca_raw_02-25/DCORR_grouppca_raw_02-25.pdf

# gcca n_components=5, normed
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/gcca_raw_ncomp-5_scaled_02-27 --test DCORR --data gcca --label raw_ncomp-5_scaled_normed_02-27 --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/gcca_raw_ncomp-5_scaled_02-27 --test DCORR --data gcca --label raw_ncomp-5_scaled_normed_02-27 --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/gcca_raw_ncomp-5_scaled_02-27 --test DCORR --data gcca --label raw_ncomp-5_scaled_normed_02-27 --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/gcca_raw_ncomp-5_scaled_02-27 --test DCORR --data gcca --label raw_ncomp-5_scaled_normed_02-27 --n-perms 10000 --norm
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_gcca_raw_ncomp-5_scaled_normed_02-27/ --save /home/rflperry/meditation/data/DCORR_gcca_raw_ncomp-5_scaled_normed_02-27/DCORR_gcca_raw_ncomp-5_scaled_normed_02-27.pdf

# gcca n_components=5
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/gcca_raw_ncomp-5_scaled_02-27 --test DCORR --data gcca --label raw_ncomp-5_scaled_02-27 --n-perms 1000 --k-sample 6
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/gcca_raw_ncomp-5_scaled_02-27 --test DCORR --data gcca --label raw_ncomp-5_scaled_02-27 --n-perms 10000 --k-sample 3N
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/gcca_raw_ncomp-5_scaled_02-27 --test DCORR --data gcca --label raw_ncomp-5_scaled_02-27 --n-perms 10000 --k-sample 3E
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/gcca_raw_ncomp-5_scaled_02-27 --test DCORR --data gcca --label raw_ncomp-5_scaled_02-27 --n-perms 10000
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_gcca_raw_ncomp-5_scaled_02-27/ --save /home/rflperry/meditation/data/DCORR_gcca_raw_ncomp-5_scaled_02-27/DCORR_gcca_raw_ncomp-5_scaled_02-27.pdf

# grouppca, prewhitened and normed
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/grouppca_raw_prewhitened_02-26   --test DCORR --data grouppca --label raw_prewhitened_normed_02-26 --n-perms 1000 --k-sample 6 --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/grouppca_raw_prewhitened_02-26   --test DCORR --data grouppca --label raw_prewhitened_normed_02-26 --n-perms 10000 --k-sample 3N --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/grouppca_raw_prewhitened_02-26   --test DCORR --data grouppca --label raw_prewhitened_normed_02-26 --n-perms 10000 --k-sample 3E --norm
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/grouppca_raw_prewhitened_02-26   --test DCORR --data grouppca --label raw_prewhitened_normed_02-26 --n-perms 10000 --norm
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_grouppca_raw_prewhitened_normed_02-26/ --save /home/rflperry/meditation/data/DCORR_grouppca_raw_prewhitened_normed_02-26/DCORR_grouppca_raw_prewhitened_normed_02-26.pdf

# grouppca, prewhitened
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/grouppca_raw_prewhitened_02-26   --test DCORR --data grouppca --label raw_prewhitened_02-26 --n-perms 1000 --k-sample 6
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/grouppca_raw_prewhitened_02-26   --test DCORR --data grouppca --label raw_prewhitened_02-26 --n-perms 10000 --k-sample 3N
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/grouppca_raw_prewhitened_02-26   --test DCORR --data grouppca --label raw_prewhitened_02-26 --n-perms 10000 --k-sample 3E
# python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/grouppca_raw_prewhitened_02-26   --test DCORR --data grouppca --label raw_prewhitened_02-26 --n-perms 10000
# python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_grouppca_raw_prewhitened_02-26/ --save /home/rflperry/meditation/data/DCORR_grouppca_raw_prewhitened_02-26/DCORR_grouppca_raw_prewhitened_02-26.pdf

# template aligned
python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/template_align --test DCORR --data dmap2 --label template_align_normed --n-perms 1000 --k-sample 6 --exclude-ids 073 --norm
python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/template_align --test DCORR --data dmap2 --label template_align_normed --n-perms 10000 --k-sample 3N --exclude-ids 073 --norm
python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/template_align --test DCORR --data dmap2 --label template_align_normed --n-perms 10000 --k-sample 3E --exclude-ids 073 --norm
python3 ./gradient_ksample_test.py --source /mnt/ssd3/ronan/data/template_align --test DCORR --data dmap2 --label template_align_normed --n-perms 10000 --exclude-ids 073 --norm
python3 ./make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_dmap2_template_align_normed/ --save /home/rflperry/meditation/data/DCORR_dmap2_template_align_normed/DCORR_dmap2_template_align_normed.pdf
python make_pvalue_heatmap.py --source /home/rflperry/meditation/data/DCORR_dmap2_template_align_normed/ --save /home/rflperry/meditation/data/DCORR_dmap2_template_align_normed/magnitudes_dmap2_template_align_normed.pdf --magnitudes

cd ../