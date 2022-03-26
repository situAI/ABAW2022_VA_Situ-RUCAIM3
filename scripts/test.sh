set -e

# ------ Modify before running -------------
gpu_ids=0
test_checkpoints="
transformer/transformer_both_affectnet-vggish-wav2vec_bs16_lr5e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run1/2;
transformer-lstm/transformer_lstm_both_affectnet-vggish-wav2vec_res-y_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run1/2
"
test_target='arousal'
# ------------------------------------------

cmd="python test.py --test_log_dir=./test_results
--checkpoints_dir=./checkpoints
--test_target=$test_target --test_checkpoints='$test_checkpoints' --gpu_ids=$gpu_ids"

echo "-------------------------------------------------------------------------------------"
echo $cmd | sh
exit

# test_checkpoints="
# path_after_checkpoints_dir/prefix;
# path_after_checkpoints_dir/prefix
# "

# prefix: pth file name before `_net_xxx.pth` (correspoding to the best eval epoch on your test_target, you can find in your log files)

# bash scripts/test.sh
