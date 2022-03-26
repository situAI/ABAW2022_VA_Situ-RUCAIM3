set -e

# ------ Modify before running -------------
name='submit1'
test_results="
transformer/transformer_both_affectnet-vggish-wav2vec_bs16_lr5e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run1;
transformer-lstm/transformer_lstm_both_affectnet-vggish-wav2vec_res-y_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run1
"
test_target='arousal'
smoothed='y'
# ------------------------------------------

cmd="python get_preds.py --test_log_dir=./test_results
--test_target=$test_target --submit_dir=./submit
--name=$name --test_results='$test_results'"

echo "-------------------------------------------------------------------------------------"
echo $cmd | sh
exit

# smoothed: 'y' or 'n'

# bash scripts/get_preds.sh