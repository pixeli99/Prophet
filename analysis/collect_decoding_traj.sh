

CUDA_VISIBLE_DEVICES=0 python collect_decoding_traj_gsm8k.py --decode_policy low_confidence \
                            --constraint_policy none \
                            --blocklen 32 \
                            --gen_length 256 \
                            --range_lst 0 1319 


CUDA_VISIBLE_DEVICES=1 python collect_decoding_traj_gsm8k.py --decode_policy low_confidence \
                            --constraint_policy constraint \
                            --blocklen 32 \
                            --gen_length 256 \
                            --range_lst 0 1319 


CUDA_VISIBLE_DEVICES=2 python collect_decoding_traj_gsm8k.py --decode_policy random \
                            --constraint_policy none \
                            --blocklen 256 \
                            --gen_length 256 \
                            --range_lst 0 1319 


CUDA_VISIBLE_DEVICES=3 python collect_decoding_traj_gsm8k.py --decode_policy random \
                            --constraint_policy constraint \
                            --blocklen 256 \
                            --gen_length 256 \
                            --range_lst 0 1319 
