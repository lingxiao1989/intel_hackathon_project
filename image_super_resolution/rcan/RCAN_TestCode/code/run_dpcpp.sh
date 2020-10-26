#!/bin/bash
export IPEX_LAZY_REORDER=1
export IPEX_WEIGHT_CACHE=1
DIR_DATA='/home/zpan5/rcan_data'
PRE_TRAIN='/home/zpan5/gpu-optimized-models/rcan/models_ECCV2018RCAN/RCAN_BIX2.pt'
RESULT='./log'

inference_with_fp32 () {
       echo -e '--------------------------------------'
       echo -e 'RCAN inference wich FP32'
       echo -e '--------------------------------------'

       python -u main.py \
       --data_test MyImage \
       --scale 2 \
       --model RCAN \
       --n_resgroups 5 \
       --n_resblocks 8 \
       --n_feats 64 \
       --pre_train $PRE_TRAIN \
       --test_only \
       --chop \
       --save 'RCAN' \
       --testpath ../LR/LRBI \
       --testset Set6 \
       --cpu \
       --n_thread 0 \
       --sycl \
       2>&1 | tee fp32.log
}

inference_with_fp16 () {
       echo -e '--------------------------------------'
       echo -e 'RCAN inference wich FP16'
       echo -e '--------------------------------------'

       python -u main.py \
       --data_test MyImage \
       --scale 2 \
       --model RCAN \
       --n_resgroups 5 \
       --n_resblocks 8 \
       --n_feats 64 \
       --pre_train $PRE_TRAIN \
       --test_only \
       --save_results \
       --chop \
       --save 'RCAN' \
       --testpath ../LR/LRBI \
       --testset Set6 \
       --cpu \
       --n_thread 0 \
       --sycl \
       --precision half \
       2>&1 | tee fp16.log
}

inference_with_bf16 () {
       echo -e '--------------------------------------'
       echo -e 'RCAN inference wich BF16'
       echo -e '--------------------------------------'

       python -u main.py \
       --data_test MyImage \
       --scale 2 \
       --model RCAN \
       --n_resgroups 5 \
       --n_resblocks 8 \
       --n_feats 64 \
       --pre_train $PRE_TRAIN \
       --test_only \
       --save_results \
       --chop \
       --save 'RCAN' \
       --testpath ../LR/LRBI \
       --testset Set6 \
       --cpu \
       --n_thread 0 \
       --sycl \
       --precision bfloat16 \
       2>&1 | tee bf16.log
}

main () {
       echo -e '--------------------------------------'
       echo -e 'RCAN'
       echo -e '--------------------------------------'
       inference_with_fp32
       inference_with_fp16
       inference_with_bf16
}

main
