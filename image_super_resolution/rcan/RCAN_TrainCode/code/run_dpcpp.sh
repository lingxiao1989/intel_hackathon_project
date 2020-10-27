#!/bin/bash

DIR_DATA='../rcan'
PRE_TRAIN='../rcan/models_ECCV2018RCAN/RCAN_BIX2.pt'
RESULT='./log'

inference_with_fp32 () {
       echo -e '--------------------------------------'
       echo -e 'RCAN inference wich FP32'
       echo -e '--------------------------------------'

       python -u main.py \
       --model RCAN \
       --save RCAN_BIX2_G10R20P48 \
       --scale 2 \
       --n_resgroups 5 \
       --n_resblocks 8 \
       --n_feats 64  \
       --reset \
       --chop \
       --dir_data $DIR_DATA \
       --cpu --n_threads 0 \
       --pre_train $PRE_TRAIN \
       --batch_size 1 \
       --test_only \
       --n_val 5 \
       --sycl \
       2>&1 | tee $RESULT/fp32.log
}

inference_with_fp16 () {
       echo -e '--------------------------------------'
       echo -e 'RCAN inference wich FP16'
       echo -e '--------------------------------------'

       python -u main.py \
       --model RCAN \
       --save RCAN_BIX2_G10R20P48 \
       --scale 2 \
       --n_resgroups 5 \
       --n_resblocks 8 \
       --n_feats 64  \
       --reset \
       --chop \
       --dir_data $DIR_DATA \
       --cpu --n_threads 0 \
       --pre_train $PRE_TRAIN \
       --batch_size 1 \
       --test_only \
       --n_val 5 \
       --sycl \
       --precision half \
       2>&1 | tee $RESULT/fp16.log
}

inference_with_bf16 () {
       echo -e '--------------------------------------'
       echo -e 'RCAN inference wich BF16'
       echo -e '--------------------------------------'

       python -u main.py \
       --model RCAN \
       --save RCAN_BIX2_G10R20P48 \
       --scale 2 \
       --n_resgroups 5 \
       --n_resblocks 8 \
       --n_feats 64  \
       --reset \
       --chop \
       --dir_data $DIR_DATA \
       --cpu --n_threads 0 \
       --pre_train $PRE_TRAIN \
       --batch_size 1 \
       --test_only \
       --n_val 5 \
       --sycl \
       --precision bfloat16 \
       2>&1 | tee $RESULT/bf16.log
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
