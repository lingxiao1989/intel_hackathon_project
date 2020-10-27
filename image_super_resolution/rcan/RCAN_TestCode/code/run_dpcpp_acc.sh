#!/bin/bash

DIR_DATA='/home/zpan5/rcan_data'
PRE_TRAIN='/home/zpan5/gpu-optimized-models/rcan/models_ECCV2018RCAN/RCAN_BIX2.pt'
RESULT='./log'

accuracy_cpu_fp32 () {
       echo -e '--------------------------------------'
       echo -e 'RCAN CPU inference wich FP32'
       echo -e '--------------------------------------'

       python main.py --model RCAN_ACC \
                     --save RCAN_BIX2_G10R20P48 \
                     --scale 2 \
                     --n_resgroups 5 \
                     --n_resblocks 8 \
                     --n_feats 64  \
                     --reset \
                     --chop \
                     --dir_data $DIR_DATA \
                     --cpu \
                     --n_threads 0 \
                     --pre_train $PRE_TRAIN \
                     --batch_size 1 \
                     --test_only \
                     --n_val 2
}

accuracy_dpcpp_fp32 () {
       echo -e '--------------------------------------'
       echo -e 'RCAN DPCPP inference wich FP32'
       echo -e '--------------------------------------'
       export IPEX_LAZY_REORDER=1
       export IPEX_WEIGHT_CACHE=1
       export ForceBcsCacheFlushOnAtsFamily=1
       export EnableBlitterOperationsSupport=0
       export RenderCompressedBuffersEnabled=0
       python main.py --model RCAN_ACC \
                     --save RCAN_BIX2_G10R20P48 \
                     --scale 2 \
                     --n_resgroups 5 \
                     --n_resblocks 8 \
                     --n_feats 64  \
                     --reset \
                     --chop \
                     --dir_data $DIR_DATA \
                     --cpu \
                     --n_threads 0 \
                     --pre_train $PRE_TRAIN \
                     --batch_size 1 \
                     --test_only \
                     --n_val 2 \
                     --sycl \
                     --enable-jit 1

}

accuracy_dpcpp_fp16 () {
       echo -e '--------------------------------------'
       echo -e 'RCAN DPCPP inference wich FP16'
       echo -e '--------------------------------------'
       export IPEX_LAZY_REORDER=1
       export IPEX_WEIGHT_CACHE=1
       export ForceBcsCacheFlushOnAtsFamily=1
       export EnableBlitterOperationsSupport=0
       export RenderCompressedBuffersEnabled=0
       python main.py --model RCAN_ACC \
                     --save RCAN_BIX2_G10R20P48 \
                     --scale 2 \
                     --n_resgroups 5 \
                     --n_resblocks 8 \
                     --n_feats 64  \
                     --reset \
                     --chop \
                     --dir_data $DIR_DATA \
                     --cpu \
                     --n_threads 0 \
                     --pre_train $PRE_TRAIN \
                     --batch_size 1 \
                     --test_only \
                     --n_val 2 \
                     --sycl \
                     --enable-jit 1 \
                     --precision half
}

main () {
       echo -e '--------------------------------------'
       echo -e 'RCAN Accuracy'
       echo -e '--------------------------------------'
       accuracy_cpu_fp32
       accuracy_dpcpp_fp32
       accuracy_dpcpp_fp16
}

main
