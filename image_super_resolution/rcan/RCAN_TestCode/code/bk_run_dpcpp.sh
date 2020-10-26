#!/bin/bash

DIR_DATA='/home/huang/release/rcan'
PRE_TRAIN='/home/huang/release/rcan/models_ECCV2018RCAN/RCAN_BIX2.pt'
RESULT='./log'

inference_with_fp32 () {
       echo -e '--------------------------------------'
       echo -e 'RCAN inference wich FP32'
       echo -e '--------------------------------------'

export DNNL_VERBOSE=1
export IPEX_LAZY_REORDER=1
export IPEX_WEIGHT_CACHE=1
       python -u main.py \
       --data_test MyImage \
       --scale 2 \
       --model RCAN \
       --n_resgroups 10 \
       --n_resblocks 20 \
       --n_feats 64 \
       --dir_data $DIR_DATA \
       --pre_train $PRE_TRAIN \
       --test_only \
       --save_results \
       --chop \
       --save 'RCAN' \
       --testpath ../LR/LRBI \
       --testset Set6 \
       --cpu \
       --n_thread 0 \
       --enable-jit 1 \
       --sycl \
       2>&1 | tee $RESULT/fp32.log
}

inference_with_fp16 () {
       echo -e '--------------------------------------'
       echo -e 'RCAN inference wich FP16'
       echo -e '--------------------------------------'

#export DNNL_VERBOSE=1
export IPEX_LAZY_REORDER=1
export IPEX_WEIGHT_CACHE=1
       python -u main.py \
       --data_test MyImage \
       --scale 2 \
       --model RCAN \
       --n_resgroups 5 \
       --n_resblocks 8 \
       --n_feats 64 \
       --dir_data $DIR_DATA \
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
       2>&1 | tee $RESULT/fp16.log
}

inference_with_bf16 () {
       echo -e '--------------------------------------'
       echo -e 'RCAN inference wich BF16'
       echo -e '--------------------------------------'

#export DNNL_VERBOSE=1
#export IPEX_LAZY_REORDER=1
#export IPEX_WEIGHT_CACHE=1
       python -u main.py \
       --data_test MyImage \
       --scale 2 \
       --model RCAN \
       --n_resgroups 5 \
       --n_resblocks 8 \
       --n_feats 64 \
       --dir_data $DIR_DATA \
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
       2>&1 | tee $RESULT/bf16.log
}

jit () {

echo -e '--------------------------------------'
echo -e 'RCAN inference wich JIT'
echo -e '--------------------------------------'
export DNNL_VERBOSE=1
export IPEX_LAZY_REORDER=1
export IPEX_WEIGHT_CACHE=1
python main.py --model RCAN \
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
               --n_val 3 \
               --sycl \
               --enable-jit 1

}

main () {
       echo -e '--------------------------------------'
       echo -e 'RCAN'
       echo -e '--------------------------------------'
       inference_with_fp32
       # inference_with_fp16
       #inference_with_bf16
       #jit
}

main
