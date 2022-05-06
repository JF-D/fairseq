
NAME=trans_wmt_replace_sample
CHECKPOINT_DIR=checkpoints/en-de-base-replace-sample
RESTORE="--restore-file checkpoints/en-de-base-test/checkpoint25.pt"
TENSORBOARD="--tensorboard-logdir log/tensorboard/${NAME}"

#--batch-size 32 --required-seq-len-multiple 128
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -m torch.distributed.launch --nproc_per_node 8 train.py data-bin/wmt16_en_de_bpe32k \
      --arch transformer_wmt_en_de --share-all-embeddings \
      --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
      --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
      --lr 0.0007 --stop-min-lr 1e-09 \
      --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
      --max-tokens 4096 \
      --no-progress-bar --log-format json --log-interval 50 \
      --save-interval 1 --save-interval-updates 3000 --keep-interval-updates 20 \
      --save-dir ${CHECKPOINT_DIR} ${RESTORE} ${TENSORBOARD} \
      --update-freq 1 | tee log/${NAME}.log
