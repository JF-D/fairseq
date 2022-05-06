
NAME=trans_wmt_test
CHECKPOINT_DIR=checkpoints/en-de-base-test
TENSORBOARD= #"--tensorboard-logdir log/tensorboard/${NAME}"

RESTORE="--restore-file checkpoints/en-de-base/checkpoint0.pt"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -m torch.distributed.launch --nproc_per_node 8 train.py data-bin/wmt16_en_de_bpe32k \
      --arch transformer_wmt_en_de --share-all-embeddings \
      --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
      --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
      --lr 0.0007 --stop-min-lr 1e-09 \
      --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
      --max-tokens 4096 --batch-size 32 --required-seq-len-multiple 128 \
      --no-progress-bar --log-format json --log-interval 50 \
      --save-dir ${CHECKPOINT_DIR} ${RESTORE} ${TENSORBOARD} \
      --update-freq 1 | tee log/${NAME}.log

# # ep 0
# RESTORE="--restore-file checkpoints/en-de-base-test/checkpoint0.pt"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -m torch.distributed.launch --nproc_per_node 8 train.py data-bin/wmt16_en_de_bpe32k \
#       --arch transformer_wmt_en_de --share-all-embeddings \
#       --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#       --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
#       --lr 0.0007 --stop-min-lr 1e-09 \
#       --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
#       --max-tokens 4096 --batch-size 32 --required-seq-len-multiple 128 \
#       --no-progress-bar --log-format json --log-interval 50 \
#       --save-dir ${CHECKPOINT_DIR} ${RESTORE} ${TENSORBOARD} \
#       --update-freq 1 | tee log/${NAME}.log


# # ep 25
# RESTORE="--restore-file checkpoints/en-de-base-test/checkpoint25.pt"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -m torch.distributed.launch --nproc_per_node 8 train.py data-bin/wmt16_en_de_bpe32k \
#       --arch transformer_wmt_en_de --share-all-embeddings \
#       --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#       --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
#       --lr 0.0007 --stop-min-lr 1e-09 \
#       --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
#       --max-tokens 4096 --batch-size 32 --required-seq-len-multiple 128 \
#       --no-progress-bar --log-format json --log-interval 50 \
#       --save-dir ${CHECKPOINT_DIR} ${RESTORE} ${TENSORBOARD} \
#       --update-freq 1 | tee log/${NAME}.log

# # ep 50
# RESTORE="--restore-file checkpoints/en-de-base-test/checkpoint50.pt"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -m torch.distributed.launch --nproc_per_node 8 train.py data-bin/wmt16_en_de_bpe32k \
#       --arch transformer_wmt_en_de --share-all-embeddings \
#       --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#       --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
#       --lr 0.0007 --stop-min-lr 1e-09 \
#       --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
#       --max-tokens 4096 --batch-size 32 --required-seq-len-multiple 128 \
#       --no-progress-bar --log-format json --log-interval 50 \
#       --save-dir ${CHECKPOINT_DIR} ${RESTORE} ${TENSORBOARD} \

# # ep 75
# RESTORE="--restore-file checkpoints/en-de-base-test/checkpoint75.pt"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -m torch.distributed.launch --nproc_per_node 8 train.py data-bin/wmt16_en_de_bpe32k \
#       --arch transformer_wmt_en_de --share-all-embeddings \
#       --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#       --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
#       --lr 0.0007 --stop-min-lr 1e-09 \
#       --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
#       --max-tokens 4096 --batch-size 32 --required-seq-len-multiple 128 \
#       --no-progress-bar --log-format json --log-interval 50 \
#       --save-dir ${CHECKPOINT_DIR} ${RESTORE} ${TENSORBOARD} \

# # ep 100
# RESTORE="--restore-file checkpoints/en-de-base-test/checkpoint100.pt"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  python -m torch.distributed.launch --nproc_per_node 8 train.py data-bin/wmt16_en_de_bpe32k \
#       --arch transformer_wmt_en_de --share-all-embeddings \
#       --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#       --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
#       --lr 0.0007 --stop-min-lr 1e-09 \
#       --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
#       --max-tokens 4096 --batch-size 32 --required-seq-len-multiple 128 \
#       --no-progress-bar --log-format json --log-interval 50 \
#       --save-dir ${CHECKPOINT_DIR} ${RESTORE} ${TENSORBOARD} \
