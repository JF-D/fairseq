LOG=log/eval/gen_mimic_act_ep0_p5.out
CHECKPOINT=checkpoints/en-de-base-mimic-act-multi-ep0-p5

# N=5
# NAME=checkpoint.avg$N.pt
# python scripts/average_checkpoints.py \
#     --inputs $CHECKPOINT \
#     --num-epoch-checkpoints $N \
#     --output $CHECKPOINT/$NAME

NAME=checkpoint20.pt
CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/wmt16_en_de_bpe32k \
    --path $CHECKPOINT/$NAME \
    --beam 4 --lenpen 0.6 --remove-bpe > ${LOG}

# Compute BLEU score
grep ^H ${LOG} | cut -f3- > ${LOG}.sys
grep ^T ${LOG} | cut -f2- > ${LOG}.ref
fairseq-score --sys ${LOG}.sys --ref ${LOG}.ref
