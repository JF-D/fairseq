LOG=log/gen_replace.out

CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/wmt16_en_de_bpe32k \
    --path checkpoints/en-de-base-replace-sample/checkpoint95.pt \
    --beam 4 --lenpen 0.6 --remove-bpe > ${LOG}

# Compute BLEU score
grep ^H ${LOG} | cut -f3- > ${LOG}.sys
grep ^T ${LOG} | cut -f2- > ${LOG}.ref
fairseq-score --sys ${LOG}.sys --ref ${LOG}.ref
