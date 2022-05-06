TEXT=examples/translation/wmt16_en_de
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train.bpe \
    --validpref $TEXT/val.bpe \
    --testpref $TEXT/test.bpe \
    --destdir data-bin/wmt16_en_de_bpe32k \
    --nwordssrc 32768 --nwordstgt 32768 \
    --joined-dictionary \
    --workers 20
