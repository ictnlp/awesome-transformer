<p align="center"><img src="http://o9h9w1ysg.bkt.clouddn.com/blog/180905/ecLK7hbl8L.jpg"></p>

<p align="center">
<a href="https://raw.githubusercontent.com/SkyAndCloud/awesome-transformer/master/LICENSE"><img src="https://img.shields.io/cocoapods/l/Kingfisher.svg?style=flat"></a>
</p>

A collection of transformer's guides, implementations and so on(For those who want to do some research using transformer as a baseline or simply reproduce paper's performance).

Please feel free to pull requests or report issues.

## Why this project?

Transformer is a powerful model applied in sequence to sequence learning. However, when we were using transformer as our baseline in NMT research we found no good & reliable guide to reproduce approximate result as reported in original paper(even official <a href="#t2t">tensor2tensor</a> implementation), which means our research would be unauthentic. We collected some implementations, obtained corresponding performance-reproducable approaches and other materials, which eventually formed this project.

## Papers

### NMT Basic
- seq2seq model: [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- seq2seq & attention: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- refined attention: [Effective Approaches to Attention-based Neural Machine Translation](http://arxiv.org/abs/1508.04025)
- seq2seq using CGRU: [DL4MT](https://github.com/nyu-dl/dl4mt-tutorial)
- GNMT: [Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144)
- bytenet: [Neural Machine Translation in Linear Time](https://arxiv.org/abs/1610.10099)
- convolutional-style NMT: [Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122)
- bpe: [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- word piece: [Japanese and Korean Voice Search](https://ieeexplore.ieee.org/document/6289079/)
- self attention paper: [A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)

### Transformer original paper

<p align="center">
<a><img src="http://o9h9w1ysg.bkt.clouddn.com/blog/180904/41BagIeFk9.png"></a>
</p>

- [Attention is All You Need](https://arxiv.org/abs/1706.03762)

## Implementations & How to reproduce paper's result?

Indeed there are lots of transformer implementations on the Internet, in order to simplify learning curve, here we only include **the most valuable** projects.

>**[Note]**: In transformer original paper, there are *WMT14 English-German*, *WMT14 English-French* two results
    ![results](http://o9h9w1ysg.bkt.clouddn.com/blog/180904/B4mbH19CBD.png)
Here we regard a implementation as performance-reproducable **if there exists approaches to reproduce WMT14 English-German BLEU score**. Therefore, we'll also support corresponding approach to reproduce *WMT14 English-German* result.

### Minimal, paper-equavalent but not certainly performance-reproducable implementations(both *PyTorch* implementations)

1. attention-is-all-you-need-pytorch

    - [code](https://github.com/jadore801120/attention-is-all-you-need-pytorch)

2. Harvard NLP Group's annotation   

    - [code](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

### Complex, performance-reproducable implementations

Because transformer's original implementation should run on **8 GPU** to replicate corresponding result, where each GPU loads one batch and after forward propagation 8 batch's loss is summed to execute backward operation, so we can **accumulate every 8 batch's loss** to execute backward operation if we **only have 1 GPU** to imitate this process and so on. This trick is implemented in <a href="#accum_count">OpenNMT-py</a> `-accum_count`
    
We recommend using [sacrebleu](https://github.com/awslabs/sockeye/tree/master/contrib/sacrebleu), which should be equivalent to `mteval-v13a.pl` but more convenient,  to calculate bleu score and report the signature as `BLEU+case.mixed+lang.de-en+test.wmt17 = 32.97 66.1/40.2/26.6/18.1 (BP = 0.980 ratio = 0.980 hyp_len = 63134 ref_len = 64399)` for easy reproduction.
**Note that sacrebleu already has an inner-tokenizer, so the text should be untokenized version.**

The transformer paper's original model settings can be found in [tensor2tensor transformer.py](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py). For example, You can find `base model configs` in`transformer_base` function.

#### <a id="t2t"/>Paper's original implementation: tensor2tensor(using *TensorFlow*)

##### Code

- [tensor2tensor](https://github.com/tensorflow/tensor2tensor)

##### Code annotation

- [“变形金刚”为何强大：从模型到代码全面解析Google Tensor2Tensor系统](https://cloud.tencent.com/developer/article/1153079)(only Chinese version, corresponding to tensor2tensor v1.6.3)

##### Steps to reproduce WMT14 English-German result:

Note that this implementation's code doesn't have `-accum_count`-like feature as <a href="accum_count">OpenNMT-py</a>, so you can **either train on 8 GPUs or modify code to add this feature.**

```shell
# 1. Install tensor2tensor toolkit
pip install tensor2tensor

# 2. Basic config
# For BPE model use this problem
PROBLEM=translate_ende_wmt_bpe32k
MODEL=transformer
HPARAMS=transformer_base
# or use transformer_large to reproduce large model
# HPARAMS=transformer_large
DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# 3. Download and preprocess corpus
# Note that tensor2tensor has an inner tokenizer
t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM

# 4. Train on 8 GPUs. You'll get nearly expected performance after ~250k steps and certainly expected performance after ~500k steps.
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \ 
  --train_steps=600000

# 5. Translate
DECODE_FILE=$TMP_DIR/newstest2014.tok.bpe.32000.en
BEAM_SIZE=4
ALPHA=0.6

t2t-decoder \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DECODE_FILE \
  --decode_to_file=$TMP_DIR/newstest2014.en.tok.32kbpe.transformer_base.beam5.alpha0.6.decode

# 6. Debpe and detokenization for prediction.
cat $TMP_DIR/newstest2014.en.tok.32kbpe.transformer_base.beam5.alpha0.6.decode | sed 's/@@ //g' > $TMP_DIR/newstest2014.en.tok.32kbpe.transformer_base.beam5.alpha0.6.decode.debpe
#Do compound splitting on the translation
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $TMP_DIR/newstest2014.en.tok.32kbpe.transformer_base.beam5.alpha0.6.decode.debpe > $TMP_DIR/newstest2014.en.tok.32kbpe.transformer_base.beam5.alpha0.6.decode.debpe.atat
```

##### Resources
- [t2t issue 539](https://github.com/tensorflow/tensor2tensor/issues/539)
- [t2t issue 444](https://github.com/tensorflow/tensor2tensor/issues/444)
- [t2t issue 317](https://github.com/tensorflow/tensor2tensor/issues/317)
- [Tensor2Tensor for Neural Machine Translation](https://arxiv.org/abs/1803.07416)

#### Harvard NLP Group's implementation: OpenNMT-py(using *PyTorch*)

##### Code

- [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
        
##### Steps to reproduce WMT14 English-German result:

For command arguments meaning, see [OpenNMT-py doc](http://opennmt.net/OpenNMT-py/main.html) or [OpenNMT-py opts.py](https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/opts.py)

1. Download [corpus preprocessed by OpenNMT](https://s3.amazonaws.com/opennmt-trainingdata/wmt_ende_sp.tar.gz), [sentencepiece model preprocessed by OpenNMT](https://s3.amazonaws.com/opennmt-trainingdata/wmt_ende_sp_model.tar.gz). Note that the preprocess procedure includes tokenization, bpe/word-piece operation(here using [sentencepiece](https://github.com/google/sentencepiece) powered by Google which implements word-piece algorithm), see [OpenNMT-tf script](https://github.com/OpenNMT/OpenNMT-tf/blob/master/scripts/wmt/prepare_data.sh) for more details.
        
2. Preprocess. Make sure you use a sequence length of `100` and `-share_vocab`.
    For example:
    
    ```shell
    python preprocess.py 
        -train_src ../wmt-en-de/train.en.shuf 
        -train_tgt ../wmt-en-de/train.de.shuf 
        -valid_src ../wmt-en-de/valid.en 
        -valid_tgt ../wmt-en-de/valid.de 
        -save_data ../wmt-en-de/processed 
        -src_seq_length 100 
        -tgt_seq_length 100 
        -max_shard_size 200000000 
        -share_vocab
    ```
        
3. Train. For example, if you only have 4 GPU:
    ```shell
    python  train.py 
        -data /tmp/de2/data 
        -save_model /tmp/extra 
        -layers 6 
        -rnn_size 512 
        -word_vec_size 512 
        -transformer_ff 2048 
        -heads 8  
        -encoder_type transformer 
        -decoder_type transformer 
        -position_encoding 
        -train_steps 200000  
        -max_generator_batches 2 
        -dropout 0.1 
        -batch_size 4096 
        -batch_type tokens 
        -normalization tokens  
        -accum_count 2 
        -optim adam 
        -adam_beta2 0.998 
        -decay_method noam 
        -warmup_steps 8000
        -learning_rate 2 
        -max_grad_norm 0 
        -param_init 0  
        -param_init_glorot 
        -label_smoothing 0.1 
        -valid_steps 10000 
        -save_checkpoint_steps 10000 
        -gpuid 0 1 2 3 
    ```
    
    <a id="accum_count"/>Note that here `-accum_count` means every `N` batches accumulating loss to backward, so it's 2 for 4 GPUs and so on.
        
4. Translate. For example:
    ```shell
    python translate.py -gpu 0 -replace_unk -alpha 0.6 -beta 0.0 -beam_size 5 
    -length_penalty wu -coverage_penalty wu -share_vocab vocab_file -max_length 200 -src newstest2014.en.32kspe
    ```
        
    Note that testset in corpus preprocessed by OpenNMT is newstest2017 while it is newstest2014 in original paper, which may be a mistake. To obtain newstest2014 testset as in paper, here we can use sentencepiece to encode `newstest2014.en` manually. **Note that there must be sentencepiece installed.**
    ```shell
    spm_encode --model=<model_file> --output_format=piece < newstest2014.en > newstest2014.en.32kspe
    ```

    You can set `-batch_size`(default 30) larger to boost the translation.
        
5. Detokenization. Since training data is processed by [sentencepiece](https://github.com/google/sentencepiece), step 4's translation should be sentencepiece-encoded style, so we need a decoding procedure. 
    For example: 
    ```shell
    spm_decode --model=<model_file> --input_format=piece < input > output
    ```
        
    You can find `<model_file>`in step 1's downloaded archive.
    
    As you can see, [OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf/tree/master/scripts/wmt) also has a replicable instruction, actually we prefer <a href="#t2t">tensor2tensor</a> as a baseline to reproduce paper's result if we have to use TensorFlow because it is original.

##### Resources

- [OpenNMT-py issue](https://github.com/OpenNMT/OpenNMT-py/issues/637)
- [OpenNMT: Open-Source Toolkit for Neural Machine Translation](https://arxiv.org/abs/1701.02810)

#### FAIR's implementation: fairseq-py(using *PyTorch*)
        
##### Code

- [fairseq-py](https://github.com/pytorch/fairseq/)
        
##### Steps to reproduce WMT14 English-German result:

Note that this implementation's code doesn't have `-accum_count`-like feature as <a href="accum_count">OpenNMT-py</a>), so you can **either train on 8 GPUs or modify code to add this feature.**

- <a id="instruction">fairseq-py instruction(https://github.com/pytorch/fairseq/tree/master/examples/translation)</span>

##### Resources

- [fairseq-py issue](https://github.com/pytorch/fairseq/issues/202). The corpus problem described in the issue has been fixed now, so we can directly follow the <a href="#instruction">instruction</a>.

### Complex, not certainly performance-reproducable implementations

- [Marian](https://github.com/marian-nmt/marian-examples/tree/master/transformer)(purely c++ implementation without any deep learning framework)

## Training tips

- [Training Tips for the Transformer Model](https://arxiv.org/abs/1804.00247)

## Further

- [Transformer's latest papers](http://arxiv-sanity.com/search?q=transformer)
- RNMT+: [The Best of Both Worlds: Combining Recent Advances in Neural Machine Translation](https://arxiv.org/abs/1804.09849)
- Turing-complete Transformer: [Universal Transformer](https://arxiv.org/abs/1807.03819)

## Contributors

This project is developed and maintained by Natural Language Processing Group, ICT/CAS.

- [Yong Shan](https://github.com/SkyAndCloud)
- [Jinchao Zhang](https://github.com/zhangjcqq)
