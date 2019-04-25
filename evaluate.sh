
export BIDIRECTIONAL_LM_DATA_PATH=$PWD'/1-billion-word-language-modeling-benchmark-r13output'
export BIDIRECTIONAL_LM_TRAIN_PATH=$BIDIRECTIONAL_LM_DATA_PATH'/training-monolingual.tokenized.shuffled/*'
export BIDIRECTIONAL_LM_VOCAB_PATH=$PWD'/vocabulary'
export BIDIRECTIONAL_LM_ARCHIVE_PATH=$PWD'/outputs/model.tar.gz'

allennlp evaluate --cuda-device 0 -o '{"iterator": {"base_iterator": {"maximum_samples_per_batch": ["num_tokens", 500] }}}' $BIDIRECTIONAL_LM_ARCHIVE_PATH $BIDIRECTIONAL_LM_DATA_PATH/heldout-monolingual.tokenized.shuffled/news.en-00000-of-00100
