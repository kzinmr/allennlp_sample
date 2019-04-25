# Obtain training data from http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
export BIDIRECTIONAL_LM_DATA_PATH=$PWD'/1-billion-word-language-modeling-benchmark-r13output'
export BIDIRECTIONAL_LM_TRAIN_PATH=$BIDIRECTIONAL_LM_DATA_PATH'/training-monolingual.tokenized.shuffled/*'
export BIDIRECTIONAL_LM_VOCAB_PATH=$PWD'/vocabulary'
# The multiprocess dataset reader and iterator use many file descriptors,
# so we increase the relevant ulimit here to help.
# See https://pytorch.org/docs/stable/multiprocessing.html#file-descriptor-file-descriptor
# for a description of the underlying issue.

ulimit -n 4096
allennlp train bidirectional_language_model.jsonnet --serialization-dir outputs
