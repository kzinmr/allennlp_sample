mkdir vocabulary
export BIDIRECTIONAL_LM_VOCAB_PATH=$PWD'/vocabulary'
cd $BIDIRECTIONAL_LM_VOCAB_PATH
aws --no-sign-request s3 cp s3://allennlp/models/elmo/vocab-2016-09-10.txt .
cat vocab-2016-09-10.txt | sed 's/<UNK>/@@UNKNOWN@@/' > tokens.txt
# Avoid creating garbage namespace.
rm vocab-2016-09-10.txt
echo '*labels\n*tags' > non_padded_namespaces.txt
