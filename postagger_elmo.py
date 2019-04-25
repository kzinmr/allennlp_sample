from typing import Iterator, List, Dict

import torch
import torch.optim as optim
import numpy as np

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField

from allennlp.data.dataset_readers import DatasetReader

from allennlp.common.file_utils import cached_path

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import Token
# from allennlp.data.tokenizers.word_splitter import WordSplitter


from allennlp.data.vocabulary import Vocabulary

from allennlp.models import Model

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder

from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from allennlp.training.metrics import CategoricalAccuracy, F1Measure  # FBetaMeasure

from allennlp.data.iterators import BucketIterator

from allennlp.training.trainer import Trainer

from allennlp.predictors import SentenceTaggerPredictor

torch.manual_seed(1)


class PosDatasetReader(DatasetReader):
    """
    DatasetReader for PoS tagging data, one sentence per line, like

        The###DET dog###NN ate###V the###DET apple###NN
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None
    ) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(
        self,
        tokens: List[Token],
        tags: List[str] = None
    ) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if tags:  # None for prediction
            label_field = SequenceLabelField(
                labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        max_seq_len = 200
        with open(file_path) as f:
            for line in f:
                pairs = line.strip().split()
                if pairs:
                    sentence, tags = zip(*(pair.split("###") for pair in pairs if len(pair.split("###")) == 2))
                    tags = ['1' if tag == '2' else '0' for tag in tags]
                    if tags and all(len(t) == 1 for t in tags) and sum(map(int, tags)) > 0:
                        tokens = [Token(word) for word in sentence][:max_seq_len]
                        tags = tags[:max_seq_len]
                        yield self.text_to_instance(tokens, tags)


class LstmTagger(Model):

    def __init__(
        self,
        word_embeddings: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        vocab: Vocabulary
    ) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels')
                                          )
        self.accuracy = CategoricalAccuracy()
        self.f1_measure = F1Measure(1)  # FBetaMeasure

    def forward(
        self,
        sentence: Dict[str, torch.Tensor],
        labels: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:

        mask = get_text_field_mask(sentence)  # mask to exclude the padding

        embeddings = self.word_embeddings(sentence)

        encoder_out = self.encoder(embeddings, mask)

        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}

        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            self.f1_measure(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(
                tag_logits, labels, mask)

        return output

    def get_metrics(
        self,
        reset: bool = False
    ) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset),
                "precision": self.f1_measure.get_metric(reset)[0],
                "recall": self.f1_measure.get_metric(reset)[1],
                "f1_measure": self.f1_measure.get_metric(reset)[2]}


if __name__ == "__main__":
    # Reading the Data ######
    token_indexer = ELMoTokenCharactersIndexer()

    reader = PosDatasetReader(token_indexers={"tokens": token_indexer})
    train_dataset = reader.read(cached_path('wikipedia_mecab_puncts/training.txt'))
    validation_dataset = reader.read(cached_path('wikipedia_mecab_puncts/validation.txt'))

    vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

    # Instantiating the Model ######
    # EMBEDDING_DIM = 6
    HIDDEN_DIM = 100

    weight_file = 'https://elmoja.blob.core.windows.net/elmoweights/weights.hdf5'
    options_file = 'https://elmoja.blob.core.windows.net/elmoweights/options.json'

    elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
    word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})

    lstm: Seq2SeqEncoder = PytorchSeq2SeqWrapper(
        torch.nn.LSTM(word_embeddings.get_output_dim(), HIDDEN_DIM, bidirectional=True, batch_first=True))

    model = LstmTagger(word_embeddings, lstm, vocab)

    # Train ######
    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # sort the instances by the number of tokens in the sentence field
    iterator = BucketIterator(batch_size=16,
                              sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      patience=10,
                      num_epochs=10,
                      cuda_device=cuda_device)
    trainer.train()

    # Predict ######
    predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
    tag_logits = predictor.predict("あらゆる 現実 を すべて 自分 の ほう へ ねじ曲げ た の だ")['tag_logits']

    tag_ids = np.argmax(tag_logits, axis=-1)
    # print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])

    # Here's how to save the model.
    with open("postagger-elmo_result/model.th", 'wb') as f:
        torch.save(model.state_dict(), f)
    vocab.save_to_files("postagger-elmo_result/vocabulary")

    # And here's how to reload the model.
    vocab2 = Vocabulary.from_files("postagger-elmo_result/vocabulary")
    model2 = LstmTagger(word_embeddings, lstm, vocab2)
    with open("postagger-elmo_result/model.th", 'rb') as f:
        model2.load_state_dict(torch.load(f))
    if cuda_device > -1:
        model2.cuda(cuda_device)
    predictor2 = SentenceTaggerPredictor(model2, dataset_reader=reader)
    tag_logits2 = predictor2.predict("あらゆる 現実 を すべて 自分 の ほう へ ねじ曲げ た の だ")['tag_logits']

    np.testing.assert_array_almost_equal(tag_logits2, tag_logits)
