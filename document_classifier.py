from typing import List, Dict, Callable, Optional, Iterator, Iterable, Any

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.special import softmax
from tqdm import tqdm

from allennlp.data import Instance
from allennlp.data.fields import TextField, MetadataField, ArrayField

from allennlp.data.dataset_readers import DatasetReader

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from allennlp.data.vocabulary import Vocabulary

from allennlp.models import Model

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding

from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask, move_to_device

from allennlp.training.metrics import BooleanAccuracy, F1Measure

from allennlp.training.trainer import Trainer
from allennlp.data.iterators import BucketIterator, DataIterator, BasicIterator

from pathlib import Path

text_col = 'wakati_jumanpp'
label_cols = ['true_class_0',
              'true_class_1001', 'true_class_1002', 'true_class_1003', 'true_class_2001',
              'true_class_2002', 'true_class_2003', 'true_class_2004', 'true_class_2005', 'true_class_2006',
              'true_class_2007']

torch.manual_seed(1)

EMBEDDING_DIM = 300
HIDDEN_DIM = 100
MAX_SEQ_LEN = 200
N_EPOCHS = 1
TESTING = True


def tonp(tsr): return tsr.detach().cpu().numpy()


class DocumentClassifierPredictor:
    def __init__(self, model: Model, iterator: DataIterator,
                 cuda_device: int = -1) -> None:
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device

    def _extract_data(self, batch) -> np.ndarray:
        out_dict = self.model(**batch)
        return softmax(tonp(out_dict["class_logits"]), axis=1)

    def predict(self, ds: Iterable[Instance]) -> np.ndarray:
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        preds = []
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = move_to_device(batch, self.cuda_device)
                preds.append(self._extract_data(batch))
        return np.concatenate(preds, axis=0)


class DocumentClassificationDatasetReader(DatasetReader):
    """
    DatasetReader for Document Classification data
    The CSV columns must contains 'id', '`text_col`', '`label_cols`'
    """

    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_seq_len: Optional[int] = MAX_SEQ_LEN) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_seq_len = max_seq_len

    def text_to_instance(self, tokens: List[Token], id: str = None,
                         labels: np.ndarray = None) -> Instance:
        tokens_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": tokens_field}

        id_field = MetadataField(id)
        fields["id"] = id_field

        if labels is None:
            labels = np.zeros(len(label_cols))
        label_field = ArrayField(array=labels)
        fields["label"] = label_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        df = pd.read_csv(file_path)
        if TESTING:
            df = df.head(100)
        for i, row in df.iterrows():
            yield self.text_to_instance(
                [Token(x) for x in self.tokenizer(row[text_col])],
                row["id"], row[label_cols].values,
            )


class LstmClassifier(Model):

    def __init__(
        self,
        word_embeddings: TextFieldEmbedder,
        encoder: Seq2VecEncoder,
        vocab: Vocabulary
    ) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=len(label_cols)
                                          )
        self.accuracy = BooleanAccuracy()
        self.f1_measure = F1Measure(1)  # FBetaMeasure

    def forward(
        self,
        tokens: Dict[str, torch.Tensor],
        id: Any,
        label: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:

        mask = get_text_field_mask(tokens)  # mask to exclude the padding

        embeddings = self.word_embeddings(tokens)

        encoder_out = self.encoder(embeddings, mask)

        class_logits = self.hidden2tag(encoder_out)
        output = {"class_logits": class_logits}

        if label is not None:
            max_pred = class_logits.max(dim=1)[1]
            max_gold = label.max(dim=1)[1]
            self.accuracy(max_pred, max_gold)
            output["loss"] = torch.nn.BCEWithLogitsLoss()(class_logits, label)

        return output

    def get_metrics(
        self,
        reset: bool = False
    ) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset),
                #             "precision": self.f1_measure.get_metric(reset)[0],
                #             "recall": self.f1_measure.get_metric(reset)[1],
                #             "f1_measure": self.f1_measure.get_metric(reset)[2]
                }


if __name__ == "__main__":

    # Reading the Data ######
    # token_indexer = ELMoTokenCharactersIndexer()
    reader = DocumentClassificationDatasetReader()
    train_dataset = reader.read(Path('data/train.csv'))
    validation_dataset = reader.read(Path('data/test.csv'))

    vocab = Vocabulary.from_instances(train_dataset)

    # Instantiating the Model ######

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=EMBEDDING_DIM,
                                pretrained_file="wikipedia_mecab_word2vec/word2vec_glovefmt.txt")
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    lstm: Seq2VecEncoder = PytorchSeq2VecWrapper(
        torch.nn.LSTM(word_embeddings.get_output_dim(), HIDDEN_DIM, bidirectional=True, batch_first=True))

    model = LstmClassifier(word_embeddings, lstm, vocab)

    # Train ######
    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # sort the instances by the number of tokens in the sentence field
    iterator = BucketIterator(batch_size=4,
                              sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      patience=10,
                      num_epochs=N_EPOCHS,
                      cuda_device=cuda_device)
    trainer.train()

    # Here's how to save the model.
    with open("classifier-bilstm_result/model.th", 'wb') as f:
        torch.save(model.state_dict(), f)
    vocab.save_to_files("classifier-bilstm_result/vocabulary")
    # And here's how to reload the model.
    # vocab2 = Vocabulary.from_files("classifier-bilstm_result/vocabulary")
    # model2 = LstmClassifier(word_embeddings, lstm, vocab2)
    # with open("classifier-bilstm_result/model.th", 'rb') as f:
    #     model2.load_state_dict(torch.load(f))
    # if cuda_device > -1:
    #     model2.cuda(cuda_device)

    # Predict ######
    seq_iterator = BasicIterator(batch_size=4)  # iter without changing order
    seq_iterator.index_with(vocab)
    predictor = DocumentClassifierPredictor(model, seq_iterator, cuda_device)
    probs = predictor.predict(validation_dataset)
    print(probs)
    tag_ids = np.argmax(probs, axis=-1)
    print([label_cols[i] for i in tag_ids])
