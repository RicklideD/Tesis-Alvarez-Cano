import json
import os

import attr
import torch

from nl2sql_models.gap.seq2struct.models import abstract_preproc
from nl2sql_models.gap.seq2struct.models import variational_lstm
from nl2sql_models.gap.seq2struct.utils import registry
from nl2sql_models.gap.seq2struct.utils import vocab

@attr.s
class NL2CodeEncoderState:
    state = attr.ib()
    memory = attr.ib()
    words = attr.ib()

    def find_word_occurrences(self, word):
        return [i for i, w in enumerate(self.words) if w == word]


@registry.register('encoder', 'NL2Code')
class NL2CodeEncoder(torch.nn.Module):

    batched = False

    class Preproc(abstract_preproc.AbstractPreproc):
        def __init__(
                self,
                save_path,
                min_freq=3,
                max_count=5000):
            self.vocab_path = os.path.join(save_path, 'enc_vocab.json')
            self.data_dir = os.path.join(save_path, 'enc')

            self.vocab_builder = vocab.VocabBuilder(min_freq, max_count)
            self.init_items()
            self.vocab = None
        
        def init_items(self):
            # # TODO: Escribir 'train', 'val', 'test' en algún otro lugar
            self.texts = {'train': [], 'val': [], 'test': []}


        def validate_item(self, item, section):
            return True, None
        
        def add_item(self, item, section, validation_info):
            if section == 'train':
                for token in item.text:
                    self.vocab_builder.add_word(token)
            self.texts[section].append(item.text)

        def clear_items(self):
            self.init_items()

        def preprocess_item(self, item, validation_info):
            return item.text

        def save(self):
            os.makedirs(self.data_dir, exist_ok=True)
            self.vocab = self.vocab_builder.finish()
            self.vocab.save(self.vocab_path)

            for section, texts in self.texts.items():
                with open(os.path.join(self.data_dir, section + '.jsonl'), 'w') as f:
                    for text in texts:
                        f.write(json.dumps(text) + '\n')

        def load(self):
            self.vocab = vocab.Vocab.load(self.vocab_path)

        def dataset(self, section):
            return [
                json.loads(line)
                for line in open(os.path.join(self.data_dir, section + '.jsonl'))]

    def __init__(
            self,
            device,
            preproc,
            word_emb_size=128,
            recurrent_size=256,
            dropout=0.):
        super().__init__()
        self._device = device
        self.desc_vocab = preproc.vocab

        self.word_emb_size = word_emb_size
        self.recurrent_size = recurrent_size
        assert self.recurrent_size % 2 == 0

        self.desc_embedding = torch.nn.Embedding(
                num_embeddings=len(self.desc_vocab),
                embedding_dim=self.word_emb_size)
        self.encoder = variational_lstm.LSTM(
                input_size=self.word_emb_size,
                hidden_size=self.recurrent_size // 2,
                bidirectional=True,
                dropout=dropout)

    def forward(self, desc_words):
        # # forma de desc_indices: lote (=1) x longitud de desc
        desc_indices = torch.tensor(
                self.desc_vocab.indices(desc_words),
                device=self._device).unsqueeze(0)
        # forma de desc_emb : batch (=1) x desc length x word_emb_size
        desc_emb = self.desc_embedding(desc_indices)
        # desc_emb shape: desc length x batch (=1) x word_emb_size
        desc_emb = desc_emb.transpose(0, 1)

        # forma de outputs: desc length x batch (=1) x recurrent_size
        # forma de state:
        # - h: num_layers (=1) * num_directions (=2) x batch (=1) x recurrent_size / 2
        # - c: num_layers (=1) * num_directions (=2) x batch (=1) x recurrent_size / 2
        outputs, state = self.encoder(desc_emb)

        return NL2CodeEncoderState(
            state=state,
            memory=outputs.transpose(0, 1),
            words=desc_words)
