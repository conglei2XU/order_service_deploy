import pdb
import os
import json
import pickle
from collections import Counter

import torch
# import jieba_fast as jieba
import jieba
from numpy.random import default_rng
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from models.Seq2Seq import Seq2Seq, RnnEncoder, RnnEncoderMultiple, RnnEncoderServType, RnnDecoder
from models.ClassificationLight import RNNNet
from models.ClassificationPretrained import PretrainClassificationModel, PretrainClassificationMultiple
from utilis.reader import csv_reader, read_vector
from pipeline.dataset import GeneralDataset, HierarchyDataset, HierarchyDatasetMultiple

from utilis.constants import PAD, MAX_SENT_LENGTH, UNK, THRESHOLD

# jieba.enable_paddle()

READER = {'csv': csv_reader}

_META_DATA = {'general': GeneralDataset,
              'hierarchy': HierarchyDataset,
              'hierarchy_multiple': HierarchyDatasetMultiple}

_META_MODEL = {
    'classification_servtype': RnnEncoderServType,
    'classification_general': RNNNet,
    'classification_multiple': RnnEncoderMultiple,
    'pretrained_classification_multiple': PretrainClassificationMultiple,
    'pretrained_classification_general': PretrainClassificationModel,
    'hierarchy': RnnEncoder,
    'hierarchy_multiple': RnnEncoderMultiple,
    'pretrained_hierarchy': PretrainClassificationModel,
    'pretrained_hierarchy_multiple': PretrainClassificationMultiple}


class TokenizerLight:
    def __init__(self, dataset, tokenizer=None, token2idx=None, sep='.', word2vector=None):
        self.sep = sep
        self.counter = Counter()
        self.word2vector = word2vector
        if tokenizer:
            self.basic_tokenizer = tokenizer.tokenize
        else:
            self.basic_tokenizer = self._default_tokenize
        self.token2idx = {'PAD': PAD,
                          'UNK': UNK}
        self.idx2token = {}

        if token2idx:
            print(f'loading predefined token2idx form {token2idx}')
            self.token2idx = json.load(open(token2idx, 'r'))
            for key, value in self.token2idx.items():
                self.idx2token[value] = key
        else:
            if dataset:
                self._build_vocab(dataset)
        if word2vector is not None and token2idx is None:
            self._merge_word2vector()

    def __call__(self, batch_text, is_token=False):
        # print(batch_text)
        if is_token:
            batch_token = batch_text
        else:
            if isinstance(batch_text, str):
                batch_token = [self.basic_tokenizer(batch_text)]
            else:
                batch_token = [self.basic_tokenizer(text_sample) for text_sample in batch_text]
        # print(batch_token)
        batch_token_len = []
        for token_sample in batch_token:
            if len(token_sample) > MAX_SENT_LENGTH:
                batch_token_len.append(MAX_SENT_LENGTH)
            else:
                batch_token_len.append(len(token_sample))

        batch_token_idx = self._post_processing(batch_token)
        return batch_token_idx, batch_token_len

    def _post_processing(self, batch_token):
        max_batch_len = max(map(len, batch_token))
        if max_batch_len > MAX_SENT_LENGTH:
            max_batch_len = MAX_SENT_LENGTH
        batch_token_idx = []
        for token_sample in batch_token:
            token_idx_sample = [self.token2idx.get(token, self.token2idx['UNK']) for token in token_sample]
            if len(token_sample) > MAX_SENT_LENGTH:
                token_sample = token_sample[:MAX_SENT_LENGTH]
                token_idx_sample = token_idx_sample[:MAX_SENT_LENGTH]
            padding_num = max_batch_len - len(token_sample)
            if padding_num > 0:
                token_idx_sample.extend([self.token2idx['PAD']] * padding_num)
            batch_token_idx.append(token_idx_sample)
        return batch_token_idx

    def _build_vocab(self, dataset):
        """
    build word counter and then construct token2idx from word counter
    """

        for data_idx in range(len(dataset)):
            text_sample = dataset[data_idx][0]
            tokens = self.basic_tokenizer(text_sample)
            self.counter.update(tokens)
        for token, freq in self.counter.items():
            if freq > THRESHOLD:
                self.token2idx[token] = len(self.token2idx)
                self.idx2token[len(self.idx2token)] = token

    def _default_tokenize(self, batch_text):
        if isinstance(batch_text, str):
            if self.sep is None:
                tokens = [token for token in batch_text]
            else:
                tokens = list(jieba.lcut(batch_text, HMM=False))
            if len(tokens) <= 0:
                print(f'Invalid Lengths: {batch_text}, {tokens}')

        else:
            tokens = []
            for text_sample in batch_text:
                if self.sep is None:
                    token_ = [token for token in text_sample]
                else:
                    token_ = (jieba.cut(text_sample, use_paddle=True))
                tokens.append(token_)
        return tokens

    def _merge_word2vector(self):
        for word in self.word2vector.keys():
            if word not in self.token2idx:
                self.token2idx[word] = len(self.token2idx)
                self.idx2token[len(self.idx2token)] = word

    def save_pretrained(self, output_dir, **kwargs):
        pass


def build_matrix(token_alphabet, word_vector, word_dim=300) -> torch.Tensor:
    random_generator = default_rng()
    total_word = len(token_alphabet)
    out_vocabulary = 0
    vector_matrix = random_generator.standard_normal((total_word, word_dim))
    # vector_matrix = np.zeros((total_word, word_dim))
    for word in token_alphabet:
        if word in word_vector:
            idx = token_alphabet[word]
            vector_matrix[idx, :] = word_vector[word]
        else:
            out_vocabulary += 1
    if word_vector:
        print(f'out of vocabulary number: {out_vocabulary}/{len(token_alphabet)}')
    return torch.tensor(vector_matrix, dtype=torch.float)


class PreTraining:
    def __init__(self, train_arguments, model_arguments, control_arguments):
        # model config
        self.hidden_dim = model_arguments.hidden_dim
        self.batch_size = train_arguments.per_device_train_batch_size
        self.model_path = model_arguments.model_path
        self.save_to = train_arguments.output_dir
        self.label_dim = model_arguments.label_dim
        self.label_vector = model_arguments.label_vector
        self.token2idx = control_arguments.token_to_idx
        self.dataset_meta = control_arguments.dataset_meta
        self.tag_vector_dim = model_arguments.tag_vector_dim
        self.word_vector_tag = model_arguments.word_vector_tag
        self.encoder_dim = model_arguments.encoder_dim
        self.decoder_dim = model_arguments.decoder_dim
        self.text_hidden_dim = model_arguments.text_hidden_dim
        self.tag_hidden_dim = model_arguments.tag_hidden_dim

        # config for setting pipeline
        self.dataset_path = control_arguments.dataset_path
        self.reader = READER[control_arguments.reader]
        self.Dataset = _META_DATA[control_arguments.dataset_meta]
        self.model_meta = control_arguments.model_meta

        # training arguments
        self.train_arguments = train_arguments
        setattr(self.train_arguments, 'model_meta', control_arguments.model_meta)

        # post-processing after initialization
        if os.path.exists(control_arguments.label_mapping):
            print(f'loading predefined label2idx from {control_arguments.label_mapping}')
            self.label2idx = json.load(open(control_arguments.label_mapping, 'r', encoding='utf-8'))
        else:
            print(f'Initialized label2idx from train dataset')
            self.label2idx = {}

        if not control_arguments.do_inference:
            self.train, self.val, self.test = self.init_dataset()
            if self.label2idx:
                self.train.label2idx = self.label2idx
            else:
                self.label2idx = self.train.label2idx
            self.val.label2idx = self.label2idx
            self.test.label2idx = self.label2idx
            json.dump(self.label2idx, open('label_mapping.json', 'w', encoding='utf-8'))
        else:
            self.train, self.val, self.test = None, None, None

        self.idx2label = {}
        for key, value in self.label2idx.items():
            self.idx2label[value] = key

    def create_loader(self, collate_fn=None, data_sampler=None):
        train_loader = DataLoader(self.train, batch_size=self.batch_size, collate_fn=collate_fn,
                                  sampler=data_sampler, shuffle=True)
        val_loader = DataLoader(self.val, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=True)
        test_loader = DataLoader(self.test, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=True)
        return train_loader, val_loader, test_loader

    def init_dataset(self):
        if self.model_meta == 'seq2seq':
            is_seq2seq = True
        else:
            is_seq2seq = False
        train_dataset = self.Dataset(os.path.join(self.dataset_path, 'train.csv'), self.reader, is_seq2seq=is_seq2seq)
        val_dataset = self.Dataset(os.path.join(self.dataset_path, 'eval.csv'), self.reader, is_seq2seq=is_seq2seq)
        test_dataset = self.Dataset(os.path.join(self.dataset_path, 'test.csv'), self.reader, is_seq2seq=is_seq2seq)
        return train_dataset, val_dataset, test_dataset

    def prepare_model(self):
        if 'albert' in self.model_path or 'Albert' in self.model_path:
            tokenizer_text = BertTokenizer.from_pretrained(self.model_path)
        else:
            tokenizer_text = BertTokenizer.from_pretrained(self.model_path)
        try:
            # assert 'pretrained' in self.model_meta
            if os.path.exists(self.word_vector_tag):
                print(f'loading predefined tag word2vector embedding {self.word_vector_tag}')
                word2vector_tag = torch.load(self.word_vector_tag)
            else:
                print(f'{self.word_vector_tag} not existed in this project')
                word2vector_tag = None
            tokenizer_tag = TokenizerLight(self.train, token2idx=self.token2idx,
                                           word2vector=word2vector_tag)
            model_meta_name = 'pretrained_' + self.model_meta
            model_meta = _META_MODEL[model_meta_name]

            # pdb.set_trace()
            model = model_meta(
                model_path=self.model_path,
                tag_vocabulary_size=len(tokenizer_tag.token2idx),
                tag_vector_size=self.tag_vector_dim,
                hidden_dim=self.hidden_dim,
                num_labels=len(self.label2idx),
                pretrained_vector_tag=word2vector_tag,
                idx2label=self.idx2label,
                label2idx=self.label2idx
            )

        except KeyError:
            if self.model_meta == 'seq2seq':
                if os.path.exists(self.label_vector):
                    print(f'loading pretrained label2vector: {self.label_vector}')
                    pretrained_vector_label = torch.load(self.label_vector)
                else:
                    print(f'Not found label2vector file, initializing label2vector randomly')
                    pretrained_vector_label = None
                encoder_meta_name = 'pretrained_' + self.dataset_meta
                encoder_meta = _META_MODEL[encoder_meta_name]
                encoder = encoder_meta(
                    model_path=self.model_path,
                    tag_vocabulary_size=len(tokenizer_tag.token2idx),
                    tag_vector_size=self.tag_vector_dim,
                    hidden_dim=self.hidden_dim,
                    num_labels=len(self.label2idx),
                    pretrained_vector_tag=word2vector_tag,
                    is_seq2seq=True)
                if os.path.exists(self.label_vector):
                    print(f'loading pretrained label2vector: {self.label_vector}')
                    pretrained_vector_label = torch.load(self.label_vector)
                else:
                    print(f'Not found label2vector file, initializing label2vector randomly')
                    pretrained_vector_label = None
                decoder = RnnDecoder(
                    vocabulary_size=len(self.label2idx),
                    label_vector_size=self.label_dim,
                    hidden_size=self.decoder_dim,
                    encoder_size=self.hidden_dim,
                    pretrained_vector=pretrained_vector_label
                )
                model = Seq2Seq(encoder, decoder)
            else:
                print(f"Don't support model type : {self.model_meta}")
                exit(1)
        if self.dataset_meta == 'hierarchy_multiple':
            return (tokenizer_text, tokenizer_tag), model
        else:
            return tokenizer_text, model


class PreTrainingLight(PreTraining):
    def __init__(self, train_arguments, model_arguments, control_arguments):
        self.cached_tokenizer = control_arguments.cached_tokenizer
        self.char_hidden_dim = model_arguments.char_hidden_dim
        self.input_dim = model_arguments.input_dim
        self.num_layers = model_arguments.num_layers
        self.word_vector_text = model_arguments.word_vector_text
        self.word_vector_dim = model_arguments.word_vector_dim
        self.use_vector = model_arguments.use_vector
        super(PreTrainingLight, self).__init__(train_arguments, model_arguments, control_arguments)

    def prepare_model(self):
        basic_tokenizer, word2vector_text, word2vector_tag, pretrained_vector = None, None, None, None
        try:
            print(f'try to load cached tokenizer from {self.cached_tokenizer}')
            tokenizer = pickle.load(open(self.cached_tokenizer, 'rb'))
            # tokenizer = pickle.load(open(self.cached_tokenizer, 'rb'), word2vector=word2vector)

        except FileNotFoundError:
            print(f"could not find existed cached tokenizer")
            print(f"initializing new tokenizer")
            if os.path.exists(self.word_vector_text) and self.use_vector:
                print(f'loading text word2vector from {self.word_vector_text}')
                word2vector_text = read_vector(self.word_vector_text, vector_dim=self.word_vector_dim)
                base_dir = os.path.dirname(self.cached_tokenizer)
                if not os.path.exists(base_dir):
                    os.makedirs(base_dir)
            else:
                print(f"{self.word_vector_text} doesn't exist in this project, using random initialized vector instead")
                word2vector_text = None
            tokenizer = TokenizerLight(self.train, basic_tokenizer, word2vector=word2vector_text)
            if word2vector_text is not None:
                pretrained_vector = build_matrix(tokenizer.token2idx, tokenizer.word2vector)
            pickle.dump(tokenizer, open(self.cached_tokenizer, 'wb'))

        if os.path.exists(self.word_vector_tag):
            print(f'loading predefined tag word2vector embedding {self.word_vector_tag}')
            word2vector_tag = torch.load(self.word_vector_tag)
        # tokenizer_tag = None
        tokenizer_tag = TokenizerLight(self.train, basic_tokenizer, token2idx=self.token2idx,
                                       word2vector=word2vector_tag)
        try:
            model_meta = _META_MODEL[self.model_meta]
            if self.model_meta == 'classification_general':
                model = RNNNet(
                    vocabulary_size=len(tokenizer.token2idx),
                    char_alphabet_size=None,
                    char_hidden_dim=self.char_hidden_dim,
                    input_size=self.input_dim,
                    hidden_size=self.hidden_dim,
                    word_vector_size=self.input_dim,
                    num_layers=self.num_layers,
                    num_labels=len(self.label2idx),
                    pretrained_vector=pretrained_vector
                )
            else:
                model = model_meta(
                    text_vocabulary_size=len(tokenizer.token2idx),
                    tag_vocabulary_size=len(tokenizer_tag.token2idx),
                    word_vector_size=self.word_vector_dim,
                    tag_vector_size=self.tag_vector_dim,
                    hidden_size=self.hidden_dim,
                    pretrained_vector=pretrained_vector,
                    pretrained_vector_tag=word2vector_tag,
                    num_labels=len(self.label2idx),
                    tag_hidden_size=self.tag_hidden_dim,
                    text_hidden_size=self.text_hidden_dim
                )

        except KeyError:
            if self.model_meta == 'seq2seq':
                if os.path.exists(self.label_vector):
                    print(f'loading pretrained label2vector: {self.label_vector}')
                    pretrained_vector_label = torch.load(self.label_vector)
                else:
                    print(f'Not found label2vector file, initializing label2vector randomly')
                    pretrained_vector_label = None
                model_meta = _META_MODEL[self.dataset_meta]
                encoder = model_meta(
                    text_vocabulary_size=len(tokenizer.token2idx),
                    tag_vocabulary_size=len(tokenizer_tag.token2idx),
                    word_vector_size=self.word_vector_dim,
                    tag_vector_size=self.tag_vector_dim,
                    pretrained_vector=pretrained_vector,
                    pretrained_vector_tag=word2vector_tag,
                    num_labels=len(self.label2idx),
                    is_seq2seq=True,
                    tag_hidden_size=self.tag_hidden_dim,
                    text_hidden_size=self.text_hidden_dim
                )

                decoder = RnnDecoder(
                    vocabulary_size=len(self.label2idx),
                    label_vector_size=self.label_dim,
                    hidden_size=self.decoder_dim,
                    encoder_size=self.text_hidden_dim,
                    pretrained_vector=pretrained_vector_label
                )
                model = Seq2Seq(encoder=encoder, decoder=decoder)
            else:
                print(f"Doesn't support model type : {self.model_meta}")
                exit(1)
        if self.dataset_meta == 'hierarchy_multiple':
            return (tokenizer, tokenizer_tag), model
        else:
            return tokenizer, model
