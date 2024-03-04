# -*- coding: utf-8 -*-

import re

import torch
from typing import Iterable
import pandas as pd

from utilis.constants import LABEL_PAD, MAX_SENT_LENGTH, END_TEXT_LEN


def preprocess_location(text):
    pattern_1 = r'位置信息(?::|：)(.{3,20})'
    pattern_2 = r'投诉地址(?::|：)(.{3,20})'
    pattern_3 = r'营业厅地址(?::|：)(.{3,20})'
    pattern_4 = r'地址(?::|：)(.{3,20})'
    for i in range(1, 5):
        # print(i)
        pattern = eval(f'pattern_{i}')
        match_ = re.findall(pattern, text)
        if match_:
            return match_[0]
    return text


def preprocess_rule(text):

    pattern_0 = r'(用户要求|投诉内容)(.*)\d\.'
    pattern_1 = r'(投诉内容|投诉原文)(.*)'
    pattern_2 = r'案件编号.*?(抄送主题)(.*)'
    pattern_3 = r'受理流水号.*?(用户现象)(.*)'
    pattern_4 = r'(争议内容)(.*)'
    pattern_5 = r'(问题描述)(.*)'
    pattern_6 = r'(用户诉求)(.*)'
    pattern_7 = r'(反映)(.*)'
    pattern_8 = r'(高级模板内容)(.*)'
    pattern_9 = r'投诉(原文|来源)(.*)'
    pattern_10 = r'(用户现象)(.*)'

    for i in range(0, 11):
        pattern_name = f'pattern_{i}'
        matched_result = re.findall(eval(pattern_name), text)
        if matched_result:
            # print(i)
            return ''.join(matched_result[0])
    return text


def _normalize(text):
    # replace by a specific pattern
    phone_pattern = re.compile(r'(?<![a-zA-Z0-9])\d{11}(?! *\d)')
    order_pattern = re.compile(r'[a-zA-Z]+\d{12,}')
    time_pattern = re.compile(r'\d{4}[--./]\d{2}[--./]\d{2}')
    pattern_with_content = {'order_pattern': '工单ID',
                            'phone_pattern': '号码',
                            'time_pattern': '时间'}
    # replace by None
    station_pattern = re.compile(r'(?:_[a-zA-Z0-9]+){3}_[\u4e00-\u9fff]{2,5}')
    location_pattern = re.compile(r'\d{1,3}[.]\d{6}')
    time_details = re.compile(r'\d{1,2}:\d{2}(?::\d{2})?')
    number_pattern = re.compile(r'\d{5,}')
    worker_id = re.compile(r'[a-zA-Z]{2}\d{6}')
    pattern_remove = [station_pattern, location_pattern, time_details,
                      number_pattern, worker_id]
    for pattern_name, replace_content in pattern_with_content.items():
        text = re.sub(eval(pattern_name), replace_content, text)
    for pattern_name in pattern_remove:
        text = re.sub(pattern_name, '', text)
    text = re.sub('-*', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    # cut overlength text
    match = re.search('受理流水号', text.strip())
    if match:
        start_text = text[:match.start()]
        end_text = text[len(text) - END_TEXT_LEN]
        text_cut = start_text + end_text
    else:
        text_cut = text
    return text_cut


def _padding_token(labels):
    """
    padding labels into same length
    """
    max_len = max(map(len, labels))
    pad_label_token = []
    for inner_label in labels:
        pad_num = max_len - len(inner_label)
        pad_label_token.append(inner_label + [LABEL_PAD] * pad_num)
    return pad_label_token


PREPROCESS_RULE = {'location_info': [_normalize, preprocess_location],
                   'others': [_normalize, preprocess_rule]}


def _pre_processing(batch_data, task_name='others'):
    """
    processing batch data from dataloader before feeding into model depends on task type
    return:
    all_text: list[str]
    all_label: list[int] (see this task as a token classification tasks)
    entity_spans: pd.Dataframe (in this way, model can select list of indexes efficiently
    """
    all_text, all_label = [], []
    preprocess_chains = PREPROCESS_RULE[task_name]
    for zip_sample in batch_data:
        text, label = zip_sample[0], zip_sample[1]
        # print(f'Origin length: {len(text)}')
        # text = _normalize(text)
        # print(f'After normalize {len(text)}')
        # text = preprocess_rule(text)
        # print(f'After processing: {len(text)}')
        for preprocess_fun in preprocess_chains:
            text = preprocess_fun(text)

        all_text.append(text)
        all_label.append(label)

    return all_text, all_label


def _pre_processing_multiple(batch_data, task_name='others'):
    """
    processing batch data from dataloader before feeding into model depends on task type
    return:
    all_text: list[str]
    all_tag: list[str]
    all_label: list[int] (see this task as a token classification tasks)
    entity_spans: pd.Dataframe (in this way, model can select list of indexes efficiently
    """
    all_text, all_tag, all_label = [], [], []
    preprocess_chains = PREPROCESS_RULE[task_name]
    for zip_sample in batch_data:
        text, tag, label = zip_sample[0], zip_sample[1], zip_sample[2]
        # text = _normalize(text)
        # text = preprocess_rule(text)
        for preprocess_func in preprocess_chains:
            text = preprocess_func(text)
        all_text.append(text)
        all_tag.append(tag)
        all_label.append(label)

    return all_text, all_tag, all_label


def _pre_processing_inference(inputs, task_name='others'):
    """

    :param inputs: str or [str]
    :param task_name: [location_info, others]
    :return: str or [str]
    """
    preprocess_chain = PREPROCESS_RULE[task_name]
    if isinstance(inputs, str):
        for preprocess_fun in preprocess_chain:
            inputs = preprocess_fun(inputs)
        return inputs
    else:
        inputs_ = []
        for text_ in inputs:
            for preprocess_fun in preprocess_chain:
                text_ = preprocess_fun(text_)
            inputs_.append(text_)
    return inputs_


def _post_process(batchfy_input, all_label):
    """
    post process : padding label if necessary, and add label into input.
    the return format follows the huggingface format
    :param all_label:
    :param batchfy_input: output of clollate_fn tokenizer function
    :param batch_data_sep: output of dataset.__get_item__ function
    :return: dict
    """
    if isinstance(all_label[0], Iterable):
        all_label = _padding_token(all_label)
    batchfy_input['labels'] = torch.tensor(all_label, dtype=torch.long)
    return batchfy_input


class CollateFn:
    """
    given the sample of dataset, ClllateFn do tokenization, convert these features into tensor

    """

    def __init__(self,
                 tokenizer,
                 label2idx,
                 idx2label=None,
                 is_split=False,
                 task_name='others'
                 ):
        self.tokenizer = tokenizer
        self.label2idx = label2idx
        if idx2label:
            self.idx2label = idx2label
        else:
            self.idx2label = {}
            for label, idx in label2idx.items():
                self.idx2label[idx] = label
        self.is_split = is_split
        self.task_name = task_name

    def __call__(self, batch_data):
        all_text, all_label = _pre_processing(batch_data, self.task_name)
        batchfy_input = self.processing(all_text)
        batchfy_input = _post_process(batchfy_input, all_label)
        return batchfy_input

    def processing(self, all_text):
        # (all_text, all_label, entity_spans[optional])
        batchfy_input = self.tokenizer(all_text,
                                       is_split_into_words=self.is_split,
                                       truncation=True,
                                       padding=True,
                                       return_tensors='pt',
                                       max_length=MAX_SENT_LENGTH
                                       )
        return batchfy_input

    def inference(self, serv_content, serv_type, **kwargs):
        inputs = _pre_processing_inference(serv_content, self.task_name)
        batchfy_input = self.tokenizer(inputs,
                                       is_split_into_words=self.is_split,
                                       truncation=True,
                                       padding=True,
                                       return_tensors='pt',
                                       max_length=MAX_SENT_LENGTH
                                       )
        batchfy_input['labels'] = None
        return batchfy_input


class CollateFnLight(CollateFn):
    """
    collate fn for light models such as RNNs, CNNs.
    """

    def __init__(self,
                 tokenizer,
                 label2idx,
                 idx2label=None,
                 is_split=False,
                 task_name='others'
                 ):
        super(CollateFnLight, self).__init__(tokenizer, label2idx, idx2label, is_split, task_name)

    def processing(self, all_text):
        # all_text, all_label = _pre_processing(batch_data)
        batch_token_idx, batch_token_len = self.tokenizer(all_text, self.is_split)
        batchfy_input = {'input_ids': torch.tensor(batch_token_idx, dtype=torch.long),
                         'input_lengths': batch_token_len}
        return batchfy_input

    def inference(self, serv_content, serv_type, **kwargs):
        inputs = _pre_processing_inference(serv_content, self.task_name)
        batch_token_idx, batch_token_len = self.tokenizer(inputs)
        batchfy_input = {'input_ids': torch.tensor(batch_token_idx, dtype=torch.long),
                         'input_lengths': batch_token_len,
                         'labels': None}
        return batchfy_input


class CollateFnLightMultiple:
    """
    collate function to handle multiple inputs
    """

    def __init__(self, tokenizer, label2idx, idx2label=None, is_split=False, task_name='others'):
        self.tokenizer_text, self.tokenizer_tag = tokenizer
        self.is_split = is_split
        self.label2idx = label2idx
        self.task_name = task_name

    def processing(self, texts, tags):
        # all_text, all_label = _pre_processing(batch_data)
        batch_text_idx, batch_text_len = self.tokenizer_text(texts, self.is_split)
        batch_tag_idx, batch_tag_len = self.tokenizer_tag(tags, is_token=True)
        batchfy_input = {'text_ids': torch.tensor(batch_text_idx, dtype=torch.long),
                         'tag_ids': torch.tensor(batch_tag_idx, dtype=torch.long),
                         'tag_lengths': batch_tag_len,
                         'text_lengths': batch_text_len}
        return batchfy_input

    def __call__(self, batch_data):
        all_text, all_tag, all_label = _pre_processing_multiple(batch_data, self.task_name)
        batchfy_input = self.processing(all_text, all_tag)
        batchfy_input = _post_process(batchfy_input, all_label)
        return batchfy_input

    def inference(self, serv_content, serv_type, **kwargs):
        serv_content = _pre_processing_inference(serv_content, self.task_name)
        # print(serv_content)
        batch_text_idx, batch_text_len = self.tokenizer_text(serv_content)
        serv_type = serv_type.strip().split('>>')
        batch_tag_idx, batch_tag_len = self.tokenizer_tag([serv_type], is_token=True)
        # print(batch_tag_idx)
        batchfy_input = {'text_ids': torch.tensor(batch_text_idx, dtype=torch.long),
                         'tag_ids': torch.tensor(batch_tag_idx, dtype=torch.long),
                         'tag_lengths': batch_tag_len,
                         'text_lengths': batch_text_len}
        return batchfy_input


class CollateFnPretrainedMultiple(CollateFnLightMultiple):
    def __init__(self, tokenizer, label2idx, idx2label=None, is_split=False, task_name='others'):
        super(CollateFnPretrainedMultiple, self).__init__(tokenizer, label2idx, idx2label, is_split, task_name)

    def processing(self, texts, tags):
        batchfy_input = self.tokenizer_text(texts,
                                            is_split_into_words=self.is_split,
                                            truncation=True,
                                            padding=True,
                                            return_tensors='pt',
                                            max_length=MAX_SENT_LENGTH)
        batch_tag_idx, batch_tag_len = self.tokenizer_tag(tags, is_token=True)
        batchfy_input['tag_lengths'] = batch_tag_len
        batchfy_input['tag_ids'] = torch.tensor(batch_tag_idx, dtype=torch.long)
        return batchfy_input
