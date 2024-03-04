import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers.utils import ModelOutput
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    BigBirdModel
)
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class PretrainClassificationModel(nn.Module):
    def __init__(self, model_path, hidden_dim, num_labels, is_seq2seq=False, idx2label=None, label2idx=None, **kwargs):
        super(PretrainClassificationModel, self).__init__()
        self.hidden_size = hidden_dim
        self.encoder = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels, id2label=idx2label, label2id=label2idx)
        # self.encoder = BigBirdModel.from_pretrained(model_path)
        self.linear_layer = nn.Linear(hidden_dim, num_labels)
        self.num_labels = num_labels
        self.is_seq2seq = is_seq2seq

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                labels=None,
                spans=None,
                output_attentions=None,
                return_dict=None
                ):
        # (loss[optional], logit, hidden_states[optional], output_attentions[optional]
        output = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              return_dict=return_dict)
        # sequence_output = output[0]  # (batch_size, seq_len, hidden_dim)
        # last_hidden_state = sequence_output[:, 0, :]  # (batch_size, hidden_dim)
        # # batch_size, num_span = sequence_output.size(0), len(spans[0])
        # if self.is_seq2seq:
        #     return sequence_output, last_hidden_state.unsqueeze(0)
        # pred_ = self.linear_layer(last_hidden_state)  # (batch_size, num_categories)
        # return ModelOutput(logits=pred_, loss=None)
        return output

    def dynamic_quantization(self):
        quantized_model = torch.quantization.quantize_dynamic(self.encoder, {torch.nn.Linear}, dtype=torch.qint8)
        setattr(self, 'encoder', quantized_model)


class PretrainClassificationMultiple(nn.Module):

    def __init__(self, model_path, tag_vocabulary_size, tag_vector_size, hidden_dim, num_labels,
                 pretrained_vector_tag=None, is_seq2seq=False, **kwargs):
        super(PretrainClassificationMultiple, self).__init__()
        self.hidden_size = hidden_dim
        self.text_model = AutoModel.from_pretrained(model_path)
        if pretrained_vector_tag is not None:
            vocabulary_size_, vector_dim = pretrained_vector_tag.size()
            assert vector_dim == tag_vector_size and tag_vocabulary_size == vocabulary_size_
            self.embed_tag = nn.Embedding.from_pretrained(pretrained_vector_tag, freeze=False)
        else:
            self.embed_tag = nn.Embedding(tag_vocabulary_size, tag_vector_size)
        self.inner_linear = nn.Linear(tag_vector_size + 768, tag_vector_size)
        self.tag_model = nn.LSTM(tag_vector_size, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.full_connect = nn.Linear(hidden_dim + 768, num_labels)
        # self.encoder = BigBirdModel.from_pretrained(model_path)
        self.num_labels = num_labels
        self.is_seq2seq = is_seq2seq

    def forward(self,
                input_ids=None,
                tag_ids=None,
                tag_lengths=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                labels=None,
                spans=None,
                output_attentions=None,
                return_dict=None
                ):
        # (loss[optional], logit, hidden_states[optional], output_attentions[optional]
        output = self.text_model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 position_ids=position_ids,
                                 return_dict=return_dict)
        sequence_output = output[0]  # (batch_size, seq_len, hidden_dim)
        # seq_len = sequence_output.size(1)
        tag_len = max(tag_lengths)
        last_hidden_state = sequence_output[:, 0, :]  # (batch_size, hidden_dim)
        last_hidden_state_duplicate = last_hidden_state.unsqueeze(1).repeat(1, tag_len, 1)
        tag_embed = self.embed_tag(tag_ids)  # (batch_size, seq_len) -> (batch_size, seq_len, hidden_dim)
        # fuse_input = torch.cat((last_hidden_state_duplicate, tag_embed), dim=-1)
        # fuse_input = F.relu(self.inner_linear(fuse_input))
        tag_input = pack_padded_sequence(tag_embed, tag_lengths, batch_first=True)
        tag_output, (cell, hidden) = self.tag_model(tag_input)
        tag_output_, _ = pad_packed_sequence(tag_output, batch_first=True)
        # batch_size, num_span = sequence_output.size(0), len(spans[0])
        if self.is_seq2seq:
            hidden = torch.cat([hidden, hidden], dim=-1)
            return tag_output_, hidden

        pred_ = self.full_connect(torch.cat([tag_output_[:, 0, :], last_hidden_state], dim=-1))  # (batch_size, num_categories)
        return ModelOutput(logits=pred_, loss=None)

    def dynamic_quantization(self):
        quantized_model = torch.quantization.quantize_dynamic(self.encoder, {torch.nn.Linear}, dtype=torch.qint8)
        setattr(self, 'encoder', quantized_model)


class LongFormer(nn.Module):
    def __init__(self, model_path, hidden_dim, num_labels):
        super(LongFormer, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = BigBirdModel.from_pretrained(model_path)
        self.linear_layer = nn.Linear(hidden_dim, num_labels)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                label=None,
                spans=None,
                output_attentions=None,
                return_dict=None
                ):
        # (loss[optional], logit, hidden_states[optional], output_attentions[optional]
        output = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              return_dict=return_dict)
        sequence_output = output[0]
        batch_size, num_span = sequence_output.size(0), len(spans[0])
        entity_embedding = torch.rand(batch_size, num_span, self.hidden_dim)
        for idx, span_items in enumerate(spans):
            for idx_span, span_item in enumerate(span_items):
                entity_rep = sequence_output[idx, span_item[0]:span_item[1]]
                entity_embedding[idx, idx_span, :] = torch.mean(entity_rep, dim=0)
        # (batch_size, num_spans, hidden_dim) -> (batch_size, num_spans, num_categories)
        pred_ = self.linear_layer(entity_embedding)
        pred_ = pred_.view(batch_size * num_span, -1)  # (batch_size, number of categories)
        return pred_
