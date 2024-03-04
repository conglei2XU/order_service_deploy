import math
import operator
import random
from queue import PriorityQueue

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel
from transformers.utils import ModelOutput

from utilis.tools import timeit
from utilis.constants import EOS_token, SOS_token, PRE_TOKENS_LABEL, LABEL_PAD

CELL = {'GRU': nn.GRU,
        'LSTM': nn.LSTM, }


class RnnEncoderMultiple(nn.Module):
    def __init__(
            self,
            text_vocabulary_size,
            tag_vocabulary_size,
            word_vector_size,
            tag_vector_size=768,
            tag_hidden_size=768,
            text_hidden_size=768,
            hidden_size=768,
            n_layers=1,
            dropout=0.5,
            num_labels=None,
            pretrained_vector=None,
            pretrained_vector_tag=None,
            is_seq2seq=False
    ):
        super(RnnEncoderMultiple, self).__init__()
        self.hidden_size = text_hidden_size
        self.hidden_size_text = text_hidden_size
        self.hidden_size_tag = tag_hidden_size
        if pretrained_vector is not None:
            vocabulary_size_, vector_dim = pretrained_vector.size()
            assert vector_dim == word_vector_size and text_vocabulary_size == vocabulary_size_
            self.embed_text = nn.Embedding.from_pretrained(pretrained_vector, freeze=False)
        else:
            self.embed_text = nn.Embedding(text_vocabulary_size, word_vector_size)
        if pretrained_vector_tag is not None:
            vocabulary_size_, vector_dim = pretrained_vector_tag.size()
            assert vector_dim == tag_vector_size and tag_vocabulary_size == vocabulary_size_
            self.embed_tag = nn.Embedding.from_pretrained(pretrained_vector_tag, freeze=False)
        else:
            self.embed_tag = nn.Embedding(tag_vocabulary_size, tag_vector_size)
        self.inner_linear = nn.Linear(tag_vector_size + self.hidden_size_text, tag_vector_size)
        self.text_model = nn.LSTM(word_vector_size + tag_hidden_size, text_hidden_size // 2, n_layers,
                                  bidirectional=True, batch_first=True)
        self.tag_model = nn.LSTM(tag_vector_size, tag_hidden_size // 2, n_layers, batch_first=True, bidirectional=True)
        self.full_connect = nn.Linear(text_hidden_size, num_labels)
        self.attention = Attention(text_hidden_size, tag_hidden_size)
        self.num_labels = num_labels
        self.is_seq2seq = is_seq2seq

    def forward(self, text_ids, tag_ids, text_lengths, tag_lengths, hidden=None, labels=None):
        """

        :param text_ids: (batch_size, max_length)
        :param tag_ids: (batch_size, max_length)
        :param text_lengths: (batch_size)
        :param tag_lengths: (batch_size)
        :param hidden: (batch_size, hidden)
        :return: tuple(outputs , hidden) (batch_size, length, hidden_dim) (batch_size, N, hidden_dim)
        """
        # model semantic in the tag sequence
        text_embed = self.embed_text(text_ids)
        # combine embed_text and embed_tag by attention weights 2
        # text_embed = pack_padded_sequence(text_embed, lengths=text_lengths, enforce_sorted=False, batch_first=True)
        # text_outputs, (h_n, c_n) = self.text_model(text_embed)
        # text_outputs, _ = pad_packed_sequence(text_outputs, batch_first=True)
        # text_representation = torch.cat([h_n[0], h_n[1]], dim=-1)
        # tag_embed = self.embed_tag(tag_ids)
        # tag_input = pack_padded_sequence(tag_embed, lengths=tag_lengths, enforce_sorted=False, batch_first=True)
        # output, (hidden, cell) = self.tag_model(tag_input)
        # tag_outputs, _ = pad_packed_sequence(output, batch_first=True)
        #
        # atten_weights = self.attention(text_representation, tag_outputs)
        # context = torch.bmm(atten_weights, tag_outputs)  # (batch_size, 1, seq_len) * (batch_size, seq_len,

        tag_embed = self.embed_tag(tag_ids)
        tag_input = pack_padded_sequence(tag_embed, lengths=tag_lengths, enforce_sorted=False, batch_first=True)
        output, (hidden, cell) = self.tag_model(tag_input)
        tag_output, _ = pad_packed_sequence(output, batch_first=True)
        # tag_representation = hidden[0] # use the final state of backward
        tag_representation = torch.cat([hidden[0], hidden[1]],
                                       dim=-1)  # (batch_len, hidden_size) -> (batch_len, 2 * hidden_size)
        text_seq_len = text_ids.size(1)
        # model semantic in the text

        # [batch_size, word_vector_size]

        # combine embed_text and embed_tag before text lstm
        tag_representation = tag_representation.unsqueeze(1).repeat(1, text_seq_len, 1)
        fuse_input = torch.cat([text_embed, tag_representation], dim=-1)
        text_embed = pack_padded_sequence(fuse_input, lengths=text_lengths, enforce_sorted=False, batch_first=True)
        outputs, (hidden, cell) = self.text_model(text_embed)  # ([27, 32, 256],None)=>([27, 32, 1024],[4,
        # # 32, 512])
        text_outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        text_representation = torch.cat([hidden[0], hidden[1]], dim=-1)
        logit = self.full_connect(text_representation)

        # combine embed_text and embed_tag by attention weights
        # text_embed = pack_padded_sequence(text_embed, lengths=text_lengths, enforce_sorted=False, batch_first=True)
        # text_outputs, (h_n, c_n) = self.text_model(text_embed)
        # text_outputs, _ = pad_packed_sequence(text_outputs, batch_first=True)
        # atten_weights = self.attention(tag_representation, text_outputs)
        # context = torch.bmm(atten_weights, text_outputs)  # (batch_size, 1, seq_len) * (batch_size, seq_len,
        # # hidden_size)

        # hidden_size)
        # text_representation = torch.cat([text_outputs[:, -1, :self.hidden_size_text//2], text_outputs[:, 0, self.hidden_size_text//2:]], dim=-1)  # [batch_size] -> (batch_size, tag_vector_size)
        # text_representation = text_outputs[:, -1, :]
        # text_representation = hidden[1] # use the final state of backward
        # text_representation = hidden[0] # use the final state of forward

        # logit = self.full_connect(context.squeeze(1))

        # combine embed_text and embd_tag after text lstm and tag lstm

        # text_embed = pack_padded_sequence(text_embed, lengths=text_lengths, enforce_sorted=False, batch_first=True)
        # outputs, (hidden, cell) = self.text_model(text_embed)  # ([27, 32, 256],None)=>([27, 32, 1024],[4,
        # text_outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        # text_representation = text_outputs[:, 0, :]  # [batch_size] -> (batch_size, tag_vector_size)
        # fuse_input = torch.cat((text_representation, tag_representation), dim=-1)
        # logit = self.full_connect(fuse_input)
        # seq_len = tag_ids.size(1)
        # text_representation = text_representation.unsqueeze(1).repeat(1, seq_len, 1)

        # fuse_input = self.inner_linear(fuse_input)
        # # fuse_input = F.dropout(fuse_input)
        # fuse_input = pack_padded_sequence(fuse_input, lengths=tag_lengths, enforce_sorted=False, batch_first=True)
        # output, (hidden, cell) = self.tag_model(fuse_input)

        # fuse_input = torch.cat([tag_output[:, 0, :], text_representation], dim=-1)
        # logit = self.full_connect(torch.cat([tag_representation, text_representation], dim=-1))

        if self.is_seq2seq:
            hidden_ = torch.cat([hidden[0], hidden[1]], dim=-1)
            return text_outputs, hidden_
        else:
            return ModelOutput(logits=logit, loss=None)
        # return outputs, hidden


class RnnEncoderServType(nn.Module):
    def __init__(
            self,
            text_vocabulary_size,
            tag_vocabulary_size,
            word_vector_size,
            tag_vector_size=768,
            hidden_size=768,
            n_layers=1,
            dropout=0.5,
            num_labels=None,
            pretrained_vector=None,
            pretrained_vector_tag=None,
            **kwargs
    ):
        super(RnnEncoderServType, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_size_text = 300
        if pretrained_vector is not None:
            vocabulary_size_, vector_dim = pretrained_vector.size()
            assert vector_dim == word_vector_size and text_vocabulary_size == vocabulary_size_
            self.embed_text = nn.Embedding.from_pretrained(pretrained_vector, freeze=False)
        else:
            self.embed_text = nn.Embedding(text_vocabulary_size, word_vector_size)
        if pretrained_vector_tag is not None:
            vocabulary_size_, vector_dim = pretrained_vector_tag.size()
            assert vector_dim == tag_vector_size and tag_vocabulary_size == vocabulary_size_
            self.embed_tag = nn.Embedding.from_pretrained(pretrained_vector_tag, freeze=False)
        else:
            self.embed_tag = nn.Embedding(tag_vocabulary_size, tag_vector_size)
        self.inner_linear = nn.Linear(tag_vector_size + self.hidden_size_text * 2, tag_vector_size)
        self.text_model = nn.LSTM(word_vector_size, self.hidden_size_text, n_layers, bidirectional=True,
                                  batch_first=True)
        self.tag_model = nn.LSTM(tag_vector_size, hidden_size, n_layers, batch_first=True, bidirectional=True)
        self.full_connect = nn.Linear(hidden_size * 2, num_labels)
        self.num_labels = num_labels

    def forward(self, text_ids, tag_ids, text_lengths, tag_lengths, hidden=None, labels=None):
        """

        :param text_ids: (batch_size, max_length)
        :param tag_ids: (batch_size, max_length)
        :param text_lengths: (batch_size)
        :param tag_lengths: (batch_size)
        :param hidden: (batch_size, hidden)
        :return: tuple(outputs , hidden) (batch_size, length, hidden_dim) (batch_size, N, hidden_dim)
        """
        tag_embed = self.embed_tag(tag_ids)
        tag_input = pack_padded_sequence(tag_embed, lengths=tag_lengths, enforce_sorted=False, batch_first=True)
        output, (hidden, cell) = self.tag_model(tag_input)
        tag_output, _ = pad_packed_sequence(output, batch_first=True)
        logit = self.full_connect(tag_output[:, 0, :])
        # (batch_size, length, hidden_size)
        # sum bidirectional outputs
        # outputs = (tag_output[:, :, :self.hidden_size] +
        #            tag_output[:, :, self.hidden_size:])  # =>[27, 32, 512] + [27, 32, 512]
        return ModelOutput(logits=logit, loss=None)
        # return outputs, hidden


class RnnEncoder(nn.Module):
    def __init__(
            self,
            text_vocabulary_size,
            tag_vocabulary_size,
            word_vector_size,
            tag_vector_size=768,
            tag_hidden_size=768,
            text_hidden_size=768,
            hidden_size=768,
            n_layers=1,
            dropout=0.5,
            num_labels=None,
            pretrained_vector=None,
            pretrained_vector_tag=None,
            is_seq2seq=False
    ):
        super(RnnEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_size_text = hidden_size
        if pretrained_vector is not None:
            vocabulary_size_, vector_dim = pretrained_vector.size()
            assert vector_dim == word_vector_size and text_vocabulary_size == vocabulary_size_
            self.embed_text = nn.Embedding.from_pretrained(pretrained_vector, freeze=False)
        else:
            self.embed_text = nn.Embedding(text_vocabulary_size, word_vector_size)

        self.text_model = nn.LSTM(word_vector_size, self.hidden_size_text // 2, n_layers, bidirectional=True,
                                  batch_first=True)
        self.full_connect = nn.Linear(self.hidden_size_text, num_labels)
        self.num_labels = num_labels
        self.is_seq2seq = is_seq2seq

    def forward(self, input_ids, text_lengths, hidden=None, labels=None):
        """

        :param text_ids: (batch_size, max_length)
        :param tag_ids: (batch_size, max_length)
        :param text_lengths: (batch_size)
        :param tag_lengths: (batch_size)
        :param hidden: (batch_size, hidden)
        :return: tuple(outputs , hidden) (batch_size, length, hidden_dim) (batch_size, N, hidden_dim)
        """
        embed_input = self.embed_text(input_ids)  # [batch_size, ]
        text_embed = pack_padded_sequence(embed_input, lengths=text_lengths, enforce_sorted=False, batch_first=True)
        outputs, (hidden, cell) = self.text_model(text_embed, hidden)  # ([27, 32, 256],None)=>([27, 32, 1024],[4,
        # 32, 512])
        text_outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        if self.is_seq2seq:
            hidden_ = torch.cat([hidden[0], hidden[1]],
                                dim=-1)
            return text_outputs, hidden_
        else:
            return


class Attention(nn.Module):
    def __init__(self, hidden_size, content_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size + content_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        """

        :param hidden: (D*num_layers, batch_size, hidden_size)
        :param encoder_outputs: (batch_size, seq_len, hidden_size)
        :return: (batch_size, 1, seq_len)
        """
        timestep = encoder_outputs.size(1)
        h = hidden.unsqueeze(1).repeat(1, timestep, 1)  # [32, 512]=>[32, 27, 512]
        attn_energies = self.score(h, encoder_outputs)  # =>[B*T]
        return F.softmax(attn_energies, dim=-1)  # [B*T]=>[B*1*T]

    def score(self, hidden, encoder_outputs):
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))  # [batch_size, seq_len, hidden_size]
        energy = energy.transpose(1, 2)  # (batch_size, hidden_size, seq_len)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*H] bmm [B*H*T]=>[B*1*T]
        return energy


class RnnDecoder(nn.Module):
    def __init__(
            self,
            vocabulary_size,
            label_vector_size,
            hidden_size,
            encoder_size,
            n_layers=1,
            dropout=0.2,
            pretrained_vector=None):
        """

        :param vocabulary_size: number of classes in label set
        :param label_vector_size: embedding size for each label
        :param hidden_size: size of deocer's RNNs
        :param encoder_size: size of encoder's output
        :param n_layers: the number of layers for decoders' RNNs
        :param dropout: dropout rate
        """
        super(RnnDecoder, self).__init__()
        self.hidden_size = hidden_size
        if pretrained_vector is not None:
            self.embed = nn.Embedding.from_pretrained(pretrained_vector, freeze=False)
        else:
            self.embed = nn.Embedding(vocabulary_size, label_vector_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size, encoder_size)
        self.text_model = nn.GRU(encoder_size + label_vector_size, hidden_size,
                                 n_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, vocabulary_size)
        self.vocabulary_size = vocabulary_size
        self.n_layers = n_layers
        self.encoder_size = encoder_size

    def forward(self, input, last_hidden, encoder_outputs):  # 上一步的 output,上一步的 hidden_state
        """

        :param input: tensors (batch_len, 1, 1)
        :param last_hidden: (batch_len, lstm_out)
        :param encoder_outputs: (batch_len, lstm_out)
        :return:
        """
        # Get the embedding of the current input word (last output word)
        max_value = torch.max(input)
        embedded = self.embed(input).unsqueeze(1)  # (B,N) # [32]=>[32, 256]=>[1, 32, 256]
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        # (batch_size, 1, seq_len) * (batch_size, seq_len, hidden_dim) -> (batch_size, 1, hidden_dim)
        context = attn_weights.bmm(encoder_outputs)  # (batch_size, 1, seq_len) * (batch_size, seq_len, hidden_size)
        # context = context.transpose(0, 1)  # (1,B,N) # [32, 1, 512]=>[1, 32, 512]
        # Combine embedded input word and attended context, run through RNN
        # seq_len = embedded.size(1)
        # context = context.unsqueeze(1).repeat(1, seq_len, 1)
        # context = encoder_outputs
        rnn_input = torch.cat([embedded, context], 2)  # [1, 32, 256] cat [1, 32, 512]=> [1, 32, 768]
        output, hidden = self.text_model(rnn_input,
                                         last_hidden)  # in:[1, 32, 768],[1, 32, 512]=>[1, 32, 512],[1, 32, 512]
        output = output.squeeze(1)  # (batch_size, 1, hidden_size) -> (batch_size, hidden_size)
        output = self.out(output)  # [32, 512] cat [32, 512] => [32, 512*2]
        # output = F.log_softmax(output, dim=1)
        return output, hidden  # [batch_size, vocabulary_size] [1, batch_size, hidden_dim] [batch_size,
        # 1, lenght]


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, mode='light'):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.alignment_linear = nn.Linear(self.encoder.hidden_size, self.decoder.hidden_size)
        self.mode = mode

    def forward(self,
                input_ids=None,
                input_lengths=None,
                text_ids=None,
                tag_ids=None,
                labels=None,
                text_lengths=None,
                tag_lengths=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                return_dict=None,
                teacher_forcing_ratio=0.3
                ):
        if text_ids is not None:
            batch_size = text_ids.size(0)
            device_id = text_ids.get_device()
        else:
            batch_size = input_ids.size(0)
            device_id = input_ids.get_device()
        max_len = labels.size(1)

        if device_id != -1:
            device = torch.device(f'cuda:{device_id}')
        else:
            device = torch.device(f'cpu')
        outputs = []
        if input_ids is not None:
            if self.mode == 'pretrained':
                encoder_output, hidden = self.encoder(input_ids, attention_mask, token_type_ids, position_ids,
                                                      return_dict)
            else:
                encoder_output, hidden = self.encoder(input_ids, input_lengths)
        else:
            encoder_output, hidden = self.encoder(text_ids, tag_ids, text_lengths,
                                                  tag_lengths)  # [27, 32]=> =>[27, 32, 512],[4, 32, 512]
        hidden = self.alignment_linear(hidden)
        hidden = hidden.unsqueeze(0)
        # hidden = hidden[:self.decoder.n_layers]  # [4, 32, 512][1, 32, 512]
        output = labels.data[:, 0]  # sos
        for t in range(1, max_len):
            hidden = hidden.contiguous()
            output, hidden = self.decoder(
                output, hidden, encoder_output)  # output:[32, 10004] [1, 32, 512] [32, 1, 27]
            outputs.append(output.clone())
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = torch.argmax(output, dim=1)  # 按照 dim=1 求解最大值和最大值索引,x[1] 得到的是最大值的索引=>top1.shape=32
            if is_teacher:
                output = labels[:, t]
                output = torch.where(output != LABEL_PAD, output, EOS_token)
                # output[output == LABEL_PAD].fill_(SOS_token)
                # print(output)
                # fill label pad token -100 with <SOS> token

            else:
                output = top1
            # output = top1
        outputs = torch.stack(outputs, dim=1)
        return ModelOutput(logits=outputs, loss=None)

    def decode(self,
               input_ids=None,
               input_lengths=None,
               text_ids=None,
               tag_ids=None,
               labels=None,
               text_lengths=None,
               tag_lengths=None,
               attention_mask=None,
               token_type_ids=None,
               position_ids=None,
               return_dict=None,
               method='beam-search'
               ):
        if input_ids is not None:
            if self.mode == 'pretrained':
                encoder_output, hidden = self.encoder(input_ids, attention_mask, token_type_ids, position_ids,
                                                      return_dict)
            else:
                encoder_output, hidden = self.encoder(input_ids, input_lengths)
        else:
            encoder_output, hidden = self.encoder(text_ids, tag_ids, text_lengths,
                                                  tag_lengths)  # [27, 32]=> =>[27, 32, 512],[4, 32, 512]
        hidden = self.alignment_linear(hidden)
        hidden = hidden.unsqueeze(0)
        if method == 'beam-search':
            return self.beam_decode(labels, hidden, encoder_output)
        else:
            return self.greedy_decode(labels, hidden, encoder_output)

    def greedy_decode(self, target, decoder_hidden, encoder_outputs):
        '''
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch: ModelOutput(logits='', loss=None) logits: (batch_size, seq_len)
        '''
        batch_size, seq_len = target.size()
        decoded_batch = []
        decoder_input = target[:, 0]  # decoder_input: (batch_size)
        decoded_batch.append(decoder_input.clone())
        for t in range(1, seq_len):
            decoder_hidden = decoder_hidden.contiguous()
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # topv, topi = decoder_output.data.topk(1)  # [32, 10004] get candidates
            # topv: the max tensor; topi: the index of maximum tensor
            top1 = torch.argmax(decoder_output, dim=1)
            # topi = topi.view(-1)
            decoded_batch.append(top1.clone())
            decoder_input = top1.detach()
        decoded_batch = torch.stack(decoded_batch, dim=1)
        return ModelOutput(logits=decoded_batch, loss=None)

    @timeit
    def beam_decode(self, target_tensor, decoder_hiddens, encoder_outputs=None):
        '''
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hiddens: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        '''
        target_tensor = target_tensor.permute(1, 0)
        beam_width = 10
        topk = 1  # how many sentence do you want to generate
        decoded_batch = []

        # decoding goes sentence by sentence
        for idx in range(target_tensor.size(0)):  # batch_size
            if isinstance(decoder_hiddens, tuple):  # LSTM case
                decoder_hidden = (
                    decoder_hiddens[0][:, idx, :].unsqueeze(0), decoder_hiddens[1][:, idx, :].unsqueeze(0))
            else:
                decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)  # [1, B, H]=>[1,H]=>[1,1,H]
            encoder_output = encoder_outputs[:, idx, :].unsqueeze(1)  # [T,B,H]=>[T,H]=>[T,1,H]

            # Start with the start of the sentence token
            decoder_input = torch.LongTensor([SOS_token]).cuda()

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1

            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 2000: break

                # fetch the best node
                score, n = nodes.get()
                # print('--best node seqs len {} '.format(n.leng))
                decoder_input = n.wordid
                decoder_hidden = n.h

                if n.wordid.item() == EOS_token and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_output)

                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, indexes = torch.topk(decoder_output, beam_width)
                nextnodes = []

                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].view(-1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid)
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid)

                utterance = utterance[::-1]
                utterances.append(utterance)

            decoded_batch.append(utterances)

        return decoded_batch


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward  # 注意这里是有惩罚参数的，参考恩达的 beam-search

    def __lt__(self, other):
        return self.leng < other.leng  # 这里展示分数相同的时候怎么处理冲突，具体使用什么指标，根据具体情况讨论

    def __gt__(self, other):
        return self.leng > other.leng


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()
