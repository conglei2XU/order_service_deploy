# Custom hugging face trainner class
import logging
from typing import Optional, List, Dict

import torch
import numpy as np
import torch.nn as nn
from transformers import Trainer
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report
from transformers.trainer_utils import PredictionOutput

from utilis.constants import LABEL_PAD

EXCEPTION = {'tag_lengths', 'text_lengths', 'input_lengths'}


def to_device(batch_data, device):
    for key, value in batch_data.items():
        if key not in EXCEPTION:
            batch_data[key] = value.to(device)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        logit = model(**inputs).get('logits')
        loss_fct = nn.CrossEntropyLoss(ignore_index=LABEL_PAD)
        if len(logit.size()) > 2:
            labels = labels[:, 1:]
            batch_size, seq_len, _ = logit.size()
            loss = loss_fct(logit.reshape(batch_size * seq_len, -1), labels.reshape(-1))
        else:
            loss = loss_fct(logit.view(-1, self.model.num_labels), labels.view(-1))
        return (loss, logit) if return_outputs else loss

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        self._memory_tracker.start()
        all_pred, all_target = [], []
        report = {}
        correct = 0
        local_rank = self.args.local_rank
        if torch.cuda.is_available() and local_rank != -1:
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cpu')
        dataloader = self.get_eval_dataloader(eval_dataset=eval_dataset)
        for batch_data in dataloader:
            to_device(batch_data, device)
            target = batch_data['labels']
            output = self.model(**batch_data)
            pred_score = output.get('logits')
            pred_class = torch.argmax(pred_score, dim=-1)
            correct += torch.sum(pred_class == target).item()
            assert target.size(0) == pred_class.size(0)
            all_pred.append(pred_class)
            all_target.append(target)
        all_pred, all_target = torch.cat(all_pred, dim=0), torch.cat(all_target, dim=0)
        if all_pred.is_cuda:
            all_pred, all_target = all_pred.cpu().numpy(), all_target.cpu().numpy()
        else:
            all_pred, all_target = all_pred.numpy(), all_target.numpy()
        all_num = all_pred.shape[0]
        # pdb.set_trace()
        indies = all_pred == all_target
        # correct_array = all_pred[indies]
        correct_num = np.sum(indies)
        acc = correct_num / all_num
        report['eval_accuracy'] = acc
        self.log_metrics(split='eval', metrics={'accuracy': f'{acc}'})
        self.log_metrics(split='eval', metrics={'statistics': f'number of correct: {correct_num}; number of all '
                                                              f'samples: {all_num}'})

        return report

    def predict(
        self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput:
        self._memory_tracker.start()
        all_pred, all_target = [], []
        report = {}
        correct = 0
        local_rank = self.args.local_rank
        if torch.cuda.is_available() and local_rank != -1:
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cpu')
        dataloader = self.get_test_dataloader(test_dataset=test_dataset)
        for batch_data in dataloader:
            to_device(batch_data, device)
            target = batch_data['labels']
            output = self.model(**batch_data)
            pred_score = output.get('logits')
            pred_class = torch.argmax(pred_score, dim=-1)
            correct += torch.sum(pred_class == target).item()
            assert target.size(0) == pred_class.size(0)
            all_pred.append(pred_class)
            all_target.append(target)
        all_pred, all_target = torch.cat(all_pred, dim=0), torch.cat(all_target, dim=0)
        if all_pred.is_cuda:
            all_pred, all_target = all_pred.cpu().numpy(), all_target.cpu().numpy()
        else:
            all_pred, all_target = all_pred.numpy(), all_target.numpy()
        all_num = all_pred.shape[0]
        # pdb.set_trace()
        indies = all_pred == all_target
        # correct_array = all_pred[indies]
        correct_num = np.sum(indies)
        acc = correct_num / all_num
        report['test_accuracy'] = acc
        self.log_metrics(split='test', metrics={'accuracy': f'{acc}'})
        self.log_metrics(split='test', metrics={'statistics': f'number of correct: {correct_num}; number of all '
                                                              f'samples: {all_num}'})
        return PredictionOutput(predictions=all_pred, label_ids=None, metrics=report)


class Seq2SeqTrainer(CustomTrainer):
    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        self._memory_tracker.start()
        all_pred, all_target = [], []
        report = {}
        correct_sequence, all_sequence = 0, 0
        local_rank = self.args.local_rank
        if torch.cuda.is_available() and local_rank != -1:
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cpu')
        dataloader = self.get_eval_dataloader(eval_dataset=eval_dataset)
        for batch_data in dataloader:
            to_device(batch_data, device)
            target = batch_data['labels']  # (batch_size label_len)
            batch_size = target.size(0)
            output = self.model.decode(**batch_data)  # ModelOutput(logits, loss)
            pred_ = output.get('logits')  # (batch_size, label_len)
            for i in range(batch_size):
                all_sequence += 1
                valid_ = target[i, :] != LABEL_PAD
                if torch.equal(target[i, valid_], pred_[i, valid_]):
                    correct_sequence += 1
                # else:
                #     print(target[i], pred_[i])
            all_target.append(target.view(-1))
            all_pred.append(pred_.view(-1))

        all_target, all_pred = torch.cat(all_target, dim=0), torch.cat(all_pred, dim=0)
        if all_pred.is_cuda:
            all_pred, all_target = all_pred.cpu().numpy(), all_target.cpu().numpy()
        else:
            all_pred, all_target = all_pred.numpy(), all_target.numpy()
        assert all_pred.shape == all_target.shape
        valid_indies = all_target != LABEL_PAD
        valid_target = all_target[valid_indies]
        valid_pred = all_pred[valid_indies]
        correct_num = np.sum(valid_target == valid_pred)
        all_num = len(valid_target)
        acc = correct_num / all_num
        report['eval_accuracy'] = acc
        accuracy_sequence = correct_sequence / all_sequence
        self.log_metrics(split='eval', metrics={'accuracy': f'{acc}; accuracy in a view of sequence: {accuracy_sequence}'})
        self.log_metrics(split='eval', metrics={'statistics': f'Correct Number: {correct_num} Correct Sequence: {correct_sequence} All Samples: {all_num} All sequence: {all_sequence}'})
        return report

    def predict(
        self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput:
        """

        :param test_dataset:
        :param ignore_keys:
        :param metric_key_prefix:
        :return: PredictionOutput: logits, labels, metrics.
        """
        self._memory_tracker.start()
        all_pred, all_target = [], []
        report = {}
        correct_sequence, all_sequence = 0, 0
        local_rank = self.args.local_rank
        if torch.cuda.is_available() and local_rank != -1:
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cpu')
        dataloader = self.get_test_dataloader(test_dataset)
        for batch_data in dataloader:
            to_device(batch_data, device)
            target = batch_data['labels']  # (batch_size label_len)
            batch_size = target.size(0)
            output = self.model.decode(**batch_data)  # ModelOutput(logits, loss)
            pred_ = output.get('logits')  # (batch_size, label_len)
            for i in range(batch_size):
                all_sequence += 1
                valid_ = target[i, :] != LABEL_PAD
                if torch.equal(target[i, valid_], pred_[i, valid_]):
                    correct_sequence += 1
                all_target.append(target.view(-1))
                all_pred.append(pred_.view(-1))
        all_target, all_pred = torch.cat(all_target, dim=0), torch.cat(all_pred, dim=0)
        if all_pred.is_cuda:
            all_pred, all_target = all_pred.cpu().numpy(), all_target.cpu().numpy()
        else:
            all_pred, all_target = all_pred.numpy(), all_target.numpy()
        assert all_pred.shape == all_target.shape
        valid_indies = all_target != LABEL_PAD
        valid_target = all_target[valid_indies]
        valid_pred = all_pred[valid_indies]
        correct_num = np.sum(valid_target == valid_pred)
        all_num = len(valid_target)
        acc = correct_num / all_num
        report['test_accuracy'] = acc
        accuracy_sequence = correct_sequence / all_sequence
        self.log_metrics(split='test',
                         metrics={'accuracy': f'{acc}; accuracy in a view of sequence: {accuracy_sequence}'})
        self.log_metrics(split='test', metrics={
            'statistics': f'Correct Number: {correct_num} Correct Sequence: {correct_sequence} All Samples: {all_num} All sequence: {all_sequence}'})
        PredictionOutput(predictions=all_pred, label_ids=None, metrics=report)


def compute_metric(eval_):
    logit, label = eval_
    correct = 0
    pred_class = torch.argmax(logit, dim=-1)
    assert logit.size(0) == label.size(0)
    correct += torch.sum(pred_class == label).item()
    if logit.is_cuda:
        pred_class, label = pred_class.cpu().numpy(), label.cpu().numpy()
    else:
        pred_class, label = pred_class.numpy(), label.numpy()
    acc = accuracy_score(label, pred_class)
    report = classification_report(label, pred_class)
    report['eval_accuracy'] = acc
    return report
