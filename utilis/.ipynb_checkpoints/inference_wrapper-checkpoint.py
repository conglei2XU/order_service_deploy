import os
import sys
import json
from typing import Iterable

import torch
from transformers import AutoModel

from pipeline.pre_training import PreTraining, PreTrainingLight
from pipeline.batch_editor import CollateFnLight, CollateFn, CollateFnPretrainedMultiple, CollateFnLightMultiple
from pipeline.trainer import Seq2SeqTrainer, CustomTrainer


class InferenceWrapper:
    """
    wrap the models to make perform inference action
    """

    def __init__(self, train_arguments, model_arguments, control_arguments, task_specific_kwargs):
        """

        :param train_arguments:
        :param model_arguments:
        :param control_arguments:
        :param task_specific_kwargs: [path,str] {mode: [light, pretrained],
        dataset_meta: [general, hierarchy, hierarchy_multiple,
        code_table: str}
        """

        for key, value in task_specific_kwargs.items():
            # rewrite label_mapping location, best_model_path location, model_mode.
            # five tasks can use same tokenizer because of same input content.
            if hasattr(control_arguments, key):
                setattr(control_arguments, key, value)
        # setattr(model_arguments, 'cache_tokenizer', task_specific_kwargs['cache_tokenizer'])
        assert control_arguments.do_inference
        try:
            self.code_table = json.load(open(task_specific_kwargs['code_table'], 'r'))
        except KeyError:
            print(f' code_table is not existed in {task_specific_kwargs.keys()}')
            sys.exit(1)
        except Exception as e:
            print(f'Loading code_table from {task_specific_kwargs["code_table"]} errors!')
            sys.exit(1)
        if control_arguments.task_name != 'location_info':
            task_name = 'others'
        else:
            task_name = 'location_info'
        print(f'Model: {control_arguments.model_meta}. Mode: {control_arguments.dataset_meta}. Task: {control_arguments.task_name}')
        if control_arguments.mode == 'pretrained':
            self.prepare = PreTraining(train_arguments, model_arguments, control_arguments)
            if control_arguments.dataset_meta == 'hierarchy_multiple':
                collate_fn_meta = CollateFnPretrainedMultiple
            else:
                collate_fn_meta = CollateFn
        else:
            self.prepare = PreTrainingLight(train_arguments, model_arguments, control_arguments)
            if control_arguments.dataset_meta == 'hierarchy_multiple':
                collate_fn_meta = CollateFnLightMultiple
            else:
                collate_fn_meta = CollateFnLight
        if torch.cuda.is_available() and train_arguments.local_rank != -1:
            device = torch.device(f'cuda:{train_arguments.local_rank}')
        else:
            device = torch.device('cpu')
        tokenizer, model = self.prepare.prepare_model()
        self.collate_fn = collate_fn_meta(tokenizer=tokenizer, label2idx=self.prepare.label2idx,
                                     is_split=control_arguments.is_split, task_name=task_name)
        setattr(train_arguments, 'label_names', list(self.prepare.idx2label.keys()))
        setattr(self.collate_fn, 'tokenizer', tokenizer)
        print(f'loading model from {control_arguments.best_model_path} ...')
        try:
            best_model = task_specific_kwargs['best_model_path']

            if control_arguments.model_meta == 'seq2seq':
                # scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=100)
                trainer = Seq2SeqTrainer(
                    model=model,
                    args=train_arguments,
                    train_dataset=self.prepare.train,
                    eval_dataset=self.prepare.val,
                    # tokenizer=tokenizer,
                    data_collator=self.collate_fn,
                )

            else:
                trainer = CustomTrainer(
                    model=model,
                    args=train_arguments,
                    train_dataset=self.prepare.train,
                    eval_dataset=self.prepare.val,
                    # tokenizer=tokenizer,
                    data_collator=self.collate_fn,
                )
            trainer._load_from_checkpoint(best_model)
            self.model = trainer.model
            self.model.to(device)
            self.model.eval()
        except FileNotFoundError:
            print(f"{best_model} is not existed")
            sys.exit(1)
        except Exception as e:
            print(f" Error loading model: {repr(e)}")
            sys.exit(1)
        if control_arguments.model_meta == 'seq2seq':
            print(f'{control_arguments.task_name} launches in a seq2seq manner.')
        else:
            print(f'{control_arguments.task_name} launches in a classification manner. ')
        self.task_name = control_arguments.task_name

        # print(f'Model Type: {control_arguments.model_meta} Mode: {control_arguments.mode}')
    def __call__(self, serv_content, serv_type):
        pred_hanzi, pred_code = self.predict(serv_content, serv_type)
        return pred_hanzi, pred_code

    def predict(self, serv_content, serv_type, **kwargs):
        """

        :param serv_content:
        :param serv_type:
        :param kwargs:
        :return:
        """
        # print(serv_content)
        if isinstance(serv_content, str) and isinstance(serv_type, str):
            batchify_input = self.collate_fn.inference(serv_content, serv_type, **kwargs)
        else:
            print(f'Unsupported datatype: {type(serv_content)} {type(serv_type)}')
            sys.exit(1)
        # ModelOutput classes (batch_size, seq_len)
        if self.prepare.model_meta == 'seq2seq':
            model_output = self.model.decode(**batchify_input)
        else:
            model_output = self.model(**batchify_input)
        pred_label_hanzi, pred_label_code = self.post_processing(model_output)
        return pred_label_hanzi, pred_label_code

    def post_processing(self, model_output):
        """
        covert idx to label and then finds its corresponding code in database.
        :return:
        """
        logits = model_output.get('logits', None)
        # (batch_size, label_length) or (batch_size,)
        # print(logits.size())
        pred_class = torch.argmax(logits, dim=-1)
        batch_size = pred_class.size(0)
        pred_label_code = []
        # print(pred_class.size())
        if len(pred_class.size()) > 1:
            pred_label_hanzi = []
            for per_sample in range(batch_size):
                tmp = [self.prepare.idx2label[i.item()]for i in pred_class[per_sample]]
                pred_label_hanzi.append('>>'.join(tmp))
        else:
            pred_label_hanzi = [self.prepare.idx2label[i.item()] for i in pred_class]
        try:
            for item in pred_label_hanzi:
                tmp = self.code_table[item]
                pred_label_code.append(tmp)
        except KeyError:
            print(self.task_name, item)
            sys.exit(1)
        return pred_label_hanzi, pred_label_code


if __name__ == '__main__':
    pass

