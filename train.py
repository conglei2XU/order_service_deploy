import random

import torch
import torch.optim as optim
import numpy as np
from transformers import HfArgumentParser
from transformers import EarlyStoppingCallback
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report

from pipeline.batch_editor import CollateFn, CollateFnLight, CollateFnLightMultiple, CollateFnPretrainedMultiple
from pipeline.pre_training import PreTrainingLight, PreTraining
from pipeline.arguments import ModelArguments, CustomTrainArguments, ControlArguments
from pipeline.trainer import CustomTrainer, Seq2SeqTrainer


def init_args():
    parser = HfArgumentParser((ControlArguments, ModelArguments, CustomTrainArguments))
    control_arguments, model_arguments, train_arguments = parser.parse_args_into_dataclasses()
    return train_arguments, model_arguments, control_arguments


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False


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
    report['accuracy'] = acc
    return report


def train(train_arguments, model_arguments, control_arguments):
    fix_seed(train_arguments.seed)
    if control_arguments.task_name != 'location_info':
        task_name = 'others'
    else:
        task_name = 'location_info'
    if control_arguments.mode == 'pretrained':
        prepare = PreTraining(train_arguments, model_arguments, control_arguments)
        if control_arguments.dataset_meta == 'hierarchy_multiple':
            collate_fn_meta = CollateFnPretrainedMultiple
        else:
            collate_fn_meta = CollateFn
    else:
        prepare = PreTrainingLight(train_arguments, model_arguments, control_arguments)
        if control_arguments.dataset_meta == 'hierarchy_multiple':
            collate_fn_meta = CollateFnLightMultiple
        else:
            collate_fn_meta = CollateFnLight

    tokenizer, model = prepare.prepare_model()
    collate_fn = collate_fn_meta(tokenizer=tokenizer, label2idx=prepare.label2idx, is_split=control_arguments.is_split, task_name=task_name)
    # set this attributes for the evaluation
    # class level post initializing processing
    setattr(train_arguments, 'label_names', list(prepare.idx2label.keys()))

    train_loader, val_loader, test_loader = prepare.create_loader(collate_fn=collate_fn)

    # for batch_data in train_loader:
    #     print(batch_data.keys())

    # model.to(device=device)
    # optimizer, scheduler, loss_fn = prepare.prepare_optimizer(model, train_loader)
    def preprocess_logits_for_metrics(logit, labels):
        if logit is tuple:
            return logit[0]
        else:
            return logit

    print(f'Model Type: {control_arguments.model_meta} Mode: {control_arguments.mode}')
    if control_arguments.model_meta == 'seq2seq':
        print(f'Launching seq2seq trainer...')
        param_groups = [{'params': [p for n, p in model.named_parameters() if 'decoder' in n],
                         'lr': 1e-3},
                        {'params': [p for n, p in model.named_parameters() if 'decoder' not in n]}]
        optimizer = optim.AdamW(param_groups, lr=train_arguments.learning_rate)
        # scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=100)
        trainer = Seq2SeqTrainer(
            model=model,
            args=train_arguments,
            train_dataset=prepare.train,
            eval_dataset=prepare.val,
            # tokenizer=tokenizer,
            data_collator=collate_fn,
            compute_metrics=compute_metric,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            optimizers=(optimizer, None)
        )
    else:
        print(f'Launching classification trainer......')
        # param_groups = [{'params': [p for n, p in model.named_parameters() if n != "text_model"],
        #                  'lr': 1e-3},
        #                 {'params': [p for n, p in model.named_parameters() if n == 'text_model'], 'lr': train_arguments.learning_rate}]
        # optimizer = optim.AdamW(param_groups)
        trainer = CustomTrainer(
            model=model,
            args=train_arguments,
            train_dataset=prepare.train,
            eval_dataset=prepare.val,
            # tokenizer=tokenizer,
            data_collator=collate_fn,
            compute_metrics=compute_metric,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    if train_arguments.do_train:
        trainer.train()
    if train_arguments.do_predict:
        all = trainer.predict(prepare.test)
    return trainer


def main():
    train_arguments, model_arguments, control_arguments = init_args()
    train(train_arguments, model_arguments, control_arguments)


if __name__ == "__main__":
    main()
