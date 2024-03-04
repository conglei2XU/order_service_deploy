from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    encoder_layers: int = field(default=1, metadata={'help': 'number of encoder layers'})
    decoder_layers: int = field(default=1, metadata={'help': 'number of decoder layers'})
    dropout: float = field(default=0.3, metadata={'help': 'rate for dropout function'})
    char_hidden_dim: int = field(default=30, metadata={'help': 'dimension of char feature vector'})
    input_dim: int = field(default=768, metadata={'help': 'the last dimension of input tensor'})
    word_vector_text: str = field(default='Word2Vector/sgns.weibo.bigram', metadata={'help': 'location of the '
                                                                                             'word2vector'
                                                                                             'file'})
    word_vector_tag: str = field(default='config/config_Qinghai/serv_type/label_embedding.pt')
    label_vector: str = field(default='config/config_Qinghai/label_embedding_.pt',
                              metadata={'help': 'location of the label embedding'})
    word_vector_dim: int = field(default=300, metadata={'help': 'dimension of word2vector'})
    tag_vector_dim: int = field(default=768, metadata={'help': 'dimension of embedding for tags'})
    use_vector: bool = field(default=True, metadata={'help': 'whether use vector or not'})
    num_layers: int = field(default=2, metadata={'help': 'number of layers for RNNs'})
    hidden_dim: int = field(default=768, metadata={'help': 'dimension for output of tensors'})
    model_path: str = field(default='pretrained_models/albert', metadata={'help': 'path to loading initialized model '
                                                                                  'weights from huggingface hub'})
    label_dim: int = field(default=100, metadata={'help': 'dimension of label calsses when perform task in a decoding '
                                                          'way'})
    encoder_dim: int = field(default=600, metadata={'help': 'size of the encoder neural network'})
    decoder_dim: int = field(default=600, metadata={'help': 'size of the decoder neural network'})
    text_hidden_dim: int = field(default=300, metadata={'help': 'hidden size of the text model'})
    tag_hidden_dim: int = field(default=300, metadata={'help': 'hidden size of the tag model'})


@dataclass
class ControlArguments:
    dataset_meta: str = field(default='general', metadata={'help': '[general, hierarchy, hierarchy_multiple]'})
    model_meta: str = field(default='classification_general', metadata={'help': '[rnn_classification, seq2seq, '
                                                                 'pretrained_classification]'})
    reader: str = field(default='csv', metadata={'help': 'reader for the dataset'})
    label_mapping: str = field(default='config/config_Qinghai/duty_reason/label_mapping_cls.json',
                               metadata={'help': 'file path for the predefined label2idx mapping'})
    mode: str = field(default='pretrained', metadata={'help': 'training mode choice [pretrained, light]'})
    # seed: int = field(default=42, metadata={'help': 'random seed for torch, random and numpy'})
    dataset_path: str = field(default='Dataset/order_Qinghai')
    cached_tokenizer: str = field(default='cache/tokenizer.bin')
    best_model_path: str = field(default='model/')
    do_inference: bool = field(default=False)
    task_name: str = field(default='duty_reason')
    token_to_idx: str = field(default='config/config_Qinghai/serv_type/label_mapping.json')
    is_split: bool = field(default=False)


@dataclass
class CustomTrainArguments(TrainingArguments):
    evaluation_strategy: str = field(default='steps')
    output_dir: str = field(default='saved_model')
    per_device_train_batch_size: int = field(default=32)
    learning_rate: float = field(default=5e-3)
    num_train_epochs: int = field(default=10)
    eval_steps: int = field(default=1000)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default='accuracy')
    local_rank: int = field(default=-1)
    do_train: bool = field(default=True)
    do_eval: bool = field(default=True)
    do_predict: bool = field(default=True)
    save_steps: int = field(default=1000)
    log_level: str = field(default='info')
    save_strategy: str = field(default='steps')
    logging_dir: str = field(default='log/')
    logging_steps: int = field(default=500)
    warmup_ratio: int = field(default=0.1)
    resume_from_checkpoint: str = field(default=r'D:\python project\OrderAnalysis\saved_model')
    save_safetensors: bool = field(default=False)