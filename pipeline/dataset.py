from collections.abc import Iterable

from torch.utils.data import Dataset

from utilis.constants import PRE_TOKENS_LABEL

START = '<START>'
END = '<END>'


# PREFIX= f'标签位置{idx + 1}:'

class GeneralDataset(Dataset):
    """
    Dataset class handles (content, label) and (content, [label]).
    The concrete output depended on the behaviour of reader function.
    """

    def __init__(self, dataset_path, reader, label2idx=None, specific_tokens=None, is_seq2seq=False):
        # self.all_samples: list[ list[str,] ]; self.labels: list[ list[str] ]
        self.all_samples, self.labels = reader(dataset_path)
        if label2idx is not None:
            self.label2idx = label2idx
        else:
            self.label2idx = specific_tokens if specific_tokens else {}
            if is_seq2seq:
                self._gen_label2idx()

    def _gen_label2idx(self):
        for label in self.labels:
            if isinstance(label, Iterable) and not isinstance(label, str):
                for inner_label in label:
                    if inner_label not in self.label2idx:
                        self.label2idx[inner_label] = len(self.label2idx)
            else:
                if label not in self.label2idx:
                    self.label2idx[label] = len(self.label2idx)

    def __getitem__(self, item):
        try:
            label_cur = self.labels[item]
        except IndexError:
            print(f'Item: {item}. Number of labels: {len(self.labels)}')
        if isinstance(label_cur, Iterable) and not isinstance(label_cur, str):
            label_idx_cur = [self.label2idx.get(i, None) for i in label_cur]
            for i in label_idx_cur:
                if i is None:
                    raise KeyError(f"found unexisted key in {label_cur}")
        else:
            label_idx_cur = self.label2idx.get(label_cur, None)
            if label_idx_cur is None:
                raise KeyError(f"{label_cur} doesn't exist in label list")
        return self.all_samples[item], label_idx_cur

    def __len__(self):
        return len(self.all_samples)


class HierarchyDataset(GeneralDataset):
    """
    HierarchyDataset aims to build a dataset, converting classification task
    into decoding task with a hierarchy structure.
    it needs to do two more things:
    1. add <start> and <end> token in labels of each content
    2. manage labels into hierarchy dataset (optional)
    """

    def __init__(self, dataset_path, reader, label2idx=None, specific_tokens=None, is_seq2seq=False):

        super().__init__(dataset_path, reader, label2idx, specific_tokens, is_seq2seq)

    def _gen_label2idx(self):
        """
        generate label2idx dictionary for hierarchy dataset each label split into a sequence labels
        eg. [a>>b>>c] -> [[a, b, c]]
        :return:
        """
        new_format_labels = []
        for label in self.labels:
            sequence_label = [f'标签位置{idx_ + 1}:' + label for idx_, label in enumerate(label.strip().split('>>'))]
            assert len(sequence_label) > 2
            sequence_label = [START] + sequence_label + [END]
            new_format_labels.append(sequence_label)
            for inner_label in sequence_label:
                if inner_label not in self.label2idx:
                    self.label2idx[inner_label] = len(self.label2idx)
            self.labels = new_format_labels


class HierarchyDatasetMultiple:
    """
    HierarchyDatasetMultiple dataset aims to generate handle three input sample:
    for labels, it will be converted a sequence labels as HierarchyDataset,
    input: [text (serv_content), tags (serv_type)] will be used
    """

    def __init__(self, dataset_path, reader, label2idx=None, specific_tokens=None, is_seq2seq=False):
        self.texts, self.tags, self.labels = reader(dataset_path)
        if label2idx is not None:
            self.label2idx = label2idx
        else:
            self.label2idx = specific_tokens if specific_tokens else {}
            if is_seq2seq:
                self._gen_label2idx()

    def __getitem__(self, item):

        text, tag, label = self.texts[item], self.tags[item], self.labels[item]
        if isinstance(label, Iterable) and not isinstance(label, str):
            label_idx_cur = [self.label2idx.get(i, None) for i in label]
            for i in label_idx_cur:
                if i is None:
                    raise KeyError(f"found unexisted key in {label}")
        else:
            label_idx_cur = self.label2idx.get(label, None)
            if label_idx_cur is None:
                raise KeyError(f"{label} doesn't exist in label list")
        return text, tag, label_idx_cur

    def _gen_label2idx(self):
        """
        generate label2idx dictionary for hierarchy dataset each label split into a sequence labels
        eg. [a>>b>>c] -> [[a, b, c]]
        :return:
        """
        new_format_labels = []
        for label in self.labels:
            sequence_label = [f'标签位置{idx_ + 1}:' + label for idx_, label in enumerate(label.strip().split('>>'))]
            assert len(sequence_label) > 2
            sequence_label = [START] + sequence_label + [END]
            new_format_labels.append(sequence_label)
            assert len(sequence_label) > 2
            for inner_label in sequence_label:
                if inner_label not in self.label2idx:
                    self.label2idx[inner_label] = len(self.label2idx)
            self.labels = new_format_labels

    def __len__(self):
        return len(self.tags)
