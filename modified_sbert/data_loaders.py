import random
import logging
from typing import Union, List
import numpy as np

from sentence_transformers.readers import InputExample
from torch.utils.data import  IterableDataset


import cluster_fns

logger = logging.getLogger(__name__)


def load_data_as_individuals(data, type):

    sentence_1_list = data['sentence_1']
    sentence_2_list = data['sentence_2']
    labels = data['labels']

    # Organise by cluster
    edges_list = []
    for i in range(len(sentence_1_list)):
        if labels[i] == "same":
            edges_list.append([sentence_1_list[i], sentence_2_list[i]])

    cluster_dict = cluster_fns.clusters_from_edges(edges_list)

    # Pull out texts and cluster IDs
    indv_data = []
    guid = 1
    for cluster_id in list(cluster_dict.keys()):

        for text in cluster_dict[cluster_id]:
            indv_data.append(InputExample(guid=guid, texts=[text], label=cluster_id))

            guid += 1

    print(f'{len(indv_data)} {type} examples')

    return indv_data


def load_data_as_pairs(data, type):

    sentence_1_list = data['sentence_1']
    sentence_2_list = data['sentence_2']
    labels = data['labels']

    label2int = {"same": 1, "different": 0, 1: 1, 0: 0}

    paired_data = []
    for i in range(len(sentence_1_list)):
        label_id = label2int[labels[i]]
        paired_data.append(InputExample(texts=[sentence_1_list[i], sentence_2_list[i]], label=float(label_id)))

    print(f'{len(paired_data)} {type} pairs')

    return paired_data


def load_data_as_triplets(data, type):

    sentence_1_list = data['sentence_1']
    sentence_2_list = data['sentence_2']
    labels = data['labels']

    # Create dict of examples where you have labels, at the anchor level
    def add_to_samples(sent1, sent2, label):
        if sent1 not in anchor_dict:
            anchor_dict[sent1] = {'same': set(), 'different': set()}
        anchor_dict[sent1][label].add(sent2)

    anchor_dict = {}
    for i in range(len(sentence_1_list)):
        add_to_samples(sentence_1_list[i], sentence_2_list[i], labels[i])
        add_to_samples(sentence_2_list[i], sentence_1_list[i], labels[i])  #Also add the opposite

    # Create triplets
    triplet_data = []
    for anchor, others in anchor_dict.items():
        while len(others['same']) > 0 and len(others['different']) > 0:

            same_sent = random.choice(list(others['same']))
            dif_sent = random.choice(list(others['different']))

            triplet_data.append(InputExample(texts=[anchor, same_sent, dif_sent]))

            others['same'].remove(same_sent)
            others['different'].remove(dif_sent)

    print(f'{len(triplet_data)} {type} triplets')

    return triplet_data


def load_data_as_lambert_pairs(data, type):

    sentence_1_list = data['sentence_1']
    sentence_2_list = data['sentence_2']
    bbox_1_list = data['bbox_1']
    bbox_2_list = data['bbox_2']
    labels = data['labels']

    label2int = {"same": 1, "different": 0, 1: 1, 0: 0}

    paired_data = []
    for i in range(len(sentence_1_list)):
        label_id = label2int[labels[i]]
        paired_data.append(Lambert_InputExample(
            texts=[sentence_1_list[i], sentence_2_list[i]],
            bboxes=[bbox_1_list[i], bbox_2_list[i]],
            label=float(label_id)
        ))

    print(f'{len(paired_data)} {type} pairs')

    return paired_data


class HardBatchDataset(IterableDataset):

    def __init__(self, examples: List[InputExample], samples_per_batch: int = 16):

        super().__init__()

        self.samples_per_batch = samples_per_batch

        #Group examples by subbatch
        batched_examples = [examples[i * self.samples_per_batch:(i + 1) * self.samples_per_batch] for i in range((len(examples) + self.samples_per_batch - 1) // self.samples_per_batch)]
        print(len(batched_examples))
        batch2ex = {}
        for b, batch in enumerate(batched_examples):
            batch2ex[b] = batch

        #Include only batchs with at least 2 examples
        self.grouped_inputs = []
        self.groups_right_border = []
        num_batch = 0

        for batch, batch_examples in batch2ex.items():
            if len(batch_examples) >= self.samples_per_batch:
                self.grouped_inputs.extend(batch_examples)
                self.groups_right_border.append(len(self.grouped_inputs))  # At which position does this subbatch group / bucket end?
                num_batch += 1

        self.batch_range = np.arange(num_batch)
        np.random.shuffle(self.batch_range)

        logger.info("HardBatchDataset: {} examples, from which {} examples could be used (grouped in {} different subbatches of size {}).".format(len(examples), len(self.grouped_inputs),num_batch, self.samples_per_batch))

    def __iter__(self):
        batch_idx = 0
        count = 0
        already_seen = {}
        while count < len(self.grouped_inputs):
            batch = self.batch_range[batch_idx]
            if batch not in already_seen:
                already_seen[batch] = set()

            left_border = 0 if batch == 0 else self.groups_right_border[batch-1]
            right_border = self.groups_right_border[batch]

            selection = [i for i in np.arange(left_border, right_border) if i not in already_seen[batch]]

            if len(selection) >= self.samples_per_batch:
                for element_idx in np.random.choice(selection, self.samples_per_batch, replace=False):
                    count += 1
                    already_seen[batch].add(element_idx)
                    yield self.grouped_inputs[element_idx]

            batch_idx += 1
            if batch_idx >= len(self.batch_range):
                batch_idx = 0
                already_seen = {}
                np.random.shuffle(self.batch_range)

    def __len__(self):
        return len(self.grouped_inputs)


# class HardBatchDataset(IterableDataset):

#     def __init__(self, examples: List[InputExample], samples_per_label: int = 8):

#         super().__init__()

#         self.samples_per_label = samples_per_label

#         #Group examples by label
#         label2ex = {}
#         for example in examples:
#             if example.label not in label2ex:
#                 label2ex[example.label] = []
#             label2ex[example.label].append(example)

#         #Include only labels with at least 2 examples
#         self.grouped_inputs = []
#         self.groups_right_border = []
#         num_labels = 0

#         for label, label_examples in label2ex.items():
#             if len(label_examples) >= self.samples_per_label:
#                 self.grouped_inputs.extend(label_examples)
#                 self.groups_right_border.append(len(self.grouped_inputs))  # At which position does this label group / bucket end?
#                 num_labels += 1

#         self.label_range = np.arange(num_labels)
#         np.random.shuffle(self.label_range)

#         logger.info("SentenceLabelDataset: {} examples, from which {} examples could be used (those labels appeared at least {} times). {} different labels found.".format(len(examples), len(self.grouped_inputs), self.samples_per_label, num_labels ))

#     def __iter__(self):
#         label_idx = 0
#         count = 0
#         already_seen = {}
#         while count < len(self.grouped_inputs):
#             label = self.label_range[label_idx]
#             if label not in already_seen:
#                 already_seen[label] = set()

#             left_border = 0 if label == 0 else self.groups_right_border[label-1]
#             right_border = self.groups_right_border[label]

#             selection = [i for i in np.arange(left_border, right_border) if i not in already_seen[label]]

#             if len(selection) >= self.samples_per_label:
#                 for element_idx in np.random.choice(selection, self.samples_per_label, replace=False):
#                     count += 1
#                     already_seen[label].add(element_idx)
#                     yield self.grouped_inputs[element_idx]

#             label_idx += 1
#             if label_idx >= len(self.label_range):
#                 label_idx = 0
#                 already_seen = {}
#                 np.random.shuffle(self.label_range)

#     def __len__(self):
#         return len(self.grouped_inputs)


class Lambert_InputExample(InputExample):

    """
    Update of sbert's InputExample to include bounding boxes
    """
    def __init__(self, guid: str = '', texts: List[str] = None,  bboxes: List[str] = None, label: Union[int, float] = 0):

        self.guid = guid
        self.texts = texts
        self.label = label
        self.bboxes = bboxes
    """
    Based on SentenceLabelDataset. This dataset can be used for specific loss function which require multiple
    positives and hard negatives in the same batch. 
    It draws n consecutive, random and unique samples from one label at a time. This is repeated for each label.
    Labels with fewer than n unique samples are ignored.
    All draws are without replacement.
    """
    def __init__(self, examples: List[InputExample], samples_per_label: int = 8, batch_size: int = 16):
        """
        Creates a LabelSampler for a SentenceLabelDataset.
        :param examples:
            a list of InputExamples of form texts=[sentence 1, sentence 2], label=float
        :param samples_per_label:
            the number of positive samples drawn per label. The same number of negatives will be drawn. 
            Batch size should be a multiple of 2 * samples_per_label
        """
        super().__init__()

        self.samples_per_label = samples_per_label
        self.batch_size = batch_size

        if self.batch_size % self.samples_per_label != 0:
            raise ValueError("samples_per_label must be a divisor of batch_size")

        #Group examples by label
        label2ex = {}
        for example in examples:
            if example.label not in label2ex:
                label2ex[example.label] = []
            label2ex[example.label].append(example)

        #Include only labels with at least n examples
        self.grouped_inputs = []
        self.groups_right_border = []
        num_labels = 0

        for label, label_examples in label2ex.items():
            if len(label_examples) >= self.samples_per_label:
                self.grouped_inputs.extend(label_examples)
                self.groups_right_border.append(len(self.grouped_inputs))  # At which position does this label group / bucket end?
                num_labels += 1

        self.label_range = np.arange(num_labels)   # like base python range, but a np array 
        np.random.shuffle(self.label_range)

        logger.info("SentenceLabelDataset: {} examples, from which {} examples could be used (those labels appeared at least {} times). {} different labels found.".format(len(examples), len(self.grouped_inputs), self.samples_per_label, num_labels ))

    def __iter__(self):

        label_idx = 0        # Used to iterate through self.label_range 
        count = 0            # Used to keep track of number of texts added so far 
        already_seen = {}

        while count < len(self.grouped_inputs):
            label = self.label_range[label_idx]         # Pick a cluster ID (these have been randomly shuffled)

            if label not in already_seen:
                already_seen[label] = set()

            left_border = 0 if label == 0 else self.groups_right_border[label-1]     # At which position does this label group / bucket begin?
            right_border = self.groups_right_border[label]

            selection = [i for i in np.arange(left_border, right_border) if i not in already_seen[label]]   # all texts in clusters that not already seen 

            if len(selection) >= self.samples_per_label:
                for element_idx in np.random.choice(selection, self.samples_per_label, replace=False):
                    count += 1
                    already_seen[label].add(element_idx)
                    yield self.grouped_inputs[element_idx]

            label_idx += 1
            if label_idx >= len(self.label_range):
                label_idx = 0
                already_seen = {}
                np.random.shuffle(self.label_range)

    def __len__(self):
        return len(self.grouped_inputs)
