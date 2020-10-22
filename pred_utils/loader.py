import os
import numpy as np
from copy import deepcopy
from collections import Counter
from collections import OrderedDict
from ordered_set import OrderedSet

class Alphabet(object):
    """
    Storage and serialization a set of elements.
    """

    def __init__(self, name, if_use_pad, if_use_unk):

        self.__name = name
        self.__if_use_pad = if_use_pad
        self.__if_use_unk = if_use_unk

        self.__index2instance = OrderedSet()
        self.__instance2index = OrderedDict()

        self.__counter = Counter()

    def name(self):
        return self.__name

    def get_index(self, instance):
        """ Serialize given instance and return.
        For unknown words, the return index of alphabet
        depends on variable self.__use_unk:
            1, If True, then return the index of "<UNK>";
            2, If False, then return the index of the
            element that hold max frequency in training data.
        :param instance: is given instance or a list of it.
        :return: is the serialization of query instance.
        """

        if isinstance(instance, (list, tuple)):
            return [self.get_index(elem) for elem in instance]

        assert isinstance(instance, str)

        try:
            return self.__instance2index[instance]
        except KeyError:
            if self.__if_use_unk:
                return self.__instance2index[self.__sign_unk]
            else:
                max_freq_item = self.__counter.most_common(1)[0][0]
                return self.__instance2index[max_freq_item]

    def get_instance(self, index):
        """ Get corresponding instance of query index.
        if index is invalid, then throws exception.
        :param index: is query index, possibly iterable.
        :return: is corresponding instance.
        """

        if isinstance(index, list):
            return [self.get_instance(elem) for elem in index]

        return self.__index2instance[index]

    def load_content(self, dir_path):
        """ Save the content of alphabet to files.
        There are two kinds of saved files:
            1, The first is a list file, elements are
            sorted by the frequency of occurrence.
            2, The second is a dictionary file, elements
            are sorted by it serialized index.
        :param dir_path: is the directory path to save object.
        """

        # Check if dir_path exists.
        if not os.path.exists(dir_path):
            raise Exception("Path does not exist.")

        list_path = os.path.join(dir_path, self.__name + "_list.txt")
        with open(list_path, 'r') as fr:
            for line in fr.readlines():
                items = line.split()
                self.__counter[items[0]] = int(items[1])

        dict_path = os.path.join(dir_path, self.__name + "_dict.txt")
        with open(dict_path, 'r') as fr:
            for line in fr.readlines():
                items = line.split()
                self.__instance2index[items[0]] = int(items[1])
                self.__index2instance.append(items[0])

    def __len__(self):
        return len(self.__index2instance)

    def __str__(self):
        return 'Alphabet {} contains about {} words: \n\t{}'.format(self.name, len(self), self.__index2instance)


class TorchDataset(Dataset):
    """
    Helper class implementing torch.utils.data.Dataset to
    instantiate DataLoader which deliveries data batch.
    """

    def __init__(self, text):
        self.__text = text

    def __getitem__(self, index):
        return self.__text[index]

    def __len__(self):
        # Pre-check to avoid bug.
        return len(self.__text)

class DatasetManager(object):

    def __init__(self, args):

        # Instantiate alphabet objects.
        self.__word_alphabet = Alphabet('word', if_use_pad=True, if_use_unk=True)
        self.__intent_alphabet = Alphabet('intent', if_use_pad=False, if_use_unk=False)

        # Record the raw text of dataset.
        self.__text_word_data = {}

        # Record the serialization of dataset.
        self.__digit_word_data = {}

        self.__args = args

    def get_dataset(self, data_name, is_digital):
        """ Get dataset of given unique name.
        :param data_name: is name of stored dataset.
        :param is_digital: make sure if want serialized data.
        :return: the required dataset.
        """

        if is_digital:
            return self.__digit_word_data[data_name]
        else:
            return self.__text_word_data[data_name]

    def add_file(self, file_path, data_name, if_train_file):
        text, intent = self.__read_file(file_path)

        # Record the raw text of dataset.
        self.__text_word_data[data_name] = text

        # Serialize raw text and stored it.
        self.__digit_word_data[data_name] = self.__word_alphabet.get_index(text)

    def __read_file(file_path):
        """ Read data file of given path.
        :param file_path: path of data file.
        :return: list of sentence and list of intent.
        """

        texts = []

        with open(file_path, 'r') as fr:
            for line in fr.readlines():
                items = line.strip().split()
                for item in items:
	texts.append([item.strip()])

        return texts

    def batch_delivery(self, data_name, batch_size=None, is_digital=True, shuffle=True):
        if batch_size is None:
            batch_size = 1

        if is_digital:
            text = self.__digit_word_data[data_name]
        else:
            text = self.__text_word_data[data_name]
        dataset = TorchDataset(text)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.__collate_fn)


    def add_padding(texts, items=None, digital=True):
        len_list = [len(text) for text in texts]
        max_len = max(len_list)

        # Get sorted index of len_list.
        sorted_index = np.argsort(len_list)[::-1]

        trans_texts, seq_lens, trans_items = [], [], None
        if items is not None:
            trans_items = [[] for _ in range(0, len(items))]

        for index in sorted_index:
            seq_lens.append(deepcopy(len_list[index]))
            trans_texts.append(deepcopy(texts[index]))
            if digital:
                trans_texts[-1].extend([0] * (max_len - len_list[index]))
            else:
                trans_texts[-1].extend(['<PAD>'] * (max_len - len_list[index]))

            # This required specific if padding after sorting.
            if items is not None:
                for item, (o_item, required) in zip(trans_items, items):
                    item.append(deepcopy(o_item[index]))
                    if required:
                        if digital:
                            item[-1].extend([0] * (max_len - len_list[index]))
                        else:
                            item[-1].extend(['<PAD>'] * (max_len - len_list[index]))

        if items is not None:
            return trans_texts, trans_items, seq_lens
        else:
            return trans_texts, seq_lens

    @staticmethod
    def __collate_fn(batch):
        """
        helper function to instantiate a DataLoader Object.
        """

        n_entity = len(batch[0])
        modified_batch = [[] for _ in range(0, n_entity)]

        for idx in range(0, len(batch)):
            for jdx in range(0, n_entity):
                modified_batch[jdx].append(batch[idx][jdx])

        return modified_batch