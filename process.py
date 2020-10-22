import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import os
import time
import random
import numpy as np
from tqdm import tqdm
from collections import Counter



class Processor(object):

    def __init__(self, dataset, model, batch_size):
        self.__dataset = dataset
        self.__model = model
        self.__batch_size = batch_size

        if torch.cuda.is_available():
            time_start = time.time()
            self.__model = self.__model.cuda()

            time_con = time.time() - time_start
            print("The model has been loaded into GPU and cost {:.6f} seconds.\n".format(time_con))

        )


    @staticmethod
    def validate(model_path, dataset_path, batch_size):
        """
        validation will write mistaken samples to files and make scores.
        """

        model = torch.load(model_path)
        dataset = torch.load(dataset_path)

        # Get the sentence list in test dataset.
        sent_list = dataset.test_sentence

        exp_pred_intent, real_intent, pred_intent = Processor.prediction(
            model, dataset, "test", batch_size
        )
        print(exp_pred_intent, real_intent, pred_intent)

        # To make sure the directory for save error prediction.
        mistake_dir = os.path.join(dataset.save_dir, "error")
        if not os.path.exists(mistake_dir):
            os.mkdir(mistake_dir)

        intent_file_path = os.path.join(mistake_dir, "intent.txt")
        both_file_path = os.path.join(mistake_dir, "both.txt")

        # Write those sample with mistaken intent prediction.
        with open(intent_file_path, 'w') as fw:
            for w_list, p_intent_list, r_intent, p_intent in zip(sent_list, pred_intent, real_intent, exp_pred_intent):
                if p_intent != r_intent:
                    for w, p in zip(w_list, p_intent_list):
                        fw.write(w + '\t' + p + '\n')
                    fw.write(r_intent + '\t' + p_intent + '\n\n')

        intent_acc = Evaluator.accuracy(exp_pred_intent, real_intent)
        sent_acc = Evaluator.semantic_acc(exp_pred_intent, real_intent)

        return intent_acc, sent_acc

    @staticmethod
    def prediction(model, dataset, mode, batch_size):
        model.eval()

        if mode == "dev":
            dataloader = dataset.batch_delivery('dev', batch_size=batch_size, shuffle=False, is_digital=False)
        elif mode == "test":
            dataloader = dataset.batch_delivery('test', batch_size=batch_size, shuffle=False, is_digital=False)
        else:
            raise Exception("Argument error! mode belongs to {\"dev\", \"test\"}.")
            
        pred_intent, real_intent = [], []

        for text_batch, intent_batch in tqdm(dataloader, ncols=50):
            padded_text, seq_lens = dataset.add_padding(
                text_batch, digital=False
            )

            real_intent.extend(list(Evaluator.expand_list(sorted_intent)))

            digit_text = dataset.word_alphabet.get_index(padded_text)
            var_text = Variable(torch.LongTensor(digit_text))

            if torch.cuda.is_available():
                var_text = var_text.cuda()

            intent_idx = model(var_text, seq_lens, n_predicts=1)
            nested_intent = Evaluator.nested_list([list(Evaluator.expand_list(intent_idx))], seq_lens)[0]
            pred_intent.extend(dataset.intent_alphabet.get_instance(nested_intent))

        exp_pred_intent = Evaluator.max_freq_predict(pred_intent)
        return exp_pred_intent, real_intent, pred_intent


class Evaluator(object):

    """
    Max frequency prediction. 
    """

    @staticmethod
    def max_freq_predict(sample):
        predict = []
        for items in sample:
            predict.append(Counter(items).most_common(1)[0][0])
        return predict

    @staticmethod
    def exp_decay_predict(sample, decay_rate=0.8):
        predict = []
        for items in sample:
            item_dict = {}
            curr_weight = 1.0
            for item in items[::-1]:
                item_dict[item] = item_dict.get(item, 0) + curr_weight
                curr_weight *= decay_rate
            predict.append(sorted(item_dict.items(), key=lambda x_: x_[1])[-1][0])
        return predict

    @staticmethod
    def expand_list(nested_list):
        for item in nested_list:
            if isinstance(item, (list, tuple)):
                for sub_item in Evaluator.expand_list(item):
                    yield sub_item
            else:
                yield item

    @staticmethod
    def nested_list(items, seq_lens):
        num_items = len(items)
        trans_items = [[] for _ in range(0, num_items)]

        count = 0
        for jdx in range(0, len(seq_lens)):
            for idx in range(0, num_items):
                trans_items[idx].append(items[idx][count:count + seq_lens[jdx]])
            count += seq_lens[jdx]

        return trans_items