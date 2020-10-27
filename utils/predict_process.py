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
    @staticmethod
    def validate(model_path, dataset, batch_size):
        """
        validation will write mistaken samples to files and make scores.
        """

        model = torch.load(model_path)

        # Get the sentence list in test dataset.
        #sent_list = dataset.test_sentence

        confidence, pred_intent = Processor.prediction(
            model, dataset, "utterance", batch_size
        )
        return confidence, pred_intent

    @staticmethod
    def prediction(model, dataset, mode, batch_size):
        model.eval()

        if mode == "utterance":
            dataloader = dataset.batch_delivery('utterance', batch_size=batch_size, shuffle=False, is_digital=False)
        else:
            raise Exception("Argument error! mode belongs to {\"dev\", \"test\"}.")
            
        p, pred_intent, real_intent = 0,[], []
        num_of_words = 0

        for text_batch, intent_batch in tqdm(dataloader, ncols=50):
            padded_text, [sorted_intent], seq_lens = dataset.add_padding(
                text_batch, [(intent_batch, False)], digital=False
            )

            real_intent.extend(list(Evaluator.expand_list(sorted_intent)))

            digit_text = dataset.word_alphabet.get_index(padded_text)
            var_text = Variable(torch.LongTensor(digit_text))

            if torch.cuda.is_available():
                var_text = var_text.cuda()

            intent_idx = model(var_text, seq_lens, n_predicts=1)
            nested_intent = Evaluator.nested_list([list(Evaluator.expand_list(intent_idx))], seq_lens)[0]
            pred_intent.extend(dataset.intent_alphabet.get_instance(nested_intent))

            intent_i = model(var_text, seq_lens)
            intent_i = torch.exp(intent_i)
            p+=intent_i
            num_of_words +=1
        exp_pred_intent = Evaluator.max_freq_predict(pred_intent)
        p = (p/num_of_words).squeeze(0)
        confidence, pred_intent = p.max(0,keepdims=False)
        #pred_intent = dataset.intent_alphabet.get_instance(int(pred_intent))
        return float(confidence*100), exp_pred_intent[0]
        #return real_intent, pred_intent


class Evaluator(object):
    def max_freq_predict(sample):
        predict = []
        for items in sample:
            predict.append(Counter(items).most_common(1)[0][0])
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
