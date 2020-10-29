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
    @staticmethod
    def validate(model, model_path, dataset, batch_size):
        """
        validation will write mistaken samples to files and make scores.
        """
        #model = torch.load(model_path)
        model.load_state_dict(torch.load(model_path))

        # Get the sentence list in test dataset.
        #sent_list = dataset.test_sentence
        confidence, exp_pred_intent = Processor.prediction(
            model, dataset, "utterance", batch_size
        )
        return confidence, exp_pred_intent

    @staticmethod
    def prediction(model, dataset, mode, batch_size):
        model.eval()

        if mode == "utterance":
            dataloader = dataset.batch_delivery('utterance', batch_size=batch_size, shuffle=False, is_digital=False)
        else:
            raise Exception("Argument error! mode belongs to {\"dev\", \"test\"}.")
        pred_intent, real_intent = '', []
        confidence = 0

        for text_batch, intent_batch in tqdm(dataloader, ncols=50):
            padded_text, [sorted_intent], seq_lens = dataset.add_padding(
                text_batch, [(intent_batch, False)], digital=False
            )

            real_intent.extend(list(Evaluator.expand_list(sorted_intent)))

            digit_text = dataset.word_alphabet.get_index(padded_text)
            var_text = Variable(torch.LongTensor(digit_text))
            num_of_words = var_text.shape[1]
            if torch.cuda.is_available():
                var_text = var_text.cuda()

            intent_prob = model(var_text, seq_lens)
            intent_prob = torch.exp(intent_prob)
            intent_prob = torch.sum(intent_prob,dim=0)/num_of_words
            confidence, intent_idx = intent_prob.max(0,keepdims=False)
            pred_intent = dataset.intent_alphabet.get_instance(intent_idx.unsqueeze(0))
        return float(confidence)*100, pred_intent[0]

class Evaluator(object):
    
    @staticmethod
    def expand_list(nested_list):
        for item in nested_list:
            if isinstance(item, (list, tuple)):
                for sub_item in Evaluator.expand_list(item):
                    yield sub_item
            else:
                yield item
