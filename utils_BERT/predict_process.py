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
            
        pred_intent, real_intent = 0, []
        num_of_words = 0

        for text_batch, intent_batch in tqdm(dataloader, ncols=50):
            bert_tokenizer = BERT()
            padded_text, [sorted_intent], seq_lens = dataset.add_padding(
                text_batch, [(intent_batch, False)], digital=False
            )

            real_intent.extend(list(Evaluator.expand_list(sorted_intent)))
            var_text, att_var = bert_tokenizer.tokenize(padded_text)
            var_text = Variable(torch.LongTensor(var_text))
            att_var = Variable(torch.LongTensor(att_var))

            if torch.cuda.is_available():
                var_text = var_text.cuda()
                att_var = att_var.cuda()

            intent_idx = model(var_text, att_var, seq_lens)
            intent_idx = torch.exp(intent_idx)
            pred_intent+=intent_idx
            num_of_words +=1
        pred_intent = (pred_intent/num_of_words).squeeze(0)
        confidence, pred_intent = pred_intent.max(0,keepdims=False)
        pred_intent = dataset.intent_alphabet.get_instance(int(pred_intent))
        return float(confidence*100), pred_intent
        #return real_intent, pred_intent


class Evaluator(object):

    @staticmethod
    def expand_list(nested_list):
        for item in nested_list:
            if isinstance(item, (list, tuple)):
                for sub_item in Evaluator.expand_list(item):
                    yield sub_item
            else:
                yield item
