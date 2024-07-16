import copy
import random
import torch

import torch.nn as nn
from torch.nn.functional import pad
from lm_eval import evaluator, tasks
import lm_eval
from watermarks.lm_eval_adaptor import LMEvalAdaptor


class Evaluator(object):
    def __init__(self):
        pass

    def evaluate_all(self, model, enc, args):
        self.tasks = args.tasks
        self.batch_size = args.batch_size
        self.num_fewshot = args.num_fewshot
        self.model_path = args.model_path
        
        task_names = self.tasks.split(",")
        lm_eval_model = LMEvalAdaptor(self.model_path, model, enc, self.batch_size)
        results = evaluator.simple_evaluate(model=lm_eval_model, tasks=task_names, batch_size=self.batch_size, no_cache=True, num_fewshot=self.num_fewshot,)
        print("perplexity & acc:")
        print(evaluator.make_table(results))