import argparse
import json
import os
import random
from collections import defaultdict
from typing import List, Dict

import torch
from tqdm import tqdm

#from sdb_modeling_gpt2 import GPT2Wrapper
from pt_sdb_modeling import GPT2Wrapper

DEBIASING_PREFIXES = [
    "The following text contains rude, disrespectful, or unreasonable language:\n",
    #"The following text contains very hateful, aggressive, disrespectful language:\n",
    #"The following text contains sexually explicit language:\n",
    #"The following text contains a threat:\n",
    #"The following text contains negative or hateful language targeting someone because of their identity:\n",
    #"The following text contains swear words, curse words, or other obscene or profane language:\n"
]

wrapper = GPT2Wrapper(model_name='gpt2')
prompt = "When I finished my career I lived not far"
output = wrapper.generate_self_debiasing(prompt, DEBIASING_PREFIXES)
print(output)
