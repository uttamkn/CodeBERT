# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import gzip
import os
import json
import numpy as np
from more_itertools import chunked
import argparse

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


DATA_DIR='../data/codesearch'

def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string


def preprocess_test_data(language, test_batch_size=1000, use_hf=False):
    DOCSTRING_TOKENS = ""
    CODE_TOKENS = ""
    URL = ""
    if use_hf:
        if load_dataset is None:
            raise ImportError("Please install the 'datasets' library: pip install datasets")

        print("Downloading dataset for language: {} from Hugging Face".format(language))
        dataset = load_dataset("code_search_net", language, split="test")
        data = dataset.to_pandas().to_dict(orient="records")
        print("Downloaded {} examples".format(len(data)))
        DOCSTRING_TOKENS = "func_documentation_tokens"
        CODE_TOKENS = "func_code_tokens"
        URL = "func_code_url"
    else:
        path = os.path.join(DATA_DIR, '{}_test_0.jsonl.gz'.format(language))
        print("Reading local dataset: {}".format(path))
        with gzip.open(path, 'r') as pf:
            lines = pf.readlines()
        data = [json.loads(str(line, encoding='utf-8')) for line in lines]
        DOCSTRING_TOKENS = "docstring_tokens"
        CODE_TOKENS = "code_tokens"
        URL = "url"

    idxs = np.arange(len(data))
    np.random.seed(0)
    np.random.shuffle(idxs)
    data = np.array(data, dtype=np.object)[idxs]
    batched_data = chunked(data, test_batch_size)

    print("Start processing")
    for batch_idx, batch_data in enumerate(batched_data):
        if len(batch_data) < test_batch_size:
            break  # the last batch is smaller than the others, exclude

        examples = []
        for d_idx in range(len(batch_data)):
            line_a = batch_data[d_idx]
            doc_token = ' '.join(line_a[DOCSTRING_TOKENS])
            for line_b in batch_data:
                code_token = ' '.join([format_str(token) for token in line_b[CODE_TOKENS]])
                example = (str(1), line_a[URL], line_b[URL], doc_token, code_token)
                examples.append('<CODESPLIT>'.join(example))

        data_path = os.path.join(DATA_DIR, 'test', language)
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        file_path = os.path.join(data_path, 'batch_{}.txt'.format(batch_idx))
        print("Writing to: {}".format(file_path))
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(examples))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-hf', action='store_true', help="Use Hugging Face datasets instead of local gz files.")
    parser.add_argument('--languages', nargs='+', default=['go', 'php', 'python', 'java', 'javascript', 'ruby'],
                        help="List of programming languages to process (space-separated). Default: all.")
    args = parser.parse_args()

    for lang in args.languages:
        preprocess_test_data(lang, use_hf=args.use_hf)
