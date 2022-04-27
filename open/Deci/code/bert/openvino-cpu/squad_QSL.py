# coding=utf-8
# Copyright 2021 Arm Limited and affiliates.
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from transformers import BertTokenizer, BertTokenizerFast, RobertaTokenizerFast
from create_squad_data import read_squad_examples, convert_examples_to_features
from create_squad_data import InputFeatures
import mlperf_loadgen as lg

# To support feature cache.
import pickle

max_seq_length = 384
max_query_length = 64
doc_stride = 128


class SQuAD_v1_QSL():
    def __init__(self, total_count_override=None, perf_count_override=None, cache_path='eval_features.pickle', tokenizer_type='bert'):
        print("Constructing QSL...")
        eval_features = []
        # Load features if cached, convert from examples otherwise.
        if os.path.exists(cache_path):
            print("Loading cached features from '%s'..." % cache_path)
            with open(cache_path, 'rb') as cache_file:
                eval_features = pickle.load(cache_file)
        else:
            print("No cached features at '%s'... converting from examples..." % cache_path)

            print(f"Creating {tokenizer_type} tokenizer...")
            # tokenizer = BertTokenizer("build/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt")
            if tokenizer_type == 'bert':
                tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
            elif tokenizer_type == 'deci':
                tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
            else:
                raise NotImplemented
            print("Reading examples...")
            eval_examples = read_squad_examples(input_file="build/data/dev-v1.1.json",
                is_training=False, version_2_with_negative=False)

            print("Converting examples to features...")
            def append_feature(feature):
                eval_features.append(feature)

            def prepare_validation_features(examples, counter):
                # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
                # in one example possible giving several features when a context is long, each of those features having a
                # context that overlaps a bit the context of the previous feature.
                tokenized_examples = tokenizer(
                    [examples.question_text],
                    [' '.join(examples.doc_tokens)],
                    truncation="only_second",
                    max_length=384,
                    stride=128,
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                    padding="max_length"
                )

                # Since one example might give us several features if it has a long context, we need a map from a feature to
                # its corresponding example. This key gives us just that.
                sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

                # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
                # corresponding example_id and we will store the offset mappings.
                tokenized_examples["example_id"] = []

                for i in range(len(tokenized_examples["input_ids"])):
                    # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                    sequence_ids = tokenized_examples.sequence_ids(i)
                    context_index = 1

                    # One example can give several spans, this is the index of the example containing this span of text.
                    sample_index = sample_mapping[i]
                    tokenized_examples["example_id"].append(examples.qas_id)

                    # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
                    # position is part of the context or not.
                    tokenized_examples["offset_mapping"][i] = [
                        (o if sequence_ids[k] == context_index else None)
                        for k, o in enumerate(tokenized_examples["offset_mapping"][i])
                    ]
                # Roberta tokenizer doesn't use token_type, generating from data
                ind_sep = [ind for ind, tok in enumerate(tokenized_examples.data['input_ids'][0]) if tok == tokenizer.sep_token_id]
                ind_sep = ind_sep[1]
                token_type_ids = [int(ind > ind_sep) for ind in range(len(tokenized_examples.data['input_ids'][0])) ]
                unique_id = 1000000000

                # adapting to loadgen expected format
                tokenized_examples_input_feature = InputFeatures(
                    unique_id=unique_id + counter,
                    example_index=counter,
                    doc_span_index=0,
                    tokens=tokenizer.tokenize(tokenizer.decode(tokenized_examples.data['input_ids'][0])),
                    token_to_orig_map=tokenized_examples.data['offset_mapping'],
                    token_is_max_context=None,
                    input_ids=tokenized_examples.data['input_ids'][0],
                    input_mask=tokenized_examples.data['attention_mask'][0],
                    segment_ids=token_type_ids,
                    start_position=None,
                    end_position=None,
                    is_impossible=False)

                return tokenized_examples_input_feature

            # Parsing for deci's tokenizer
            if tokenizer_type == 'deci':
                eval_features = []
                for ind, example in enumerate(eval_examples):
                    eval_features.append(prepare_validation_features(example, ind))
            else:
                convert_examples_to_features(
                    examples=eval_examples,
                    tokenizer=tokenizer,
                    max_seq_length=max_seq_length,
                    doc_stride=doc_stride,
                    max_query_length=max_query_length,
                    is_training=False,
                    output_fn=append_feature,
                    verbose_logging=False)

            print("Caching features at '%s'..." % cache_path)
            with open(cache_path, 'wb') as cache_file:
                pickle.dump(eval_features, cache_file)

        self.eval_features = eval_features
        self.count = total_count_override or len(self.eval_features)
        self.perf_count = perf_count_override or self.count
        self.qsl = lg.ConstructQSL(self.count, self.perf_count, self.load_query_samples, self.unload_query_samples)
        print("Finished constructing QSL.")

    def load_query_samples(self, sample_list):
        pass

    def unload_query_samples(self, sample_list):
        pass

    def get_features(self, sample_id):
        return self.eval_features[sample_id]

    def __del__(self):
        print("Finished destroying QSL.")

def get_squad_QSL(total_count_override=None, perf_count_override=None, tokenizer_type='bert'):
    return SQuAD_v1_QSL(total_count_override, perf_count_override, tokenizer_type=tokenizer_type)
