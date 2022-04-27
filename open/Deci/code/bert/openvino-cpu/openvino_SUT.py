# coding=utf-8
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

import array
import os
import sys
sys.path.insert(0, os.getcwd())

import mlperf_loadgen as lg
import numpy as np
from squad_QSL import get_squad_QSL
import infery  # Deci's python runtime inference engine (https://pypi.org/project/infery)
try:
    from infery_pro.parallel_infery import ParallelInfery
    USE_INFERY_PRO = True
except:
    USE_INFERY_PRO = False


def batched_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        if (i+n) <= len(lst): # last batch will be dealt with outside
            yield lst[i:i + n]

class BERT_Openvino_SUT():
    def __init__(self, args):
        """
        OpenVINO SUT. If infery-pro is available, several instances of the model will be created and run asynchronous
        """
        self.profile = args.profile
        self.model_path = args.model_path
        print(f"Loading openvino model {self.model_path}")
        self.infery_pro_return_cycle = args.infery_pro_return_cycle
        self.scenario = args.scenario
        self.responses_sent = 0
        self.batch_size = args.batch_size
        if USE_INFERY_PRO:
            self.num_inferencers = args.num_inferencers
            self.sess = []
            for _ in range(self.num_inferencers):
                self.sess.append(ParallelInfery.load(model_path=self.model_path,
                                    framework_type='openvino',
                                    inference_hardware='cpu',
                                    num_processes=1, num_threads=8))
        else:
            self.sess = infery.load(model_path=self.model_path,
                                 framework_type='openvino',
                                 inference_hardware='cpu')

        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries, self.process_latencies)
        print("Finished constructing SUT.")


        self.qsl = get_squad_QSL(args.max_examples,tokenizer_type=args.tokenizer)

    def pad_to_batch(self, x):
        x_pad = np.zeros((self.batch_size, x.shape[1]))
        x_pad[:x.shape[0], :x.shape[1]] = x
        return x_pad


    def process_batch(self, batched_features):
        '''

        :param batched_features:
        :return: parsing to DeciBERT input, padded if needed
        :rtype:
        '''
        pad_func = lambda x: self.pad_to_batch(x) if len(batched_features) != self.batch_size else x

        fd = {
            'input_ids': pad_func(np.stack(
                np.asarray([f.input_ids for f in batched_features]).astype(np.int64)[np.newaxis, :])[0, :,
                         :]),
            'attention_mask': pad_func(np.stack(
                np.asarray([f.input_mask for f in batched_features]).astype(np.int64)[np.newaxis, :])[0, :,
                              :]),
            'token_type_ids': pad_func(np.stack(
                np.asarray([f.segment_ids for f in batched_features]).astype(np.int64)[np.newaxis, :])[0,
                              :, ])
        }

        return fd

    def issue_queries(self, query_samples):
        if self.scenario == 'Offline':
            #  Extracting features to be split into batches
            eval_features = [self.qsl.get_features(query_samples[i].index) for i in range(len(query_samples))]
            if not USE_INFERY_PRO:
                for batch_ind, batched_features in enumerate(batched_list(eval_features, self.batch_size)):
                    fd = self.process_batch(batched_features)
                    scores = self.sess.predict(None, **fd)
                    scores_keys = list(scores.keys())
                    scores = [scores[scores_keys[0]], scores[scores_keys[1]]]

                    output = np.stack(scores, axis=-1)

                    # sending responses individually
                    for sample in range(self.batch_size):
                        response_array = array.array("B", output[sample].tobytes())
                        bi = response_array.buffer_info()
                        response = lg.QuerySampleResponse(query_samples[ self.responses_sent].id, bi[0], bi[1])
                        self.responses_sent += 1
                        lg.QuerySamplesComplete([response])

                # batch remainder in sync
                last_ind = (batch_ind + 1) * self.batch_size
                if last_ind < len(eval_features) - 1:
                    print("Final iteration")
                    batched_features = eval_features[last_ind:]
                    fd = self.process_batch(batched_features)
                    scores = self.sess.predict(None, **fd)
                    scores_keys = list(scores.keys())
                    scores = [scores[scores_keys[0]], scores[scores_keys[1]]]
                    output = np.stack(scores, axis=-1)[:len(batched_features)]
                    for sample in range(len(output)):
                        response_array = array.array("B", output[sample].tobytes())
                        bi = response_array.buffer_info()
                        response = lg.QuerySampleResponse(query_samples[self.responses_sent].id, bi[0],
                                                          bi[1])
                        self.responses_sent += 1
                        lg.QuerySamplesComplete([response])

            else:  # use infery pro, alternate between model instances and run async
                async_results = {i: [] for i in range(self.num_inferencers)}
                return_batch_cycle = int(self.infery_pro_return_cycle / self.batch_size)
                for batch_ind, batched_features in enumerate(batched_list(eval_features, self.batch_size)):
                    fd = self.process_batch(batched_features)

                    # Round Robin model assignment
                    async_results[batch_ind % self.num_inferencers].append((batch_ind, self.sess[batch_ind % self.num_inferencers].predict_async(None, **fd)))

                    if len(async_results[batch_ind % self.num_inferencers]) == return_batch_cycle or batch_ind == int(len(eval_features) / self.batch_size) - 1:
                        scores_list = [res[1].get() for res in async_results[batch_ind % self.num_inferencers]]
                        batch_list = [res[0] for res in async_results[batch_ind % self.num_inferencers]]

                        assert scores_list and all([r is not None for r in scores_list])
                        for ind_score, scores in enumerate(scores_list):
                            scores_keys = list(scores.keys())
                            scores = [scores[scores_keys[0]], scores[scores_keys[1]]]
                            output = np.stack(scores, axis=-1)

                            # sending responses individually
                            for sample in range(self.batch_size):
                                response_array = array.array("B", output[sample].tobytes())
                                bi = response_array.buffer_info()
                                # print(f' Sending response for: {batch_list[ind_score] * self.batch_size + sample}')
                                response = lg.QuerySampleResponse(query_samples[batch_list[ind_score] * self.batch_size + sample].id,
                                                          bi[0], bi[1])
                                self.responses_sent += 1
                                lg.QuerySamplesComplete([response])
                        async_results[batch_ind % self.num_inferencers] = []

                    if batch_ind % 50 == 0:
                        print(f'processed {batch_ind} batches')

                # check if all sessions were completed
                for results in async_results.values():
                    if len(results) > 0:
                        scores_list = [res[1].get() for res in results]
                        batch_list = [res[0] for res in results]

                        assert scores_list and all([r is not None for r in scores_list])
                        for ind_score, scores in enumerate(scores_list):
                            scores_keys = list(scores.keys())
                            scores = [scores[scores_keys[0]], scores[scores_keys[1]]]
                            output = np.stack(scores, axis=-1)

                            # sending responses individually
                            for sample in range(self.batch_size):
                                response_array = array.array("B", output[sample].tobytes())
                                bi = response_array.buffer_info()
                                response = lg.QuerySampleResponse(
                                    query_samples[batch_list[ind_score] * self.batch_size + sample].id,
                                    bi[0], bi[1])
                                self.responses_sent += 1
                                lg.QuerySamplesComplete([response])

                # batch remainder in sync
                last_ind = (batch_ind + 1) * self.batch_size
                if last_ind < len(eval_features) - 1:
                    print("Final iteration")
                    batched_features = eval_features[last_ind:]
                    fd = self.process_batch(batched_features)
                    scores = self.sess[0].predict(None, **fd)
                    scores_keys = list(scores.keys())
                    scores = [scores[scores_keys[0]], scores[scores_keys[1]]]
                    output = np.stack(scores, axis=-1)[:len(batched_features)]

                    # sending responses individually
                    for sample in range(len(output)):
                        response_array = array.array("B", output[sample].tobytes())
                        bi = response_array.buffer_info()
                        response = lg.QuerySampleResponse(query_samples[ self.responses_sent].id,
                                                          bi[0], bi[1])
                        self.responses_sent += 1
                        lg.QuerySamplesComplete([response])
                del self.sess, async_results

        else:  # Server
            for i in range(len(query_samples)):
                self.responses_sent += 1
                if self.responses_sent % 500 == 0:
                    print(f'Processed {self.responses_sent} samples')
                eval_features = self.qsl.get_features(query_samples[i].index)

                fd = {
                    'input_ids': np.array(eval_features.input_ids).astype(np.int64)[np.newaxis, :],
                    'attention_mask': np.array(eval_features.input_mask).astype(np.int64)[np.newaxis, :],
                    'token_type_ids': np.array(eval_features.segment_ids).astype(np.int64)[np.newaxis, :]
                }
                # stack all input
                scores = self.sess.predict(None, **fd)
                # Parsing openvino format
                scores_keys = list(scores.keys())
                scores = [scores[scores_keys[0]], scores[scores_keys[1]]]
                output = np.stack(scores, axis=-1)[0]

                response_array = array.array("B", output.tobytes())
                bi = response_array.buffer_info()
                response = lg.QuerySampleResponse(query_samples[i].id, bi[0], bi[1])
                lg.QuerySamplesComplete([response])

    def flush_queries(self):
        pass

    def process_latencies(self, latencies_ns):
        pass

    def __del__(self):
        if self.profile:
            print("openvino runtime profile dumped to: '{}'".format(self.sess.end_profiling()))
        print("Finished destroying SUT.")


def get_openvino_sut(args):
    return BERT_Openvino_SUT(args)
