# Copyright 2023 The Distilling-step-by-step authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import re
import json
import numpy as np
import torch
import torch.nn.functional as F

from datasets import Dataset, DatasetDict, load_dataset
from torch.nn.utils.rnn import pad_sequence


DATASET_ROOT = 'datasets'


class DatasetLoader(object):
    def __init__(self, dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None):
        self.data_root = DATASET_ROOT
        self.dataset_name = dataset_name
        self.source_dataset_name = source_dataset_name
        self.dataset_version = dataset_version
        self.has_valid = has_valid
        self.split_map = split_map

        self.batch_size = batch_size
        self.train_batch_idxs = train_batch_idxs
        self.test_batch_idxs = test_batch_idxs
        self.valid_batch_idxs = valid_batch_idxs
        
        assert self.split_map is not None    


    def load_from_source(self):
        if self.source_dataset_name is None:
            self.source_dataset_name = self.dataset_name
        if self.dataset_version is None:
            datasets = load_dataset(self.source_dataset_name)
        else:
            datasets = load_dataset(self.source_dataset_name, self.dataset_version)
        return datasets


    def to_json(self, datasets):
        for k, v in self.split_map.items():
            datasets[v].to_json(f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_{k}.json')


    def load_from_json(self):
        data_files = {
            'train': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_train.json',
            'test': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_test.json',
        }

        if self.has_valid:
            data_files.update({'valid': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_valid.json',})
        print("data_files")
        print(data_files)
        datasets = load_dataset('json', data_files=data_files)
        datasets = self._post_process(datasets) 

        # subsample training dataset if needed
        num_train = len(datasets['train'])
        idxs = list()
        for idx in self.train_batch_idxs:
            idxs += range(idx*self.batch_size, (idx+1)*self.batch_size)        
        datasets['train'] = Dataset.from_dict(datasets['train'][[idx for idx in idxs if idx < num_train]])

        return datasets


    def load_llm_preds(self, split):
        labels = list()
        rationales = list()
        for idx in getattr(self, f'{split}_batch_idxs'):
            with open(f'{self.data_root}/{self.dataset_name}/llm/{split}_CoT_{idx}.json') as f:
                outputs = json.load(f)

            for output in outputs:
                rationale, label = self._parse_llm_output(output)

                rationales.append(rationale)
                labels.append(label)

        return rationales, labels


    def load_gpt_preds(self, split):
        labels = list()
        rationales = list()
        
        with open(f'{self.data_root}/gpt-neox/{self.dataset_name}/{split}.json') as f:
            outputs = json.load(f)

        for output in outputs:
            rationale, label = self._parse_gpt_output(output)

            rationales.append(rationale)
            labels.append(label)

        return rationales, labels


    def _post_process(self, datasets):
        raise NotImplementedError


    def _parse_llm_output(self, output):
        raise NotImplementedError


    def _parse_gpt_output(self, output):
        raise NotImplementedError


class CQADatasetLoader(DatasetLoader):
    def __init__(self):
        dataset_name = 'cqa'
        source_dataset_name = 'cos_e'
        dataset_version = 'v1.11'
        has_valid = False
        split_map = {
            'train': 'train',
            'test': 'validation',
        }
        batch_size = 1000
        train_batch_idxs = range(10)
        test_batch_idxs = range(2)

        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None)


    def _post_process(self, datasets):
        
        def prepare_input(example):
            question = example['question']
            c_0 = example['choices'][0]
            c_1 = example['choices'][1]
            c_2 = example['choices'][2]
            c_3 = example['choices'][3]
            c_4 = example['choices'][4]

            input = f'{question}\nAnswer Choices:\n(a) {c_0}\n(b) {c_1}\n(c) {c_2}\n(d) {c_3}\n(e) {c_4}'

            example['input'] = input
            example['label'] = example['answer']

            return example

        datasets = datasets.map(prepare_input)
        datasets = datasets.remove_columns(['id', 'question', 'choices', 'answer', 'abstractive_explanation', 'extractive_explanation'])

        return datasets


    def _parse_llm_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip()
        return rationale_label, None 
        '''
        rationale, label = rationale_label.split('So the answer is')
        rationale = rationale.rstrip()

        try:
            label = re.split(r'\(.\)', label)[1].strip()
            label = label if label[-1]!='.' else label[:-1]
        except:
            label = ' '
        
        return rationale, label
        '''

    def _parse_gpt_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip().lstrip()
        try:
            rationale, label = rationale_label.split('So the answer is')
            rationale = rationale.rstrip()
        except:
            rationale = ' '
            label = ' '
            return rationale, label

        try:
            label = re.split(r'\(.\)', label)[1].strip()
            label = label if label[-1]!='.' else label[:-1]
        except:
            label = ' '
        
        return rationale, label


class SVAMPDatasetLoader(DatasetLoader):
    def __init__(self):
        dataset_name = 'svamp'
        source_dataset_name = 'svamp'
        dataset_version = None
        has_valid = False
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        batch_size = 500
        train_batch_idxs = range(2)
        test_batch_idxs = range(1)


        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None)


    def load_from_source(self):
        with open(f'{self.data_root}/{self.dataset_name}/SVAMP.json') as f:
            original_dataset = json.load(f)

        dataset = list()
        for data in original_dataset:
            input = f'{data["Body"]}\n{data["Question"]}'
            equation = data["Equation"]

            dataset.append({
                'input': input,
                'label': equation,
            })

        idxs = np.random.RandomState(seed=0).permutation(len(dataset))
        train_idxs = idxs[:800]
        test_idxs = idxs[800:]

        train_dataset = Dataset.from_list(np.array(dataset)[train_idxs].tolist())
        test_dataset = Dataset.from_list(np.array(dataset)[test_idxs].tolist())

        datasets = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })

        return datasets
        

    def _post_process(self, datasets):
        return datasets


    def _parse_llm_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip()
        try:
            rationale, label = rationale_label.split('The answer is')
        except:
            rationale = ' '
            label = ' '
            return rationale, label
            
        rationale = rationale.rstrip()
        try:
            label = re.search(r'\(.*\)', label).group(0)
        except:
            label = ' '

        return rationale, label

    def _parse_gpt_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip().lstrip()
        try:
            rationale, label = rationale_label.split('The answer is')
        except:
            rationale = ' '
            label = ' '
            return rationale, label
            
        rationale = rationale.rstrip()
        try:
            label = re.search(r'\(.*\)', label).group(0)
        except:
            label = ' '

        return rationale, label


class ASDivDatasetLoader(DatasetLoader):
    def __init__(self):
        dataset_name = 'asdiv'
        dataset_version = None
        has_valid = False
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        batch_size = 1000
        train_batch_idxs = range(3)
        test_batch_idxs = range(1)

        super().__init__(dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None)


    def load_from_source(self):
        raise NotImplementedError
        

    def _post_process(self, datasets):

        def prepare_input(example):
            example['input'] = example['Body'] + '\n' + example['Question']
            answer = example['Answer'].split(' ')[0]
            example['label'] = answer

            return example

        datasets = datasets.map(prepare_input)
        datasets = datasets.remove_columns(['Body', 'Question', 'Formula', 'Answer'])

        return datasets


    def _parse_llm_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip()
        try:
            rationale, label = rationale_label.split('The answer is')
        except:
            rationale = ' '
            label = ' '
            return rationale, label
            
        rationale = rationale.rstrip()
        try:
            label = re.search(r'\(.*\)', label).group(0)
        except:
            label = ' '

        return rationale, label


    def _parse_gpt_output(self, output):
        raise NotImplementedError


class ESNLIDatasetLoader(DatasetLoader):
    def __init__(self, subset='full'):
        dataset_name = 'esnli'
        source_dataset_name = 'esnli'
        dataset_version = None
        has_valid = True
        split_map = {
            'train': 'train',
            'valid': 'validation',
            'test': 'test',
        }
        batch_size = 5500
        if subset == 'full':
            train_batch_idxs = range(100)
        elif subset == 'small':
            train_batch_idxs = range(10)
        else:
            raise ValueError
        test_batch_idxs = range(2)
        valid_batch_idxs = range(2)

        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=valid_batch_idxs)


    def _post_process(self, datasets):
        
        def prepare_input(example):
            if example['label'] == 0:
                example['label'] = 'entailment'
            elif example['label'] == 1:
                example['label'] = 'neutral'
            elif example['label'] == 2:
                example['label'] = 'contradiction'

            return example

        datasets = datasets.map(prepare_input)
        datasets = datasets.remove_columns(['explanation_1', 'explanation_2', 'explanation_3'])

        return datasets


    def _parse_llm_output(self, output):
        rationale = output.split("Answer:")[0].rstrip()
        try:
            label = output.split("Answer: ")[1].split("Premise")[0].rstrip()
        except:
            label = ' '

        return rationale, label

    
    def _parse_gpt_output(self, output):
        rationale = output.split("Answer:")[0].rstrip().lstrip()
        try:
            label = output.split("Answer: ")[1].split("Premise")[0].rstrip()
        except:
            label = ' '

        return rationale, label


class ANLIDatasetLoader(DatasetLoader):
    def __init__(self, dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs):

        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=valid_batch_idxs)

    def _post_process(self, datasets):
        
        def label_idx2text(example):            
            if example['label'] == 0:
                example['label'] = 'entailment'
            elif example['label'] == 1:
                example['label'] = 'neutral'
            elif example['label'] == 2:
                example['label'] = 'contradiction'
            return example

        datasets = datasets.map(label_idx2text)
        datasets = datasets.remove_columns(['uid', 'reason'])

        return datasets


    def _parse_llm_output(self, output):
        try:
            rationale, label = output.split("Premise:")[0].rstrip().split("So the answer is")
        except:
            rationale = ''
            label = ''
        
        rationale = rationale.rstrip()
        label = label.lstrip()[:-1]

        return rationale, label


    def _parse_gpt_output(self, output):
        try:
            rationale, label = output.split("Premise:")[0].rstrip().lstrip().split("So the answer is")
        except:
            try:
                rationale, label = output.split("Premise:")[0].rstrip().lstrip().split("The answer is")
            except:
                rationale = ''
                label = ''

        
        rationale = rationale.rstrip()
        label = label.lstrip()[:-1]

        return rationale, label


class ANLI1DatasetLoader(ANLIDatasetLoader):
    def __init__(self):
        dataset_name = 'anli1'
        source_dataset_name = 'anli'
        dataset_version = None
        has_valid = True
        split_map = {
            'train': 'train_r1',
            'valid': 'dev_r1',
            'test': 'test_r1',
        }
        batch_size = 5000
        train_batch_idxs = range(4)
        test_batch_idxs = range(1)
        valid_batch_idxs = range(1)

        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=valid_batch_idxs)



def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx


class DataCollatorGAIL:

    def __init__(self, max_token_len, inference_tp_size):
        self.max_token_len = max_token_len
        self.inference_tp_size = inference_tp_size

    def __call__(self, data):
        #print("collator ... ")
        #print(data)
        batch = {}
        pad_token_id = data[0][-1]
        #print("data")
        #print(data)
        token_id = pad_sequence([f[0] for f in data],
                            padding_value=pad_token_id,
                            batch_first=True)
        token_mask = pad_sequence([f[1] for f in data],
                            padding_value=0,
                            batch_first=True)
        length = token_id.size()[-1]
        pad_length = self.max_token_len - length
        if pad_length > 0 :
            batch["tokens"] = F.pad(token_id,
                                    pad=(0, pad_length),
                                    mode='constant',
                                    value=pad_token_id)
            batch["token_att_mask"] = F.pad(token_mask,
                                            pad=(0, pad_length),
                                            mode="constant",
                                            value=0)
        else:
            batch["tokens"] = token_id
            batch["token_att_mask"] = token_mask
            
        #batch["tokens"] = batch["tokens"].flip(1)
        #batch["token_att_mask"] = batch["token_att_mask"].flip(1)
        return batch



class MiniDataset:

    def __init__(self, max_size, small_batch_size):
        self.dataset = []
        self.max_size = max_size
        self.small_batch_size = small_batch_size

    def seperate(self):
        small_dataset = []
        for large_batch in self.dataset:
            if type(large_batch) == list or type(large_batch) == tuple:
                large_size = len(large_batch[0])
            elif type(large_batch) == dict:
                large_size = len(large_batch[list(large_batch.keys())[0]])
            else:
                large_size = len(large_batch)
            for i in range(0, large_size, self.small_batch_size):
                if type(large_batch) == list or type(large_batch) == tuple:
                    small_dataset.append(
                        [x[i:i + self.small_batch_size] for x in large_batch])
                elif type(large_batch) == dict:
                    small_dataset.append({
                        k: v[i:i + self.small_batch_size]
                        for k, v in large_batch.items()
                    })
                else:
                    small_dataset.append(large_batch[i:i +
                                                     self.small_batch_size])
        self.free()

        return small_dataset

    def add(self, data):
        if len(self.dataset) < self.max_size:
            self.dataset.append(data)
            if len(self.dataset) == self.max_size:
                return self.seperate()
            else:
                return None
        else:
            raise ValueError(
                "The dataset is full but we did not stop it. There is a bug in the code."
            )

    def free(self):
        self.dataset = []

class PromptDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, pad_token_id) -> None:
        #super().__init__()
        self.dataset = dataset
        self.pad_token_id = pad_token_id

    def __len__(self):
        length = len(self.dataset)
        return length

    def __getitem__(self, idx):
        if isinstance(idx, list):
            '''
            print('idx')
            print(idx)
            idss, masks, pads = [] , [], []
            for _idx in idx:
                idss.append(self.dataset[_idx]['input_ids'])
                masks.append(self.dataset[_idx]['attention_mask'])
                pads.append(self.pad_token_id)
            #return self.dataset[idx[:]]["input_ids"], self.dataset[idx[:]]["attention_mask"], self.pad_token_id
            return idss, masks, pads
            '''
            return self.dataset[idx[0]]["input_ids"], self.dataset[idx[0]]["attention_mask"], self.pad_token_id
        else:
            return self.dataset[idx]["input_ids"], self.dataset[idx]["attention_mask"], self.pad_token_id

