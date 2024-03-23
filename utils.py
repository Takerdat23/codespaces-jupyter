from __future__ import print_function
import numpy as np
import random
import json
import os
import re
import sys
import torch
from tqdm import tqdm
import operator
import torch.autograd as autograd
from nltk.corpus import stopwords
from transformers import BertTokenizer, AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd 
import time
from torch.utils.data import Dataset



def read_json(filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data


def write_json(filename,data):
    with open(filename, 'w', encoding='utf8') as fp:
        json.dump(data, fp)


def make_save_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

#for aspect-based tasks

def get_categories(df): 
    aspect_categories = set()
    for index , row in df.iterrows():
    # Process aspect sentiments
        aspect_sentiments = row['label'].split(';')
        for aspect_sentiment in aspect_sentiments:
            split = aspect_sentiment.strip('{}')
            if(len(split.split('#'))< 2): 
                continue
            aspect, _ = split.split('#')
            aspect_categories.add(aspect)

    return list(aspect_categories)



def process_data(df ,aspect_categories ): 

    dataset = []
   
    for _ , row in df.iterrows():
        data_dict = {}
 
        data_dict["comment"]= row['comment']
        label_vectors = {}


        aspect_sentiments = {}
        if row['label'] == None: 
            continue
        else: 
            for aspect_sentiment in row['label'].split(';'):
                split = aspect_sentiment.strip('{}')
                if(len(split.split('#'))< 2): 
                    continue
                aspect, sentiment = split.split('#')
                aspect_sentiments[aspect] = sentiment

   
            for aspect in aspect_categories:
                label_vector = [0, 0, 0]  
                sentiment = aspect_sentiments.get(aspect, None)
                if sentiment:
                    if sentiment == 'Positive':
                        label_vector[0] = 1
                    elif sentiment == 'Negative':
                        label_vector[1] = 1
                    elif sentiment == 'Neutral':
                        label_vector[2] = 1
                label_vectors[aspect] = label_vector

        
     
            data_dict["label"]= torch.tensor([label_vectors[aspect] for aspect in aspect_categories])
        dataset.append(data_dict)

    return dataset


def cc(arr, no_cuda=False):
    if no_cuda:
        return torch.from_numpy(np.array(arr))
    else:
        return torch.from_numpy(np.array(arr)).cuda()


def one_hot(indices, depth, no_cuda=False):
    shape = list(indices.size())+[depth]
    indices_dim = len(indices.size())
    if no_cuda:
        a = torch.zeros(shape, dtype=torch.float)
    else:
        a = torch.zeros(shape,dtype=torch.float).cuda()
    return a.scatter_(indices_dim,indices.unsqueeze(indices_dim),1)


def get_test(test_file):
    txts = []
    max_len = 0
    for line in open(test_file):
        words = []

        for w in line.strip().split():
            w = w.lower()
            w = re.sub('[0-9]+', 'N', w)
            words.append(w)
        if len(words) > max_len:
            max_len = len(words)

        txts.append(' '.join(words))

    print('test number:',len(txts))
    print('test max_len:',max_len)
    return txts


class SentimentDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        inputs = [example["comment"] for example in batch]
        labels = [example["label"] for example in batch]

        inputs_dict = self.tokenizer(inputs, max_length=128,  padding='max_length', truncation=True, return_tensors="pt")
     
        labels_tensor = torch.stack(labels)

      
        return {"input_ids": inputs_dict["input_ids"].squeeze(0),
                "attention_mask": inputs_dict["attention_mask"].squeeze(0),
                "labels": labels_tensor}


class ClassificationDataset(Dataset):
    def __init__(self, train_path, tokenizer,  max_seq_length = 60, label_maps= None):
        self.train_data = pd.read_csv(train_path)
        self.tokenizer = tokenizer
        self.label_maps = label_maps
        self.max_length = max_seq_length
       
        
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        

    
        
        comment = self.train_data.iloc[idx, 1]
        label = self.train_data.iloc[idx, 4]
        
        labels = label.split(';')[0]
        print(labels)
        label = labels.split('#')[1][:-1]

        encoded = self.tokenizer(comment, max_length= self.max_length , truncation=True )
        if(self.label_maps != None): 

            label = self.label_maps[label]
     
        return encoded['input_ids'], encoded['attention_mask'], label


class data_utils():
    def __init__(self, args):
        self.seq_length = args.seq_length
        self.batch_size = args.batch_size
        self.no_cuda = args.no_cuda

        self.dict_path = os.path.join(args.model_dir,'dictionary.json')
        self.train_path = args.train_path
        self.tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')    

        self.eos_id = 0
        self.unk_id = 1
        self.mask_id = 2
        self.cls_id = 3

        if args.train or not os.path.exists(self.dict_path):
            self.process_training_data()
            pass
        elif args.test:
            self.new_vocab = read_json(self.dict_path)
            pass

        print('vocab_size:',len(self.new_vocab))
        
        self.vocab_size = len(self.new_vocab)
        self.index2word = self.vocab_size*[[]]
        for w in self.new_vocab:
            self.index2word[self.new_vocab[w]] = w


    def process_training_data(self):
        self.training_data = []

        self.new_vocab = dict()
        self.new_vocab['[PAD]'] = 0
        self.new_vocab['[UNK]'] = 1
        self.new_vocab['[MASK]'] = 2
        self.new_vocab['[CLS]'] = 3
        
        dd = []
        word_count = {}
        for line in tqdm(open(self.train_path,  encoding="utf-8")):
            w_list = []
            for word in line.strip().split():
                if 'N' in word:
                    w = 'N'
                else:
                    sub_words = self.tokenizer.tokenize(word)
                    w = sub_words[0]
                word_count[w] = word_count.get(w,0) + 1
                w_list.append(w)
            w_list = ['[CLS]'] + w_list
            dd.append(w_list)

        for w in word_count:
            if word_count[w] > 1:
                self.new_vocab[w] = len(self.new_vocab)

        for d in dd:
            word_list = []
            for w in d:
                if w in self.new_vocab:
                    word_list.append(self.new_vocab[w])
                else:
                    word_list.append(self.unk_id)
            self.training_data.append(word_list)

        write_json(self.dict_path, self.new_vocab)


    def make_masked_data(self, indexed_tokens, mask_percentage=0.15, seq_length=50):
        masked_vec = np.zeros([seq_length], dtype=np.int32) + self.eos_id
        origin_vec = np.zeros([seq_length], dtype=np.int32) + self.eos_id
        target_vec = np.zeros([seq_length], dtype=np.int32) - 1

        unknown = 0.
        masked_num = 0.

        length = len(indexed_tokens)
        for i, word in enumerate(indexed_tokens):
            if i >= seq_length:
                break

            origin_vec[i] = word
            masked_vec[i] = word

        # Mask words based on the specified percentage
            if random.random() < mask_percentage:
                target_vec[i] = word
                masked_num += 1

                rand_num = random.randint(0, 9)
                if rand_num == 0:
                # Keep the word unchanged
                    pass
                elif rand_num == 1:
                # Sample a word
                    masked_vec[i] = random.randint(4, self.vocab_size - 1)
                else:
                    masked_vec[i] = self.mask_id

        if length > 70 or masked_num == 0:
            masked_vec = None

        return masked_vec, origin_vec, target_vec


    def text2id(self, text, seq_length=60):
        vec = np.zeros([seq_length] ,dtype=np.int32)
        unknown = 0.

        w_list = []
        for word in text.strip().split():
            if 'N' in word:
                w = 'N'
            else:
                sub_words = self.tokenizer.tokenize(word)
                w = sub_words[0]
            if w in self.new_vocab:
                w_list.append(self.new_vocab[w])
            else:
                w_list.append(self.unk_id)
        w_list = [self.new_vocab['[CLS]']] + w_list
        indexed_tokens = w_list
        assert len(text.strip().split())+1 == len(indexed_tokens)

        for i,word in enumerate(indexed_tokens):
            if i >= seq_length:
                break
            vec[i] = word

        return vec
    

    def train_data_yielder_classification(self, labels_map):
        """
        Generate training data for classification tasks.

        Args:
        - labels_map (dict): A dictionary mapping class labels to integer IDs.

        Yields:
        - batch (dict): A dictionary containing batches of input data and their corresponding class labels.
        """
        batch = {'input': [], 'input_mask': [], 'target_vec': [], 'y': []}
        max_len = 0
        for epo in range(1000000000):
            start_time = time.time()
            print("\nstart epo %d!!!!!!!!!!!!!!!!\n" % (epo))
            for line, label in self.training_data:  # Assuming self.training_data contains (text, label) pairs
                input_vec = self.text2id(line, 60)  # Convert text to IDs
                label_id = labels_map[label]  # Get the integer ID for the class label

                length = np.sum(input_vec != self.eos_id)
                if length > max_len:
                    max_len = length
                batch['input'].append(input_vec)
                batch['input_mask'].append(np.expand_dims(input_vec != self.eos_id, -2).astype(np.int32))
                batch['target_vec'].append(label_id)  # Append class label ID
                batch['y'].append(input_vec)  # Append original input (not used in classification)

                if len(batch['input']) == self.batch_size:
                    batch = {k: cc(v, self.no_cuda) for k, v in batch.items()}
                    yield batch
                    max_len = 0
                    batch = {'input': [], 'input_mask': [], 'target_vec': [], 'y': []}
            end_time = time.time()
            print('\nfinish epo %d, time %f!!!!!!!!!!!!!!!\n' % (epo, end_time - start_time))


    def train_data_yielder(self, mask_percentage=0.15):
        batch = {'input': [], 'input_mask': [], 'target_vec': [], 'y': []}
        max_len = 0
        for epo in range(1000000000):
            start_time = time.time()
            print("\nstart epo %d!!!!!!!!!!!!!!!!\n" % (epo))
            for line in self.training_data:
                input_vec, origin_vec, target_vec = self.make_masked_data(line, mask_percentage)  # Pass mask_percentage to make_masked_data

                if input_vec is not None:
                    length = np.sum(input_vec != self.eos_id)
                    if length > max_len:
                        max_len = length
                    batch['input'].append(input_vec)
                    batch['input_mask'].append(np.expand_dims(input_vec != self.eos_id, -2).astype(np.int32))
                    batch['target_vec'].append(target_vec)
                    batch['y'].append(origin_vec)

                    if len(batch['input']) == self.batch_size:
                        batch = {k: cc(v, self.no_cuda) for k, v in batch.items()}
                        yield batch
                        max_len = 0
                        batch = {'input': [], 'input_mask': [], 'target_vec': [], 'y': []}
            end_time = time.time()
            print('\nfinish epo %d, time %f!!!!!!!!!!!!!!!\n' % (epo, end_time - start_time))




    def id2sent(self,indices, test=False):
        sent = []
        word_dict={}
        for w in indices:
            if w != self.eos_id:
                sent.append(self.index2word[w]) 

        return ' '.join(sent)


    def subsequent_mask(self, vec):
        attn_shape = (vec.shape[-1], vec.shape[-1])
        return (np.triu(np.ones((attn_shape)), k=1).astype('uint8') == 0).astype(np.float)