import pandas as pd
from datasets import load_dataset
from transformers import BertTokenizer, AutoTokenizer
import torch
import numpy as np 
from utils import * 
from torch.utils.data import Dataset, DataLoader
from models import *
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

TRAIN_PATH = './data/UIT-ViSFD/Train.csv'
VAL_PATH = './data/UIT-ViSFD/Dev.csv'
TEST_PATH = './data/UIT-ViSFD/Test.csv'


df_train = pd.read_csv(TRAIN_PATH,  encoding = 'utf8')
df_val = pd.read_csv(VAL_PATH, encoding = 'utf8')
df_test = pd.read_csv(TEST_PATH, encoding = 'utf8')


tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')




categories = get_categories(df_train)
dataset = process_data(df_train, categories)
data_collator = SentimentDataCollator(tokenizer)
dataloader = DataLoader(dataset, batch_size=64, collate_fn=data_collator)

categories_test = get_categories(df_test)
data_test = process_data(df_test, categories_test)
print("test", categories_test )

      
# model = ABSA_Tree_transfomer( vocab_size= tokenizer.vocab_size, N= 12, d_model= 768, d_ff= 2048, h= 12, dropout = 0.1, num_categories = len(categories) , no_cuda=False)



# device = 'cuda'

# model.to(device)

# for batch in tqdm(dataloader):
#     inputs = batch['input_ids'].to(device)
#     mask = batch['attention_mask'].to(device)
#     labels = batch['labels'].to(device)

#     output = model(inputs, mask, categories)
#     print(output.shape)
#     break


def calculate_metrics(predictions, ground_truth):
    """
    Calculate accuracy, precision, recall, and F1 score given predictions and ground truth.
    """
    accuracy = accuracy_score(predictions, ground_truth)
    precision, recall, f1_score, _ = precision_recall_fscore_support(predictions, ground_truth, average='weighted')
    return accuracy, precision, recall, f1_score

def evaluate_aspect_sentiment_metrics(dataset, aspect_categories):
    """
    Evaluate accuracy, precision, recall, and F1 score given a dataset and aspect categories.
    """
    predictions = {aspect: [] for aspect in aspect_categories}
    ground_truth = {aspect: [] for aspect in aspect_categories}
    
    for data_point in dataset:
        for aspect in aspect_categories:
            predicted_sentiment = np.argmax(data_point["predicted_labels"][aspect])
            true_sentiment = np.argmax(data_point["true_labels"][aspect])
            predictions[aspect].append(predicted_sentiment)
            ground_truth[aspect].append(true_sentiment)
    
    aspect_metrics = {}
    for aspect in aspect_categories:
        aspect_predictions = predictions[aspect]
        aspect_ground_truth = ground_truth[aspect]
        aspect_accuracy, aspect_precision, aspect_recall, aspect_f1_score = calculate_metrics(aspect_predictions, aspect_ground_truth)
        aspect_metrics[aspect] = {
            "Accuracy": aspect_accuracy,
            "Precision": aspect_precision,
            "Recall": aspect_recall,
            "F1 Score": aspect_f1_score
        }
    
    return aspect_metrics


aspect_metrics = evaluate_aspect_sentiment_metrics(dataset, categories)

    
 