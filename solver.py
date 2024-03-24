import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import subprocess
from models import *
from utils import *
from parse import *
import random
from bert_optimizer import BertAdam
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import wandb

class Solver():
    def __init__(self, args):
        self.args = args

        self.model_dir = make_save_dir(args.model_dir)
        self.no_cuda = args.no_cuda
        if not os.path.exists(os.path.join(self.model_dir,'code')):
            os.makedirs(os.path.join(self.model_dir,'code'))
    
        if args.wandb_api != "": 

            wandb.login(key=args.wandb_api)

   
         

        self.tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

        df_train = pd.read_csv(self.args.train_path,  encoding = 'utf8')
        df_val = pd.read_csv(self.args.valid_path,  encoding = 'utf8')


        
        # data_utils_holder = data_utils(self.args)
        # data_yielder = data_utils_holder.train_data_yielder()
        self.categories = get_categories(df_train)
        dataset = process_data(df_train, self.categories)
        val_dataset =  process_data(df_val, self.categories)
        data_collator = SentimentDataCollator(self.tokenizer)
        self.train_loader = DataLoader(dataset, batch_size=self.args.batch_size, collate_fn=data_collator)
        self.val_loader =DataLoader(val_dataset, batch_size=self.args.batch_size, collate_fn=data_collator)
      
        self.model = ABSA_Tree_transfomer( vocab_size= self.tokenizer.vocab_size, N= 12, d_model= 768, 
                                          d_ff= 2048, h= 12, dropout = 0.1, num_categories = len(self.categories) , 
                                          no_cuda=args.no_cuda)


     




    def ModelSummary(self): 
        print(self.model)
    
    def LoadPretrain(self, model_dir): 
        path = os.path.join(model_dir, 'model.pth')
        return self.model.load_state_dict(torch.load(path)['state_dict'])
    
    # def evaluate_f1_accuracy(self):
    #     if self.args.no_cuda == False:
    #         device = "cuda"
    #     else:
    #         device = "cpu"
        
    #     self.model.to(device)
    #     self.model.eval()
        
    #     all_predictions = []
    #     all_ground_truth = []
        
    #     with torch.no_grad():
    #         for step, batch in tqdm(enumerate(self.val_loader)):
    #             inputs = batch['input_ids'].to(device)
    #             mask = batch['attention_mask'].to(device)
    #             labels = batch['labels'].to(device)

    #             output = self.model(inputs, mask, self.categories)

    #             output = torch.sigmoid(output)
    #             output = output.float()
    #             labels = labels.float()
                
    #             # Convert probabilities to binary predictions
    #             predictions = torch.argmax(output, dim=2)
    #             ground_truth = torch.argmax(labels, dim=2)
                
    #             # Flatten predictions and ground truth tensors
    #             predictions_flat = predictions.view(-1).cpu().numpy()
    #             ground_truth_flat = ground_truth.view(-1).cpu().numpy()
                
    #             all_predictions.extend(predictions_flat)
    #             all_ground_truth.extend(ground_truth_flat)
        
    #     # Compute F1 score
    #     f1_accuracy = f1_score(all_ground_truth, all_predictions, average='weighted')
        
    #     return f1_accuracy
    
    def evaluate(self):
        if self.args.no_cuda == False: 
            device = "cuda"
        else:
            device = "cpu"

        self.model.to(device)
        self.model.eval()
        aspect_true = []
        aspect_pred = []
        sentiment_true = []
        sentiment_pred = []

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                output = self.model(inputs, mask, self.categories)

                output = torch.sigmoid(output)
                output = output.float()
                labels = labels.float()

                # Iterate over each category
                for category_index in range(output.shape[1]):
                    # Extract aspect labels and predictions
                    aspect_labels = torch.argmax(labels, dim=2)
                    aspect_predictions = torch.argmax(output[:, :, :3], dim=2) 
                 
                   
                    # Extract sentiment labels and predictions
                    sentiment_labels = labels[:, category_index, :].argmax(axis=1)  # Extract sentiment labels correctly
                    sentiment_predictions = output[:, category_index, :].argmax(axis=1)
                   


                    # Extend lists
                    aspect_true.extend(aspect_labels)
                    aspect_pred.extend(aspect_predictions)
                    sentiment_true.extend(sentiment_labels)
                    sentiment_pred.extend(sentiment_predictions)

   
        aspect_accuracy = accuracy_score(aspect_true, aspect_pred)
        sentiment_accuracy = accuracy_score(sentiment_true, sentiment_pred)
        Total = (aspect_accuracy + sentiment_accuracy)/2

        return Total, aspect_accuracy, sentiment_accuracy

    def evaluate_aspect_sentiment_accuracy(self):
        if self.args.no_cuda == False:
            device = "cuda"
        else:
            device = "cpu"
        
        self.model.to(device)
        self.model.eval()
        
        all_aspect_predictions = []
        all_sentiment_predictions = []
        all_aspect_ground_truth = []
        all_sentiment_ground_truth = []
        
        with torch.no_grad():
            for step, batch in enumerate(self.val_loader):
                inputs = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                output = self.model(inputs, mask, self.categories)

                output = torch.sigmoid(output)
                output = output.float()
                labels = labels.float()
                
                predictions = (output > 0.5).long()  
              
                
                aspect_predictions = predictions[:, :, 0]  # Extract aspect predictions
                
                sentiment_predictions = predictions[:, :, 1:]  # Extract sentiment predictions
                
                ground_truth = labels[:, :, 0]  # Extract aspect ground truth
              
                sentiment_ground_truth = labels[:, :, 1:]  # Extract sentiment ground truth
           
                
                
                aspect_predictions_flat = aspect_predictions.view(-1).cpu().numpy()
                sentiment_predictions_flat = sentiment_predictions.view(-1, 3).cpu().numpy()  # Assuming there are 3 sentiment labels
                
                aspect_ground_truth_flat = ground_truth.view(-1).cpu().numpy()
                sentiment_ground_truth_flat = sentiment_ground_truth.view(-1, 3).cpu().numpy()  # Assuming there are 3 sentiment labels
                
                all_aspect_predictions.extend(aspect_predictions_flat)
                all_aspect_ground_truth.extend(aspect_ground_truth_flat)
                
                all_sentiment_predictions.extend(sentiment_predictions_flat)
                all_sentiment_ground_truth.extend(sentiment_ground_truth_flat)
        
        aspect_accuracy = accuracy_score(all_aspect_ground_truth, all_aspect_predictions)
        sentiment_accuracy = accuracy_score(all_sentiment_ground_truth, all_sentiment_predictions)
        
        return aspect_accuracy, sentiment_accuracy
    

    def evaluate_aspect_sentiment_accuracy2(self):
        if self.args.no_cuda == False:
            device = "cuda"
        else:
            device = "cpu"

        self.model.to(device)
        self.model.eval()

        all_aspect_predictions = []
        all_sentiment_predictions = []
        all_aspect_ground_truth = []
        all_sentiment_ground_truth = []

        with torch.no_grad():
            for step, batch in enumerate(self.val_loader):
                inputs = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                output = self.model(inputs, mask, self.categories)

                output = torch.sigmoid(output)
                output = output.float()

                aspect_predictions = (output[:, :, 0] > 0.5).long()
                sentiment_predictions = (output[:, :, 1:] > 0.5).long()

                aspect_ground_truth = labels[:, :, 0].long()
                sentiment_ground_truth = labels[:, :, 1:].long()

                all_aspect_predictions.append(aspect_predictions.cpu().numpy())
                all_sentiment_predictions.append(sentiment_predictions.cpu().numpy())
                all_aspect_ground_truth.append(aspect_ground_truth.cpu().numpy())
                all_sentiment_ground_truth.append(sentiment_ground_truth.cpu().numpy())

        all_aspect_predictions = np.concatenate(all_aspect_predictions)
        all_sentiment_predictions = np.concatenate(all_sentiment_predictions)
        all_aspect_ground_truth = np.concatenate(all_aspect_ground_truth)
        all_sentiment_ground_truth = np.concatenate(all_sentiment_ground_truth)

        aspect_accuracy = accuracy_score(all_aspect_ground_truth.flatten(), all_aspect_predictions.flatten())
        sentiment_accuracy = accuracy_score(all_sentiment_ground_truth.flatten(), all_sentiment_predictions.flatten())

        return aspect_accuracy, sentiment_accuracy
    
    def save_model(self, model, optimizer, epoch, step, model_dir):
        model_name = f'model_epoch_{epoch}_step_{step}.pth'
        state = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, os.path.join(model_dir, model_name))


    def train(self):
        if self.args.no_cuda == False: 
            device = "cuda"
        else: 
            device = "cpu"
        if self.args.load:
            path = os.path.join(self.model_dir, 'model.pth')
            self.model.load_state_dict(torch.load(path)['state_dict'])
        tt = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                #print(name)
                ttt = 1
                for  s in param.data.size():
                    ttt *= s
                tt += ttt
        print('total_param_num:',tt)
        if (self.args.wandb_api != ""):
            wandb.init(project="Tree_transformer", name="training ABSA")


        self.model.to(device)
     
        optim = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
        #optim = BertAdam(self.model.parameters(), lr=1e-4)
        
        total_loss = []
        start = time.time()
     

        self.model.train()
        total_loss = []
        start = time.time()
        
        for epoch in tqdm(range(self.args.epoch)):
            epoch_progress = tqdm(total=len(self.train_loader), desc=f'Epoch {epoch+1}/{self.args.epoch}', position=0)

            for step, batch in enumerate(self.train_loader):
                inputs = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optim.zero_grad()
                output = self.model(inputs, mask, self.categories)  # Assuming categories are not used for now
            
                # Calculate loss
              
                output = torch.sigmoid(output)
                output = output.float()
                labels = labels.float()
                loss = F.binary_cross_entropy(output, labels)
                
                # loss = self.model.masked_lm_loss(output, labels)
                total_loss.append(loss.item())

                # Backpropagation
                loss.backward()
                optim.step()
                if (self.args.wandb_api != ""):
                    wandb.log({"Loss": loss.item()}, step=epoch*len(self.train_loader) + step)
                epoch_progress.update(1)
                epoch_progress.set_postfix({'Loss': loss.item()})

                if (step + 1) % 100 == 0:
                    elapsed = time.time() - start
                    print(f'Epoch [{epoch + 1}/{self.args.epoch}], Step [{step + 1}/{len(self.train_loader)}], '
                        f'Loss: {loss.item():.4f}, Total Time: {elapsed:.2f} sec')
                    # aspect , sentiment = self.evaluate_aspect_sentiment_accuracy2()
               
                    # print(f"Epoch {epoch} Validation accuracy (Aspect): ", aspect)
                    # print(f"Epoch {epoch} Validation accuracy (Sentiment): ", sentiment)
            epoch_progress.close()
            #Valid stage 
            aspect , sentiment = self.evaluate_aspect_sentiment_accuracy2()
               
            print(f"Epoch {epoch} Validation accuracy (Aspect): ", aspect)
            print(f"Epoch {epoch} Validation accuracy (Sentiment): ", sentiment)

            combined_accuracy = (aspect + sentiment) / 2
            if (self.args.wandb_api != ""):
                wandb.log({"Validation Accuracy": combined_accuracy}, step=epoch)
           
                    
                
                 
        self.save_model(self.model, optim, self.args.epoch, step, self.model_dir)


 