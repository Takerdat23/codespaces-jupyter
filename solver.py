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



class Solver():
    def __init__(self, args):
        self.args = args

        self.model_dir = make_save_dir(args.model_dir)
        self.no_cuda = args.no_cuda
        if not os.path.exists(os.path.join(self.model_dir,'code')):
            os.makedirs(os.path.join(self.model_dir,'code'))
    
        


        self.tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

        df_train = pd.read_csv(self.args.train_path,  encoding = 'utf8')

        
        # data_utils_holder = data_utils(self.args)
        # data_yielder = data_utils_holder.train_data_yielder()
        self.categories = get_categories(df_train)
        dataset = process_data(df_train, self.categories)
        data_collator = SentimentDataCollator(self.tokenizer)
        self.train_loader = DataLoader(dataset, batch_size=self.args.batch_size, collate_fn=data_collator)

      
        self.model = ABSA_Tree_transfomer( vocab_size= self.tokenizer.vocab_size, N= 12, d_model= 768, 
                                          d_ff= 2048, h= 12, dropout = 0.1, num_categories = len(self.categories) , 
                                          no_cuda=args.no_cuda)


     




    def ModelSummary(self): 
        print(self.model)
    
    def LoadPretrain(self, model_dir): 
        path = os.path.join(model_dir, 'model.pth')
        return self.model.load_state_dict(torch.load(path)['state_dict'])
    
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

        self.model.to(device)
     
        optim = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
        #optim = BertAdam(self.model.parameters(), lr=1e-4)
        
        total_loss = []
        start = time.time()
        total_step_time = 0.
        total_masked = 0.
        total_token = 0.

        self.model.train()
        total_loss = []
        start = time.time()
        
        for epoch in range(self.args.epoch):
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

                if (step + 1) % 100 == 0:
                    elapsed = time.time() - start
                    print(f'Epoch [{epoch + 1}/{self.args.epoch}], Step [{step + 1}/{len(self.train_loader)}], '
                        f'Loss: {loss.item():.4f}, Total Time: {elapsed:.2f} sec')
                
                    # Save model
                    self.save_model(self.model, optim, epoch, step, self.model_dir)

                    start = time.time()



 