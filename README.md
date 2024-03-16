# Tree_transformer

#Pretraining

python main.py -train -seq_length 100 -batch_size 64 -model_dir ./Model2 -train_path ./data/demo-full.txt -num_step 60000

# Training 
python main.py -task SA -train -seq_length 100 -batch_size 8 -model_dir ./ -train_path ./data/Train.csv -num_step 60000