# Tree_transformer

# Pretraining

python main.py -train -seq_length 100 -batch_size 64 -model_dir ./Model2 -train_path ./data/demo-full.txt -num_step 60000

# Training 
python main.py -train -seq_length 128 -batch_size 64 -model_dir ./Model -train_path ./data/UIT-ViSFD/Train.csv -epoch 5 -no_cuda