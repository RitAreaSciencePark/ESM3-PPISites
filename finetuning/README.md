## Finetuning 


To run the finetuning procedure run `python training_main.py` which reads the parameters from `train_jobs.csv`.
The parameters for the finetuning are ```model,train_file,val_file,test_file,dataset_type,epochs,lr,wd,batch_size,gradient_batch,done```

The finetuned models and the performances evaluated with  `python evaluation_finetuning.py` will be saved inside **results_finetuning**
