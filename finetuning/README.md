## Finetuning 


To run the finetuning the following sbatch runs training_main.py and reads the parameter from train_jobs.csv.
The parameters for the finetuning are ```model,train_file,val_file,test_file,dataset_type,epochs,lr,wd,batch_size,gradient_batch,done``` and can be found in the train_jobs.csv file

This training runs for the amounts of input epochs and select the best models.
``` 
sbatch train2.sh 
```