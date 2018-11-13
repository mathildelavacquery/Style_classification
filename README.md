# Style Classification


### 1. Use our drive (data and functions are already downloaded) : 
https://drive.google.com/drive/folders/1TxWlhFs9hKlKEzAbL6Mw6PHLIWjCF7-w?usp=sharing

### 2. You can run directly the notebook : Notebook art classification project, in the drive 

## OR 

### 1. Download the data from these URL:
https://www.kaggle.com/c/painter-by-numbers/download/train_1.zip

https://www.kaggle.com/c/painter-by-numbers/download/train_2.zip

https://www.kaggle.com/c/painter-by-numbers/data/train_info.csv

ATTENTION : need to have a kaggle account and to approave the project

### 2. Unzip the files in the directory raw_data, organisation of the directory should be:

```bash
├── raw_data
│   ├── train_1
│   ├── train_2
│   └── train_info.csv
├── data
│   ├── classif_class1_class2
│   ├── train
│   └── valid
├── main.py
├── class_dataloader.py
├── features_generat.py
├── 
└── 
```
 The data folder will be generated automatically by the DataLoader


### 3. run: 
```bash
python main.py -o False -rd raw_data -dd data
```
