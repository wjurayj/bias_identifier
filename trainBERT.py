import torch
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from utils import TokenizedScalingDataset, compute_metrics



data_dir = '../work/batch_output/cleaned/'
fine_tune = False

df_train = pd.read_csv(data_dir + 'train.text.tsv', sep='\t', header=None, index_col=0)
df_train[4] = df_train[4].apply(lambda x: str(x).replace('<splt>', ''))

df_val = pd.read_csv(data_dir + 'val.text.tsv', sep='\t', header=None, index_col=0)
df_val[4] = df_val[4].apply(lambda x: str(x).replace('<splt>', ''))

df_test = pd.read_csv(data_dir + 'test.text.tsv', sep='\t', header=None, index_col=0)
df_test[4] = df_test[4].apply(lambda x: str(x).replace('<splt>', ''))

df_withheld = pd.read_csv(data_dir + 'withheld_val.text.tsv', sep='\t', header=None, index_col=0)
df_withheld[4] = df_withheld[4].apply(lambda x: str(x).replace('<splt>', ''))


model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.train()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

l_encoder = LabelEncoder()

train_tokens = tokenizer(list(df_train[4]), padding=True, truncation=True, return_tensors='pt')
val_tokens = tokenizer(list(df_val[4]), padding=True, truncation=True, return_tensors='pt')
test_tokens = tokenizer(list(df_test[4]), padding=True, truncation=True, return_tensors='pt')
withheld_tokens = tokenizer(list(df_withheld[4]), padding=True, truncation=True, return_tensors='pt')

train_dataset=TokenizedScalingDataset(train_tokens, l_encoder.fit_transform(list(df_train[1])))
val_dataset=TokenizedScalingDataset(val_tokens, l_encoder.fit_transform(list(df_val[1])))
test_dataset=TokenizedScalingDataset(test_tokens, l_encoder.fit_transform(list(df_test[1])))
wh_dataset=TokenizedScalingDataset(withheld_tokens, l_encoder.fit_transform(list(df_withheld[1])))


training_args_dev = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=1,              # total # of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=100,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    evaluation_strategy='steps',     # evaluate every 500 steps
    save_strategy='steps'            # save model every 500 steps
)


training_args_test = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=5,              # total # of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=20,                 # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    evaluation_strategy='steps',     # evaluate every 500 steps
    save_strategy='steps'            # save model every 500 steps
)

trainer_dev = Trainer(
    model=model,                       # the instantiated 🤗 Transformers model to be trained
    args=training_args_test,                # training arguments, defined above
    compute_metrics=compute_metrics,   # metric computation specs
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset            # evaluation dataset
)

trainer_test = Trainer(
    model=model,                       # the instantiated 🤗 Transformers model to be trained
    args=training_args_test,                # training arguments, defined above
    compute_metrics=compute_metrics,   # metric computation specs
    train_dataset=test_dataset,         # training dataset
    eval_dataset=wh_dataset            # evaluation dataset
)

if __name__ == '__main__':
    trainer_dev.train()
    print(trainer_dev.evaluate())
    print(trainer_dev.evaluate(withheld_dataset))

    trainer_test.train()
    print(trainer_test.evaluate())
    print(trainer_test.evaluate(val_dataset))
