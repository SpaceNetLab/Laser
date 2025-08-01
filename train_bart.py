import os
from glob import glob
import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    BartTokenizerFast,
    BartForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Custom dataset class
class TxtDataset(Dataset):
    def __init__(self, folder, tokenizer, max_input_length=1024, max_target_length=256):
        self.files = glob(os.path.join(folder, '*.txt'))
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.examples = []
        self._prepare()

    def _prepare(self):
        for filepath in self.files:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read().strip()

            # Split into sections using single or double line breaks, compatible with different formats
            paras = [p.strip() for p in text.split('\n\n') if p.strip()]
            if len(paras) >= 3:
                label = '\n\n'.join(paras[:2])
                input_text = '\n\n'.join(paras[2:])
            elif len(paras) >= 2:
                label = paras[0]
                input_text = '\n\n'.join(paras[1:])
            elif len(paras) == 1:
                label = paras[0]
                input_text = ""
            else:
                continue  # Skip empty files

            # Add to the training sample list
            self.examples.append((input_text, label))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_text, label = self.examples[idx]
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        targets = self.tokenizer(
            label,
            max_length=self.max_target_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

def main():
    # === Configuration ===
    data_folder = 'paper_txt'            # Path to the txt file dataset
    model_path = 'bart_model'            # Local model folder
    output_dir = 'bart_seq2seq_output'   # Path to save training output

    # === Load model and tokenizer ===
    tokenizer = BartTokenizerFast.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path)

    # === Build dataset ===
    dataset = TxtDataset(data_folder, tokenizer)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # === Set training parameters ===
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        # evaluation_strategy='steps',
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        logging_steps=50,
        learning_rate=5e-5,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU is available
        overwrite_output_dir=True,
    )

    # === Initialize Trainer ===
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # === Train and save ===
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == '__main__':
    main()