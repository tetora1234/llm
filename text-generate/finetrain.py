import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import json

# Open-CALM-1bモデルとトークナイザーの初期化
model_name = "cyberagent/open-calm-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# データセットの定義
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        text = f"タイトル: {item['タイトル']}\n本文: {item['本文']}"
        encoded = self.tokenizer.encode_plus(
            text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': encoded['input_ids'].squeeze()
        }

# メイン実行部分
if __name__ == "__main__":
    # データの読み込みと前処理
    with open('./text-generate/filtered_instruction_dataset.jsonl', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # 保存ディレクトリの作成
    save_directory = "./text-generate/models"
    os.makedirs(save_directory, exist_ok=True)

    # データセットの作成
    train_dataset = TextDataset(data, tokenizer, max_length=2048)

    # トレーニング引数の設定
    training_args = TrainingArguments(
        output_dir=save_directory,
        num_train_epochs=10000,
        per_device_train_batch_size=1,
        logging_dir='./text-generate/logs',
        logging_steps=100,
        save_steps=500,
        save_total_limit=10,
        disable_tqdm=False,
        fp16=True,
        gradient_accumulation_steps=16,
        learning_rate=1e-5,
        lr_scheduler_type="linear",
        warmup_steps=500,  # ウォームアップステップ数
    )

    # Trainerの初期化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # トレーニングの実行
    trainer.train()

    # 最終モデルの保存
    trainer.save_model(save_directory)
    tokenizer.save_pretrained(save_directory)

print("Training completed and model saved.")