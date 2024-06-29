import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import json

# Open-CALM-1bモデルとトークナイザーの初期化
model_name = "cyberagent/open-calm-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# データセットの定義
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        for item in data:
            text = f"タイトル: {item['タイトル']}\n本文: {item['本文']}"
            encoded = self.tokenizer.encode_plus(
                text, 
                max_length=self.max_length, 
                padding='max_length', 
                truncation=True, 
                return_tensors='pt'
            )
            self.examples.append({
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
                'labels': encoded['input_ids'].squeeze()
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

# メイン実行部分
if __name__ == "__main__":
    # データの読み込みと前処理
    with open('./text-generate/filtered_instruction_dataset.jsonl', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # 保存ディレクトリの作成
    save_directory = "./text-generate/models"
    os.makedirs(save_directory, exist_ok=True)

    # データセットの作成
    train_size = int(0.9 * len(data))
    train_data, val_data = data[:train_size], data[train_size:]
    train_dataset = TextDataset(train_data, tokenizer, max_length=512)
    val_dataset = TextDataset(val_data, tokenizer, max_length=512)

    # トレーニング引数の設定
    training_args = TrainingArguments(
        output_dir=save_directory,
        num_train_epochs=3,  # エポック数を設定（例：3エポック）
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_ratio=0.1,  # ウォームアップステップの代わりに比率を使用
        weight_decay=0.01,
        logging_dir='./text-generate/logs',
        logging_strategy="epoch",  # エポックごとにログを記録
        evaluation_strategy="epoch",  # エポックごとに評価
        save_strategy="epoch",  # エポックごとに保存
        save_total_limit=3,  # 保存するチェックポイントの数を制限
        fp16=True,  # 混合精度トレーニングを有効化
        load_best_model_at_end=True,  # トレーニング終了時に最良のモデルをロード
        metric_for_best_model="eval_loss",  # 最良のモデルを判断する指標
    )

    # Trainerの初期化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # トレーニングの実行
    trainer.train()

    # 最終モデルの保存
    trainer.save_model(save_directory)
    tokenizer.save_pretrained(save_directory)