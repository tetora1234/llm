import os
import subprocess
import gc
import pandas as pd
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import Dataset, DatasetDict, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# 定数と設定
CSV_PATH = r"C:\Users\nider\Desktop\git\llm\data\data.csv"
MODEL_NAME = r"C:\Users\nider\Desktop\git\llm\Visual-novel-transcriptor-hentai"
LANGUAGE = "Japanese"
TASK = "transcribe"
OUTPUT_DIR = "./Visual-novel-transcriptor-hentai2"
MODELPATH = r"C:\Users\nider\Desktop\git\llm\Visual-novel-transcriptor-hentai"
SAMPLING_RATE = 16000

# データの読み込みと前処理
def load_and_prepare_data(csv_path: str) -> DatasetDict:
    df = pd.read_csv(csv_path)
    
    train_dataset = Dataset.from_pandas(df).cast_column("ファイルパス", Audio(sampling_rate=SAMPLING_RATE)).rename_column("ファイルパス", "audio")
    
    return DatasetDict({
        "train": train_dataset,
    })

# データセットの前処理関数
def prepare_dataset(batch: Dict, processor: WhisperProcessor) -> Dict:
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = processor.tokenizer(batch["内容"]).input_ids
    return batch

# データコラトラの定義
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# メイン処理
if __name__ == "__main__":
    # データの読み込みと前処理
    datasets = load_and_prepare_data(CSV_PATH)
    
    # プロセッサとモデルの設定
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
    prepared_datasets = datasets.map(lambda batch: prepare_dataset(batch, processor), remove_columns=datasets.column_names["train"], num_proc=1)
    
    # モデルの設定
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME) #新規トレーニングコード
    # model = WhisperForConditionalGeneration.from_pretrained(MODELPATH) #途中保存したモデルから再開

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ja", task=TASK)
    model.config.suppress_tokens = []
        
    # データコラトラの設定
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        learning_rate=1e-5,
        max_steps=2000,
        fp16=True,
        group_by_length=True,
        evaluation_strategy="no",  # 評価を行わない設定
        save_steps=100,
        logging_steps=1,
        report_to=["tensorboard"],
        push_to_hub=False,
    )

    log_dir = training_args.output_dir
    os.makedirs(log_dir, exist_ok=True)
    subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", "6006"])

    # トレーナーの作成と学習の実行
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=prepared_datasets["train"],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )
    
    torch.cuda.empty_cache()  # メモリの解放
    gc.collect()  # ガベージコレクション
    trainer.train() #新規トレーニングコード
    # trainer.train(resume_from_checkpoint=True) #途中保存したモデルから再開

    # モデルの保存
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)