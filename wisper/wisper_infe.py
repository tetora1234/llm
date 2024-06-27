import numpy as np
import torch
import librosa
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os
import gc


MODEL_PATH = r"C:\Users\nider\Desktop\git\llm\wisper\Visual-novel-transcriptor-hentai"
PROCESSOR_PATH = r"C:\Users\nider\Desktop\git\llm\wisper\Visual-novel-transcriptor-hentai"
AUDIO_FILE_PATH = r"C:\Users\nider\Desktop\419592.mp3"
OUTPUT_DIR = r"C:\Users\nider\Desktop\419592"

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_audio(file_path, target_sr=16000):
    audio, orig_sr = librosa.load(file_path, sr=None)
    if orig_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return {"array": audio, "sampling_rate": target_sr}

def split_audio(audio, chunk_length=30):
    sample_rate = audio["sampling_rate"]
    audio_array = audio["array"]
    chunk_size = chunk_length * sample_rate
    
    chunks = []
    for start in range(0, len(audio_array), chunk_size):
        end = min(start + chunk_size, len(audio_array))
        chunks.append({"array": audio_array[start:end], "sampling_rate": sample_rate})
    return chunks

def save_audio_chunks(chunks, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    chunk_paths = []
    for i, chunk in enumerate(chunks):
        chunk_path = os.path.join(output_dir, f"{i + 1}.wav")
        sf.write(chunk_path, chunk["array"], chunk["sampling_rate"])
        chunk_paths.append(chunk_path)
    
    return chunk_paths

def transcribe(audio_file_path, processor, model):
    audio_input, _ = librosa.load(audio_file_path, sr=16000)
    input_features = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to(device)
    
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="Japanese", task="transcribe")
    
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids
        )
    
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    del input_features, predicted_ids
    torch.cuda.empty_cache()
    gc.collect()
    
    return transcription[0]

def process_chunk(chunk_path, processor, model, output_dir, chunk_index):
    result = transcribe(chunk_path, processor, model)
    print(f'{chunk_index + 1}番目: {result}')
    return result

def save_full_transcription(transcriptions, output_dir):
    full_transcription_path = os.path.join(output_dir, "full_transcription.txt")
    with open(full_transcription_path, "w", encoding="utf-8") as f:
        for i, transcription in enumerate(transcriptions):
            audio_file = os.path.join(output_dir, f"{i + 1}.wav")
            f.write(f"{audio_file},{transcription}\n")
    print(f"Full transcription saved to {full_transcription_path}")

def main():
    processor = WhisperProcessor.from_pretrained(PROCESSOR_PATH, language="Japanese", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)

    audio = load_audio(AUDIO_FILE_PATH)
    audio_chunks = split_audio(audio)
    chunk_paths = save_audio_chunks(audio_chunks, OUTPUT_DIR)

    transcriptions = []
    batch_size = 5  # バッチサイズを設定

    for i in range(0, len(chunk_paths), batch_size):
        batch_chunks = chunk_paths[i:i+batch_size]
        batch_results = []

        for j, chunk_path in enumerate(batch_chunks):
            result = process_chunk(chunk_path, processor, model, OUTPUT_DIR, i+j)
            batch_results.append(result)

        transcriptions.extend(batch_results)
        save_full_transcription(transcriptions, OUTPUT_DIR)

        gc.collect()
        torch.cuda.empty_cache()

    print(f"Processed {len(chunk_paths)} chunks. Transcriptions saved to {OUTPUT_DIR}")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()