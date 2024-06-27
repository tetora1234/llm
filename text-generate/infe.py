import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

model_path = r"C:\Users\nider\Desktop\git\llm\My-Finetuned-Japanese-Model-1b"

# GPUが利用可能かチェック
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用デバイス: {device}")

# トークナイザーとモデルの読み込み
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# トークナイザーの語彙サイズをモデルに合わせる
if len(tokenizer) != model.config.vocab_size:
    print("トークナイザーの語彙サイズを更新しています...")
    tokenizer.resize_token_embeddings(model.config.vocab_size)

print(f"更新後のトークナイザーの語彙サイズ: {len(tokenizer)}")
print(f"モデルの語彙サイズ: {model.config.vocab_size}")

model.to(device)

print("モデルの読み込みが完了しました。")

initial_prompt = """タイトル：隣の部屋で、ボクっ娘幼馴染のＮＴＲごっくんフェラチオ

"""

def generate_text_stream(prompt, max_new_tokens=1000):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    generated_tokens = []
    for i in range(max_new_tokens):
        with torch.no_grad():
            output = model.generate(
                inputs.input_ids,
                max_new_tokens=1,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                top_k=100,
                repetition_penalty=1.1,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        new_token = output[0][-1]
        # トークンIDをモデルの語彙サイズ内に制限
        new_token = torch.clamp(new_token, max=model.config.vocab_size-1)
        generated_tokens.append(new_token)
        
        try:
            decoded_token = tokenizer.decode(new_token)
            print(decoded_token, end='', flush=True)
            sys.stdout.flush()
        except KeyError:
            print(f"[Unknown token: {new_token.item()}]", end='', flush=True)
        
        if new_token.item() == tokenizer.eos_token_id:
            break
        
        inputs.input_ids = torch.cat([inputs.input_ids, new_token.unsqueeze(0)], dim=-1)
    
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

try:
    generated_text = generate_text_stream(initial_prompt)
    initial_prompt += generated_text

    # コンテキストの長さを管理
    max_context_length = 10000
    if len(initial_prompt) > max_context_length:
        initial_prompt = initial_prompt[-max_context_length:]

    print("\n\n生成されたテキスト:")
    print(initial_prompt)

except Exception as e:
    print(f"エラーが発生しました: {str(e)}")
    print("トークナイザーとモデルの詳細:")
    print(f"トークナイザータイプ: {type(tokenizer)}")
    print(f"モデルタイプ: {type(model)}")
    print(f"トークナイザーの語彙サイズ: {len(tokenizer)}")
    print(f"モデルの語彙サイズ: {model.config.vocab_size}")
    raise  # エラーの詳細な情報を表示