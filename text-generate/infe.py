import os
import pickle
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
import json
from train import TransformerXLConfig, TransformerXLModel

# テキスト生成関数
def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, top_k=50):
    model.eval()
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    mems = None
    generated = []
    
    with torch.no_grad():
        for _ in range(max_length):
            # input_ids の形状を確認し、必要に応じて調整
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            elif input_ids.dim() == 3:
                input_ids = input_ids.squeeze(1)
            
            outputs = model(input_ids, mems=mems)
            logits, _, new_mems = outputs  # アンパックを明示的に行う
            mems = new_mems
            
            logits = logits[:, -1, :] / temperature
            
            # Top-k サンプリング
            top_k_logits, top_k_indices = torch.topk(logits, k=top_k)
            probs = F.softmax(top_k_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices.gather(-1, next_token)
            
            generated.append(next_token.item())
            input_ids = next_token.unsqueeze(0)
            
            if next_token.item() == tokenizer.encode('<|endoftext|>')[0]:
                break
    
    return tokenizer.decode(generated)

if __name__ == "__main__":
    # モデルの読み込み
    model_path = "./text-generate/models/japanese_transformerxl_model.safetensors"
    config_path = "./text-generate/models/config.json"
    tokenizer_path = "./text-generate/models/tokenizer.pkl"

    # ファイルの存在を確認
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    # 設定の読み込み
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    config = TransformerXLConfig.from_dict(config_dict)

    # モデルの初期化と重みの読み込み
    model = TransformerXLModel(config)
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)

    # トークナイザーの読み込み
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # GPUが利用可能な場合はGPUを使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # テキスト生成の例
    prompt = "タイトル: 隣の部屋で、ボクっ娘幼馴染のＮＴＲごっくんフェラチオ\n本文: "
    generated_text = generate_text(model, tokenizer, prompt, max_length=200, temperature=0.7, top_k=50)

    print("生成されたテキスト:")
    print(generated_text)