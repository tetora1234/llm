import os
import torch
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_file, save_file

def merge_models_layer_by_layer(model_path_A, model_path_B, output_path, alpha=0.7, beta=0.3):
    # 設定を読み込む
    config_A = AutoConfig.from_pretrained(model_path_A)
    config_B = AutoConfig.from_pretrained(model_path_B)

    # 出力ディレクトリを作成
    os.makedirs(output_path, exist_ok=True)

    # safetensorsファイルを見つける
    files_A = [f for f in os.listdir(model_path_A) if f.endswith('.safetensors')]
    files_B = [f for f in os.listdir(model_path_B) if f.endswith('.safetensors')]

    # 各safetensorsファイルを処理
    for file_A in files_A:
        if file_A in files_B:  # 両方のモデルに存在する場合のみマージ
            print(f"Processing {file_A}")
            tensor_A = load_file(os.path.join(model_path_A, file_A))
            tensor_B = load_file(os.path.join(model_path_B, file_A))

            merged_tensor = {}
            for key in tensor_A.keys():
                if key in tensor_B:
                    merged_tensor[key] = alpha * tensor_A[key] + beta * tensor_B[key]
                else:
                    merged_tensor[key] = tensor_A[key]

            # マージしたテンソルを保存
            save_file(merged_tensor, os.path.join(output_path, file_A))
        else:
            # モデルAにのみ存在する場合はそのままコピー
            print(f"Copying {file_A} from model A")
            tensor_A = load_file(os.path.join(model_path_A, file_A))
            save_file(tensor_A, os.path.join(output_path, file_A))

    # 設定ファイルを保存
    config_A.save_pretrained(output_path)

    # トークナイザーをコピー
    tokenizer = AutoTokenizer.from_pretrained(model_path_A)
    tokenizer.save_pretrained(output_path)

    print(f"Merged model saved to {output_path}")

# 使用例
model_path_A = r"C:\Users\nider\Desktop\git\llm\text-generate\models\Vecteus-V2-7B"
model_path_B = r"C:\Users\nider\Desktop\git\llm\text-generate\models\Ninja-V2-7B"
output_path = r"C:\Users\nider\Desktop\git\llm\text-generate\models\Merged-Model"

merge_models_layer_by_layer(model_path_A, model_path_B, output_path, alpha=0.7, beta=0.3)