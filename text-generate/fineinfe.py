import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# モデルとトークナイザーの読み込み
model_path = r"C:\Users\nider\Desktop\git\llm\text-generate\models\checkpoint-4000"
model_name = "cyberagent/open-calm-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_path)

# GPUが利用可能な場合はGPUを使用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_text(prompt, min_length, max_length):
    # 入力テキストのエンコード
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # テキスト生成
    with torch.no_grad():
        output = model.generate(
            input_ids,
            min_length=min_length,
            max_length=max_length,
            num_return_sequences=3,  # デフォルト: 1
            no_repeat_ngram_size=0,  # デフォルト: 0
            do_sample=True,  # デフォルト: False
            top_k=25,  # デフォルト: 50
            top_p=1.00,  # デフォルト: 1.0
            temperature=1.0,  # デフォルト: 1.0
            repetition_penalty=1.0  # デフォルト: 1.0
        )

    # 生成されたテキストのデコード
    generated_texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in output]
    return generated_texts

# テキスト生成の例
prompt = "タイトル: 公衆便所でおトイレ扱いえっち\n本文:"
generated_texts = generate_text(prompt, min_length=300, max_length=500)
print(f"Prompt: {prompt}")
for i, text in enumerate(generated_texts, 1):
    print(f"\nGenerated text {i}:")
    print(text)