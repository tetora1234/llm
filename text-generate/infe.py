import os
import pickle
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file
import json
import math

# モデルの設定
class TransformerXLConfig:
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, mem_len, dropout=0.1):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.mem_len = mem_len
        self.dropout = dropout

    def to_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "n_embd": self.n_embd,
            "n_head": self.n_head,
            "n_layer": self.n_layer,
            "block_size": self.block_size,
            "mem_len": self.mem_len,
            "dropout": self.dropout
        }

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

# TransformerXLモデルの定義
class TransformerXLModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([TransformerXLBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, mems=None, targets=None):
        device = idx.device
        if isinstance(idx, (list, tuple)):
            idx = torch.tensor(idx, dtype=torch.long, device=device)
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)  # (1, seq_len)となるように調整
        elif idx.dim() == 3:
            idx = idx.squeeze(1)  # (batch, 1, seq_len) -> (batch, seq_len)
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # 以下は変更なし
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)

        new_mems = []
        if mems is None:
            mems = [None] * self.config.n_layer
        for layer, mem in zip(self.transformer.h, mems):
            x, new_mem = layer(x, mem)
            new_mems.append(new_mem)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # 損失計算
        loss = None
        if targets is not None:
            targets = targets[:, :logits.size(1)]
            logits = logits[:, :targets.size(1), :]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)

        return (logits, loss, new_mems)  # タプルとして明示的に返す

        # 損失計算
        loss = None
        if targets is not None:
            targets = targets[:, :logits.size(1)]
            logits = logits[:, :targets.size(1), :]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)

        return (logits, loss, new_mems)

# TransformerXLブロックの定義
class TransformerXLBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = RelativeMultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = nn.GELU(),
            dropout = nn.Dropout(config.dropout),
        ))

    def forward(self, x, mem):
        x, new_mem = self.attn(self.ln_1(x), mem)
        x = x + self.mlp_forward(self.ln_2(x))
        return x, new_mem

    def mlp_forward(self, x):
        x = self.mlp.c_fc(x)
        x = self.mlp.act(x)
        x = self.mlp.c_proj(x)
        x = self.mlp.dropout(x)
        return x

# 相対位置を考慮したマルチヘッド注意機構の定義
class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.mem_len = config.mem_len

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.r_emb = nn.Parameter(torch.Tensor(self.n_head, config.n_embd // self.n_head))
        self.r_bias = nn.Parameter(torch.Tensor(self.n_head, config.n_embd // self.n_head))

        nn.init.normal_(self.r_emb, 0.0, 0.02)
        nn.init.normal_(self.r_bias, 0.0, 0.02)

    def forward(self, x, mem):
        B, T, C = x.size()

        if mem is None:
            mem = torch.empty(B, 0, C).to(x.device)
        else:
            mem = mem.view(B, -1, C)

        cat = torch.cat([mem, x], dim=1)
        K = cat.size(1)

        q, k, v = self.c_attn(cat).split(self.n_embd, dim=2)
        q = q.view(B, K, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, K, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, K, self.n_head, C // self.n_head).transpose(1, 2)

        q = q + self.r_bias.view(1, self.n_head, 1, C // self.n_head)
        r = self.r_emb.unsqueeze(1).expand(self.n_head, K, C // self.n_head)

        AC = torch.matmul(q, k.transpose(-2, -1))
        BD = torch.matmul(q, r.transpose(-2, -1))
        BD = self._rel_shift(BD)

        attn = AC + BD
        attn = attn / math.sqrt(self.n_embd // self.n_head)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        y = torch.matmul(attn, v)
        y = y.transpose(1, 2).contiguous().view(B, K, C)

        y = self.resid_dropout(self.c_proj(y))

        new_mem = cat[:, -self.mem_len:]

        return y, new_mem

    def _rel_shift(self, x):
        zero_pad = torch.zeros((x.size(0), x.size(1), x.size(2), 1),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=3)
        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)
        return x

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