import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from encode_swe import SWEEncoder_ja
import json
import math
from safetensors.torch import save_file

# トークナイザーの初期化
with open('./text-generate/ja-swe32kfix.txt', 'r', encoding='utf-8') as f:
    bpe = f.read().split('\n')
with open('./text-generate/emoji.json', 'r', encoding='utf-8') as f:
    emoji = json.loads(f.read())
tokenizer = SWEEncoder_ja(bpe, emoji)

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

        # パラメータの初期化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, mems=None, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # 順伝播
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
            # ターゲットを入力と同じサイズに調整
            targets = targets[:, :logits.size(1)]
            logits = logits[:, :targets.size(1), :]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)

        return logits, loss, new_mems

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
            # メモリの形状を調整
            mem = mem.view(B, -1, C)

        # メモリと入力を結合
        cat = torch.cat([mem, x], dim=1)
        K = cat.size(1)

        # 計算効率のために、クエリ、キー、バリューを一度に計算
        q, k, v = self.c_attn(cat).split(self.n_embd, dim=2)
        q = q.view(B, K, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, K, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, K, self.n_head, C // self.n_head).transpose(1, 2)

        # 相対位置エンコーディングを適用
        q = q + self.r_bias.view(1, self.n_head, 1, C // self.n_head)
        r = self.r_emb.unsqueeze(1).expand(self.n_head, K, C // self.n_head)

        # 注意機構の計算
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

        # 新しいメモリを作成
        new_mem = cat[:, -self.mem_len:]

        return y, new_mem

    def _rel_shift(self, x):
        zero_pad = torch.zeros((x.size(0), x.size(1), x.size(2), 1),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=3)
        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)
        return x

# データセットの定義（メモリ対応）
class TextDataset(Dataset):
    def __init__(self, data, config):
        self.tokenizer = tokenizer
        self.block_size = config.block_size
        self.mem_len = config.mem_len
        self.n_embd = config.n_embd
        self.examples = []
        
        for item in data:
            text = f"タイトル: {item['タイトル']}\n本文: {item['本文']}"
            tokenized = self.tokenizer.encode(text)
            if len(tokenized) > self.block_size:
                for i in range(0, len(tokenized) - self.block_size + 1, self.block_size):
                    chunk = tokenized[i:i + self.block_size]
                    mem_start = max(0, i - self.mem_len)
                    mem = tokenized[mem_start:i]
                    # メモリサイズを固定
                    if len(mem) < self.mem_len:
                        mem = [0] * (self.mem_len - len(mem)) + mem
                    else:
                        mem = mem[-self.mem_len:]
                    # メモリを埋め込み次元に調整
                    mem = torch.tensor(mem).unsqueeze(1).expand(-1, config.n_embd).tolist()
                    self.examples.append((chunk, mem))
            else:
                # 短いテキストの場合、パディングを追加
                chunk = tokenized + [0] * (self.block_size - len(tokenized))
                mem = [0] * self.mem_len
                # メモリを埋め込み次元に調整
                mem = torch.tensor(mem).unsqueeze(1).expand(-1, config.n_embd).tolist()
                self.examples.append((chunk, mem))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        chunk, mem = self.examples[i]
        # チャンクとメモリのサイズを一致させる
        chunk = chunk[:self.block_size]
        mem = mem[:self.mem_len]
        # 必要に応じてパディング
        if len(chunk) < self.block_size:
            chunk = chunk + [0] * (self.block_size - len(chunk))
        if len(mem) < self.mem_len:
            mem = mem + [[0] * self.n_embd for _ in range(self.mem_len - len(mem))]
        return torch.tensor(chunk, dtype=torch.long), torch.tensor(mem, dtype=torch.float)

# トレーニング関数（メモリ対応）
def train(model, train_dataset, val_dataset, epochs, batch_size, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch, mem in train_loader:
            print(f"Batch shape: {batch.shape}")
            print(f"Mem shape: {mem.shape}")
            
            batch = batch.to(device)
            mem = mem.to(device)
            
            # バッチとメモリの形状を調整
            B, T = batch.size()
            mem = mem.view(B, model.config.mem_len, model.config.n_embd)
            mems = [mem] * model.config.n_layer
            
            optimizer.zero_grad()
            _, loss, _ = model(batch, mems=mems, targets=batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader):.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch, mem in val_loader:
                batch = batch.to(device)
                mem = mem.to(device)
                
                # メモリの形状を調整
                B, T = batch.size()
                mem = mem.view(B, model.config.mem_len, model.config.n_embd)
                mems = [mem] * model.config.n_layer
                
                _, loss, _ = model(batch, mems=mems, targets=batch)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss/len(val_loader):.4f}")
          
if __name__ == "__main__":
    # モデル設定
    config = TransformerXLConfig(
        vocab_size=32000,  # トークナイザーの語彙サイズ
        n_embd=768,        # 埋め込みの次元
        n_head=4,         # 注意ヘッドの数
        n_layer=4,        # レイヤー数
        block_size=1024,   # 最大シーケンス長
        mem_len=1024,      # メモリ長
        dropout=0.1        # ドロップアウト率
    )

    # モデルの初期化
    model = TransformerXLModel(config)

    # データの読み込みと前処理
    with open('./text-generate/instruction_dataset.jsonl', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # データセットの作成
    train_size = int(0.9 * len(data))
    train_data, val_data = data[:train_size], data[train_size:]
    train_dataset = TextDataset(train_data, config)
    val_dataset = TextDataset(val_data, config)

    # トレーニングの実行
    train(model, train_dataset, val_dataset, epochs=10, batch_size=4, lr=3e-5)

    # モデルの保存（safetensors形式）
    state_dict = model.state_dict()
    save_file(state_dict, "./text-generate/models/japanese_transformerxl_model.safetensors")

    print("トレーニングが完了し、モデルがsafetensors形式で保存されました。")