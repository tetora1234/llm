import re
import json
import chardet
import os

# 入力フォルダのパス
input_folder_path = r"C:\Users\nider\Downloads"
# 出力ファイルのパス
output_file_path = os.path.join(r"C:\Users\nider\Desktop\git\lib\text_format", "tracks.jsonl")

# テキストをクリーンアップする関数
def clean_text(text):
    text = re.sub(r'▼', '', text, flags=re.MULTILINE)
    text = re.sub(r'^;.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'/SE:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'/SE：.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'/SE.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'SE.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'；.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'ー{2,}', '', text)
    text = re.sub(r'【.*?】', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[「」]', '', text)
    text = re.sub(r'●', '', text)
    text = re.sub(r'/', '', text)
    text = re.sub(r'―', '', text)
    return text.strip()

# タイトルから余分な文字を削除する関数
def clean_title(title):
    return re.sub(r'^トラック\d+[：。]?\s*', '', title).strip()

# ファイルのエンコーディングを検出する関数
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    return chardet.detect(raw_data)['encoding']

# ファイルを処理する関数
def process_file(input_file_path, output_file):
    encoding = detect_encoding(input_file_path)
    filename = os.path.splitext(os.path.basename(input_file_path))[0]
    
    with open(input_file_path, 'r', encoding=encoding) as file:
        text = file.read()

    cleaned_text = clean_text(text)
    tracks = re.split(r'(トラック\d+.*?)\s', cleaned_text)
    tracks = [t for t in tracks if t.strip()]

    while tracks and not tracks[0].startswith('トラック'):
        tracks.pop(0)
    
    for i in range(0, len(tracks) - 1, 2):
        title = clean_title(tracks[i])
        content = tracks[i+1].strip() if i+1 < len(tracks) else ""
        
        if title and content:
            # full_title = f"{filename}の{title}"
            full_title = f"{filename}"
            entry = {
                "タイトル": full_title,
                "本文": content
            }
            json.dump(entry, output_file, ensure_ascii=False)
            output_file.write('\n')

    print(f"処理完了: {input_file_path}")
    print(f"入力ファイルのエンコーディング: {encoding}")

# 出力ディレクトリが存在しない場合は作成
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# メイン処理
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for filename in os.listdir(input_folder_path):
        if filename.endswith('.txt'):
            input_file_path = os.path.join(input_folder_path, filename)
            process_file(input_file_path, output_file)

print(f"すべてのファイルの処理が完了しました。")
print(f"出力ファイル: {output_file_path}")