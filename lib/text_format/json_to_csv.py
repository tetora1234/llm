import pandas as pd
import json
import re

# パスと設定
json_file_path = r"C:\Users\nider\Desktop\git\lib\text_format\index.json"
eroge_name = 'ωstar_Bishoujo Mangekyou -Tsumi to Batsu no Shoujo-'
regex_file_path = r"C:\Users\nider\Desktop\git\lib\text_format\regex_patterns.txt"
csv_file_path_not_filtered = r"C:\Users\nider\Desktop\git\lib\text_format\output_not_filtered.csv"
csv_file_path_filtered = r"C:\Users\nider\Desktop\git\lib\text_format\output_filtered.csv"
base_file_path = base_file_path = "C:\\Users\\nider\\Desktop\\git\\llm\\data\\audio\\" + eroge_name + "\\"

# 正規表現パターンをファイルから読み込む
with open(regex_file_path, 'r', encoding='utf-8') as regex_file:
    regex_list = [line.strip() for line in regex_file if line.strip()]

# JSONファイルを読み込む
with open(json_file_path, 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# DataFrameを作成
df = pd.DataFrame(json_data)
df = df[['FilePath', 'Text']]

# FilePath列の各値にファイルパスを付与
df['FilePath'] = base_file_path + df['FilePath']

# 正規表現に一致する行だけを抽出する関数
def filter_text(text, regex_list):
    for regex_pattern in regex_list:
        if re.search(regex_pattern, text):
            return False  # 正規表現に一致する場合はFalseを返す
    return True  # 正規表現に一致しない場合はTrueを返す

# 正規表現に一致しない行だけを抽出
filtered_df = df[df['Text'].apply(lambda x: filter_text(x, regex_list))]

# CSVファイルに保存（正規表現に一致しない行）
filtered_df.to_csv(csv_file_path_not_filtered, index=False)

# 正規表現に一致する行だけを抽出
filtered_df_inclusive = df[df['Text'].apply(lambda x: not filter_text(x, regex_list))]

# CSVファイルに保存（正規表現に一致する行）
filtered_df_inclusive.to_csv(csv_file_path_filtered, index=False)
