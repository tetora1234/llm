import csv
import shutil

# CSVファイルと移動先のフォルダのパスを指定
csv_file_path = r'C:\Users\nider\Desktop\git\lib\text_format\output_not_filtered.csv'
move_to_folder = r'C:\Users\nider\Desktop\git\lib\text_format\move'

# CSVファイルからファイルパスを読み取り、移動する
with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        file_path = row[0]
        try:
            shutil.move(file_path, move_to_folder)
            print(f"{file_path} を {move_to_folder} に移動しました。")
        except Exception as e:
            print(f"エラー: {e}")
