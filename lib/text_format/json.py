import json

def filter_jsonl(input_file, output_file, min_length):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            if len(data['本文']) > min_length:
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')

# 使用例
input_file = r"C:\Users\nider\Desktop\git\llm\text-generate\instruction_dataset.jsonl"
output_file = r"C:\Users\nider\Desktop\git\llm\text-generate\filtered_instruction_dataset.jsonl"
min_length = 2048  # 最小文字数を指定

filter_jsonl(input_file, output_file, min_length)