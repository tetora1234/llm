import os
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
# 変換したいフォルダのパスを指定
root_folder = r"C:\Users\nider\Desktop\git\llm\data\audio"

def convert_ogg_to_wav(ogg_path):
    try:
        # wavファイルのパス
        wav_path = os.path.splitext(ogg_path)[0] + '.wav'
        
        # oggファイルを読み込み
        audio = AudioSegment.from_ogg(ogg_path)
        # wavファイルに変換して保存
        audio.export(wav_path, format='wav')
        
        # 変換が成功したら元のoggファイルを削除
        if os.path.exists(wav_path):
            os.remove(ogg_path)
            print(f'Converted and deleted: {ogg_path} to {wav_path}')
        else:
            print(f'Failed to convert: {ogg_path}')
    except Exception as e:
        print(f'Error converting {ogg_path}: {e}')

def convert_ogg_to_wav_multithreaded(root_folder, max_workers=16):
    # サブフォルダを含めてすべてのファイルを探索
    ogg_files = []
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.ogg'):
                ogg_path = os.path.join(foldername, filename)
                ogg_files.append(ogg_path)

    # マルチスレッドで変換を実行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(convert_ogg_to_wav, ogg_files)

convert_ogg_to_wav_multithreaded(root_folder)
