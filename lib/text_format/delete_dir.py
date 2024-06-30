import os

def remove_empty_folders(path, remove_root=True):
    """
    再帰的に空のフォルダを削除します。

    Args:
        path (str): チェックするディレクトリのパス。
        remove_root (bool): ルートディレクトリ自体も空であれば削除するかどうか。
    """
    # フォルダの内容を再帰的にチェック
    for root, dirs, files in os.walk(path, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                os.rmdir(dir_path)
                print(f"Removed empty folder: {dir_path}")
            except OSError:
                # フォルダが空でない場合は削除できないのでスキップ
                pass
    
    # ルートディレクトリ自体も空であれば削除
    if remove_root:
        try:
            os.rmdir(path)
            print(f"Removed empty root folder: {path}")
        except OSError:
            # フォルダが空でない場合は削除できないのでスキップ
            pass

# 使用例
remove_empty_folders(r'C:\Users\nider\Desktop\git\llm\data\audio')
