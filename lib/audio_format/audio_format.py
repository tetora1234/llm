import ffmpeg
import re
import subprocess

# 入力ファイルのパス
input_file = "C:\\Users\\nider\\Desktop\\git\\lib\\audio_format\\video.mp4"

# 無音検出を実行し、ログを取得
def detect_silence(input_file):
    command = [
        "ffmpeg", "-i", input_file, "-af", "silencedetect=noise=-30dB:d=0.5",
        "-f", "null", "-"
    ]
    result = subprocess.run(command, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    return result.stderr

# 無音区間の情報を解析
def parse_silence_log(log):
    silence_starts = [float(x) for x in re.findall(r"silence_start: (\d+\.\d+)", log)]
    silence_ends = [float(x) for x in re.findall(r"silence_end: (\d+\.\d+)", log)]
    return list(zip(silence_starts, silence_ends))

# 動画の長さを取得
def get_video_duration(input_file):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", input_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8'
    )
    return float(result.stdout)

# 動画を無音区間で分割
def split_video(input_file, silence_intervals, video_duration):
    previous_end = 0
    part_number = 1

    for start, end in silence_intervals:
        if previous_end < start:
            output_file = f"output_part_{part_number}.mp4"
            (
                ffmpeg
                .input(input_file, ss=previous_end, to=start)
                .output(output_file, c='copy')
                .run(overwrite_output=True)
            )
            part_number += 1
        previous_end = end

    # 最後の区間を抽出
    if previous_end < video_duration:
        output_file = f"output_part_{part_number}.mp4"
        (
            ffmpeg
            .input(input_file, ss=previous_end, to=video_duration)
            .output(output_file, c='copy')
            .run(overwrite_output=True)
        )

# メイン処理
log = detect_silence(input_file)
silence_intervals = parse_silence_log(log)
video_duration = get_video_duration(input_file)
split_video(input_file, silence_intervals, video_duration)
