# -*- coding: utf-8 -*-
import os
import re
import asyncio
# import tkinter as tk
# from tkinter import filedialog, ttk
import edge_tts
from pydub import AudioSegment
import shutil
import random
from edge_tts import VoicesManager
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib

def check_ffmpeg():
    """检查ffmpeg是否可用"""
    if not shutil.which('ffmpeg'):
        print("错误：未找到ffmpeg。请确保已安装ffmpeg并添加到系统环境变量中。")
        print("您可以从 https://ffmpeg.org/download.html 下载ffmpeg，或使用包管理器安装。")
        return False
    return True

def ffprobe_duration(path):
    """用 ffprobe 取文件的容器声明时长（秒），失败返回 None。"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception as e:
        print(f"  [ffprobe] 取时长失败 {path}: {e}", flush=True)
    return None


def ffprobe_streams(path):
    """列出每个流的类型、时长、码率，方便定位异常流。"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries',
             'stream=index,codec_type,duration,bit_rate', '-of', 'csv=p=0', path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                print(f"    {line}", flush=True)
        else:
            print(f"    (ffprobe 无输出)", flush=True)
    except Exception as e:
        print(f"    ffprobe 失败: {e}", flush=True)


async def text_to_speech(text, output_file, voice="zh-CN-XiaoxiaoNeural", max_retries=5):
    """
    将文本转换为语音并保存为音频文件
    添加重试机制和延迟，处理edge-tts API的503错误
    """
    retry_count = 0
    base_delay = 1  # 基础延迟时间（秒）
    while retry_count <= max_retries:
        try:
            # 添加随机延迟，避免请求过于规律
            if (retry_count > 0):
                delay = base_delay * (2 ** (retry_count - 1)) + (random.random() * 0.5)
                print(f"第{retry_count}次重试，等待{delay:.2f}秒后继续...")
                await asyncio.sleep(delay)
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_file)
            return  # 成功则退出循环
        except Exception as e:
            error_msg = str(e).lower()
            retry_count += 1
            # 检查是否是503错误或其他可重试的错误
            if "503" in error_msg or "timeout" in error_msg or "connection" in error_msg:
                if retry_count <= max_retries:
                    print(f"遇到API错误: {e}，准备第{retry_count}次重试...")
                else:
                    print(f"达到最大重试次数({max_retries})，无法完成转换: {e}")
                    raise  # 达到最大重试次数，抛出异常
            else:
                # 其他类型的错误直接抛出
                print(f"遇到非重试类型的错误: {e}")
                raise

def run_text_to_speech(text, output_file, voice="zh-CN-XiaoxiaoNeural", max_retries=5):
    """
    在多进程中运行text_to_speech的包装函数
    """
    # 创建新的事件循环并在其中运行异步函数
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(text_to_speech(text, output_file, voice, max_retries))
    finally:
        loop.close()

def process_text_segment(task):
    """
    处理单个文本段落的函数，用于多进程处理
    """
    i, timestamp, content, temp_dir, voice = task
    try:
        # Use a cleaned version of timestamp for filename, removing potential special characters
        cleaned_timestamp = re.sub(r'[^\w\d]+', '_', timestamp)
        file_name = f"{cleaned_timestamp}.mp3"
        output_file = os.path.join(temp_dir, file_name)

        print(f"进程正在处理段落 {i+1}: {timestamp} - {content[:30]}...")
        run_text_to_speech(content, output_file, voice=voice)

        time_ms = parse_timestamp(f"({timestamp})")
        return i, output_file, time_ms, None
    except Exception as e:
        return i, None, None, f"处理段落 {i+1} 时出错: {str(e)}"

def parse_timestamp(timestamp):
    """
    将时间戳字符串转换为毫秒，支持 (h:mm:ss), (hh:mm:ss), (mm:ss), (h:mm:ss.ms), (hh:mm:ss.ms), (mm:ss.ms) 格式
    现在也支持三位数的分钟，例如 (123:34.56)
    """
    # Updated regex to correctly capture optional milliseconds with dot
    match = re.match(r'[\(（](?:(\d{1,2}):)?(\d{1,3}):(\d{1,2})(?:\.(\d{1,3}))?[\)）]', timestamp)
    if match:
        hours, minutes, seconds, milliseconds = match.groups()
        total_ms = 0
        if hours:
            total_ms += int(hours) * 3600 * 1000
        total_ms += int(minutes) * 60 * 1000
        total_ms += int(seconds) * 1000
        if milliseconds:
            total_ms += int(milliseconds.ljust(3, '0'))
        return total_ms
    return 0

def parse_vtt_timestamp(timestamp):
    """将VTT格式的时间戳转换为毫秒"""
    # 移除可能的BOM和空白字符
    timestamp = timestamp.strip().lstrip('\ufeff')
    # 尝试匹配 HH:MM:SS.mmm 格式
    match = re.match(r'(\d{2}):(\d{2}):(\d{2})\.(\d{3})', timestamp)
    if match:
        hours, minutes, seconds, milliseconds = map(int, match.groups())
        return hours * 3600000 + minutes * 60000 + seconds * 1000 + milliseconds
    # 尝试匹配 MM:SS.mmm 格式
    match = re.match(r'(\d{2}):(\d{2})\.(\d{3})', timestamp)
    if match:
        minutes, seconds, milliseconds = map(int, match.groups())
        return minutes * 60000 + seconds * 1000 + milliseconds
    return 0

def split_vtt_file(vtt_file):
    """处理VTT文件，返回时间戳和文本内容"""
    segments = []
    current_segment = None
    with open(vtt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 跳过WEBVTT头部
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip() == 'WEBVTT':
            start_idx = i + 1
            break
    lines = lines[start_idx:]
    for line in lines:
        line = line.strip()
        # 跳过空行和数字行（字幕序号）
        if not line or line.isdigit():
            continue
        # 检查是否是时间戳行
        if ' --> ' in line:
            if current_segment:
                segments.append(current_segment)
            start_time, end_time = line.split(' --> ')
            start_ms = parse_vtt_timestamp(start_time)
            end_ms = parse_vtt_timestamp(end_time)
            current_segment = [start_time, '', start_ms, end_ms]
        elif current_segment is not None:
            # 添加文本内容
            if current_segment[1]:
                current_segment[1] += ' '
            current_segment[1] += line
    # 添加最后一个片段
    if current_segment:
        segments.append(current_segment)
    return segments

def split_text_by_timestamp(text):
    """
    按时间戳分割文本，返回时间戳和对应的文本内容
    支持 (h:mm:ss), (hh:mm:ss), (mm:ss), (h:mm:ss.ms), (hh:mm:ss.ms), (mm:ss.ms) 格式
    现在也支持三位数的分钟，例如 (123:34.56)
    """
    # Updated pattern to match timestamps with optional hours and optional milliseconds
    # It now correctly captures the timestamp string and the content following it
    pattern = r'[\(（](\d{1,2})?:?(\d{1,3}):(\d{1,2})(?:\.(\d{1,3}))?[\)）](.+?)(?=[\(（](?:\d{1,2})?:?(\d{1,3}):(\d{1,2})(?:\.(\d{1,3}))?[\)）]|$)'
    segments = []
    matches = re.finditer(pattern, text, re.DOTALL)
    for match in matches:
        # The full matched timestamp string is group(0)
        timestamp_string = match.group(0)
        # The content is group(5) in the updated pattern
        content = match.group(5).strip()
        if content:  # Only add non-empty text content
            # Extract just the timestamp part for parsing
            timestamp_for_parsing = re.match(r'[\(（](.+?)[\)）]', timestamp_string).group(1)
            segments.append((timestamp_for_parsing, content))

    return segments

def adjust_audio_speed(task):
    """
    调整音频速度的函数，用于多进程处理
    使用低通滤波器去除调速时产生的刺耳尖啸声
    """
    i, temp_output, target_duration, speed_factor = task
    temp_output_processed = temp_output + '.processed.mp3'
    try:
        print(f"进程正在调整音频 {i+1} 的速度，原始长度 {target_duration/speed_factor:.0f}ms，目标长度 {target_duration}ms，因子 {speed_factor:.2f}")
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', temp_output, '-filter:a', f'lowpass=f=8000,atempo={speed_factor}',
             temp_output_processed],
            capture_output=True,
            text=True,
            timeout=60  # Increased timeout for ffmpeg
        )
        if result.returncode == 0:
            return i, temp_output_processed, None
        else:
            return i, None, f"ffmpeg processing failed for audio {i+1}: {result.stderr}"
    except Exception as e:
        return i, None, f"Error during ffmpeg processing for audio {i+1}: {str(e)}"

"""
原函数：process_text_file
优化点：
1. 用 numpy 共享内存一次性混音，替代循环 final_audio.overlay(...)
2. 只在最后 export 一次，省去反复内存拷贝
3. 其余逻辑（TTS、变速、清理）保持完全一致
"""

import os, math, numpy as np, multiprocessing as mp
from multiprocessing import shared_memory
#from concurrent.futures import ProcessPoolExecutor, as_completed
#from pydub import AudioSegment
from pydub.utils import get_array_type

# ---------- 混音参数 ---------- #
SR = 24_000          # 统一采样率
N_CH = 1             # 单声道
WIDTH = 2            # 16-bit
MAX_INT = 2**(8*WIDTH-1) - 1

# ---------- 小工具 ---------- #
def read_header(path):
    """只读头，返回 (samples, sr)"""
    seg = AudioSegment.from_file(path)
    return int(seg.frame_count()), seg.frame_rate

def to_int16_samples(audio: AudioSegment):
    """把 AudioSegment 转成 int16 numpy 数组"""
    audio = (audio.set_frame_rate(SR)
                  .set_channels(N_CH)
                  .set_sample_width(WIDTH))
    return np.frombuffer(audio.raw_data, dtype=np.int16)

# ---------- 一次性混音 ---------- #
def fast_overlay(audio_files, processed_audio_segments):
    """
    audio_files: [(path, start_ms), ...]   已排序
    processed_audio_segments: 与原逻辑完全一致，可能含变速后 AudioSegment
    返回：混音后的 AudioSegment
    """
    # 1. 计算总长度
    last_path, last_ms = audio_files[-1]
    last_len = len(processed_audio_segments[-1][2])
    total_ms = last_ms + last_len + 1000   # 留 1 s 尾巴
    total_samples = int(total_ms * SR / 1000)

    # 2. 共享内存混音板
    shm = shared_memory.SharedMemory(create=True, size=total_samples * N_CH * 4)
    buf = np.ndarray((total_samples * N_CH,), dtype=np.float32, buffer=shm.buf)
    buf[:] = 0.0

    # 3. 主进程内逐段叠加（纯内存，无 Python 循环 overlay）
    for (path, start_ms), (_, _, audio) in zip(audio_files, processed_audio_segments):
        samples = to_int16_samples(audio).astype(np.float32)
        start_sample = int(start_ms * SR / 1000)
        end_sample   = start_sample + len(samples)
        buf[start_sample:end_sample] += samples

    # 4. clip + 转回 int16
    np.clip(buf, -MAX_INT, MAX_INT, out=buf)
    out_bytes = buf.astype(np.int16).tobytes()
    shm.close()
    shm.unlink()

    return AudioSegment(
        data=out_bytes,
        sample_width=WIDTH,
        frame_rate=SR,
        channels=N_CH
    )

# ---------- 主流程（仅替换混音部分） ---------- #
async def process_text_file(file_path, voice="zh-CN-XiaoxiaoNeural"):
    try:
        temp_dir = os.path.join(os.path.dirname(file_path), "temp_audio")
        os.makedirs(temp_dir, exist_ok=True)

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        segments = split_text_by_timestamp(content)

        # 1. 多进程 TTS（完全不变）
        tasks = [(i, timestamp, txt, temp_dir, voice)
                 for i, (timestamp, txt) in enumerate(segments)]
        audio_files = [None] * len(segments)
        print(f"开始使用4个进程处理 {len(segments)} 个文本段落...")
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_text_segment, task) for task in tasks]
            for future in as_completed(futures):
                i, output_file, time_ms, error = future.result()
                if error or not output_file or not os.path.exists(output_file):
                    print(error or f"段落 {i+1} 未生成文件")
                    continue
                audio_files[i] = (output_file, time_ms)
                print(f"段落 {i+1} 处理完成")

        audio_files = [af for af in audio_files if af is not None]
        if not audio_files:
            print("错误：没有成功生成任何音频文件")
            return
        audio_files.sort(key=lambda x: x[1])

        # 2. 速度调整（完全不变）
        speed_adjust_tasks_list = []
        processed_audio_segments = []
        for i, (audio_file_path, time_ms) in enumerate(audio_files):
            audio = AudioSegment.from_file(audio_file_path)
            processed_audio_segments.append((audio_file_path, time_ms, audio))
            end_time = time_ms + len(audio)
            if i < len(audio_files) - 1:
                next_start = audio_files[i+1][1]
                if end_time > next_start + 100:
                    target = next_start - time_ms - 50
                    if target > 100:
                        factor = min(len(audio) / target, 2.0)
                        tmp = os.path.join(temp_dir, f"speed_{i}.mp3")
                        audio.export(tmp, format="mp3")
                        speed_adjust_tasks_list.append((i, tmp, target, factor))

        if speed_adjust_tasks_list:
            print(f"开始使用8个进程处理 {len(speed_adjust_tasks_list)} 个音频速度调整任务...")
            with ProcessPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(adjust_audio_speed, task) for task in speed_adjust_tasks_list]
                for future in as_completed(futures):
                    i, processed_file, error = future.result()
                    if error or not processed_file or not os.path.exists(processed_file):
                        print(error or f"索引 {i} 变速失败，保留原音频")
                        continue
                    _, time_ms, _ = processed_audio_segments[i]
                    processed_audio = AudioSegment.from_file(processed_file)
                    processed_audio_segments[i] = (processed_file, time_ms, processed_audio)
                    # 清理中间临时文件
                    orig_tmp = next((t[1] for t in speed_adjust_tasks_list if t[0] == i), None)
                    if orig_tmp and os.path.exists(orig_tmp):
                        os.remove(orig_tmp)
                    print(f"音频 {i+1} 速度调整成功，新长度 {len(processed_audio)}ms")

        # 3. 关键：一次性混音（替换掉原来的循环 overlay）
        print("开始一次性混音...")
        # 调试：打印最后一段的位置与长度，用于核对总时长计算
        if audio_files:
            last_path, last_ms = audio_files[-1]
            last_audio = processed_audio_segments[-1][2]
            expected_total_ms = last_ms + len(last_audio) + 1000
            print(f"  [调试] 段落总数: {len(audio_files)}", flush=True)
            print(f"  [调试] 最后段开始时间戳: {last_ms} ms ({last_ms/1000:.2f} s)", flush=True)
            print(f"  [调试] 最后段音频长度: {len(last_audio)} ms ({len(last_audio)/1000:.2f} s)", flush=True)
            print(f"  [调试] 预期音频总时长: {expected_total_ms} ms ({expected_total_ms/1000:.2f} s, {expected_total_ms/3600000:.2f} h)", flush=True)
            # 抽样输出最长/最短的 3 段
            lengths = [(i, len(processed_audio_segments[i][2])) for i in range(len(processed_audio_segments))]
            lengths.sort(key=lambda x: x[1], reverse=True)
            print(f"  [调试] 最长 3 段 (索引, ms): {lengths[:3]}", flush=True)
            print(f"  [调试] 最短 3 段 (索引, ms): {lengths[-3:]}", flush=True)
        final_audio = fast_overlay(audio_files, processed_audio_segments)

        # 4. 导出 & 清理（完全不变）
        output_file = os.path.splitext(file_path)[0] + ".mp3"
        print(f"正在保存音频文件至: {output_file}")
        final_audio.export(output_file, format="mp3")
        print(f"音频已成功保存至: {output_file}")

        # 调试：核对实际生成的 MP3 时长
        actual_dur = ffprobe_duration(output_file)
        if actual_dur is not None:
            print(f"  [调试] MP3 实际时长: {actual_dur:.2f} s ({actual_dur/3600:.2f} h)", flush=True)
            print(f"  [调试] MP3 字节数: {os.path.getsize(output_file)}", flush=True)
            if actual_dur > 36000:  # > 10h
                print(f"  [告警] MP3 时长超过 10 小时！", flush=True)

        # 清理临时文件
        for fp, _ in audio_files:
            if os.path.exists(fp):
                os.remove(fp)
        for fp, _, _ in processed_audio_segments:
            if fp.endswith('.processed.mp3') and os.path.exists(fp):
                os.remove(fp)
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        import traceback
        traceback.print_exc()

async def get_available_voices():
    """获取可用的语音列表"""
    voices = await VoicesManager.create()
    chinese_voices = [v for v in voices.voices if v["Locale"].startswith("zh")]
    return chinese_voices

def process_from_args():
    """处理命令行参数并执行TTS"""
    if not check_ffmpeg():
        return
    import sys
    input_file = None
    selected_voice = None
    # Get file path and voice from command line arguments
    if "--filepath" in sys.argv:
        try:
            filepath_index = sys.argv.index("--filepath")
            input_file = sys.argv[filepath_index + 1]
        except (ValueError, IndexError):
            print("Error: --filepath requires a file path.")
            return
    if "--char" in sys.argv:
        try:
            char_index = sys.argv.index("--char")
            selected_voice = sys.argv[char_index + 1]
        except (ValueError, IndexError):
            print("Error: --char requires a voice name.")
            return
    if input_file and selected_voice:
        print(f"正在处理文件: {input_file} 使用朗读角色: {selected_voice}")
        asyncio.run(process_text_file(input_file, voice=selected_voice))
    else:
        print("错误：未提供输入文件路径或朗读角色。请使用 --filepath <file path> 和 --char <voice name> 参数指定。")

if __name__ == "__main__":
    # Use the new function to process arguments
    process_from_args()
