#!/usr/bin/env python3
import argparse
import os
import glob
import re
from yt_dlp import YoutubeDL

def download_subtitles(url):
    """下载YouTube视频的字幕"""
    # 设置下载选项
    ydl_opts = {
        'writeautomaticsub': True,       # 下载自动生成的字幕
        'skip_download': True,           # 跳过视频下载
        'subtitleslangs': ['en'],       # 下载英文字幕
        'cookies':'cookies.txt',        #使用cookies
        'quiet': True,                   # 减少控制台输出
        'outtmpl': 'subtitles/%(title)s.%(ext)s'  # 字幕输出路径模板
    }

    # 使用 yt-dlp 下载字幕
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    # 重命名vtt文件
    vtt_files = glob.glob('subtitles/*.vtt')
    
    if vtt_files:
        # 获取第一个vtt文件
        original_vtt_file = vtt_files[0]
        
        # 构建新的文件名
        new_vtt_file = 'subtitles/word_level.vtt'
        
        # 重命名文件
        os.rename(original_vtt_file, new_vtt_file)
        print(f"重命名 '{original_vtt_file}' 为 '{new_vtt_file}'")
        
        # 处理VTT文件，提取单词级别字幕并拼接成句子
        process_vtt_file(new_vtt_file)
        
        return new_vtt_file
    else:
        print("在 'subtitles' 文件夹中未找到 .vtt 文件。")
        return None

def process_vtt_file(vtt_file):
    """处理VTT文件，提取单词级别字幕并拼接成带时间戳的句子"""
    output_file = os.path.splitext(vtt_file)[0] + "_processed.txt"
    
    with open(vtt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 移除VTT文件头和样式信息
    content = re.sub(r'WEBVTT.*?\n\n', '', content, flags=re.DOTALL)
    content = re.sub(r'NOTE.*?\n\n', '', content, flags=re.DOTALL)
    content = re.sub(r'STYLE.*?\n\n', '', content, flags=re.DOTALL)
    
    # 分割字幕块
    blocks = re.split(r'\n\n+', content.strip())
    
    processed_lines = []
    
    for block in blocks:
        if not block.strip():
            continue
            
        # 提取时间戳和文本
        lines = block.split('\n')
        timestamp_line = None
        text_lines = []
        
        for line in lines:
            if '-->' in line:
                timestamp_line = line
            elif line.strip():
                text_lines.append(line.strip())
        
        if timestamp_line and text_lines:
            # 解析时间戳
            time_match = re.search(r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})', timestamp_line)
            if time_match:
                start_time = time_match.group(1)
                # 格式化时间戳为 (HH:MM:SS.mmm)
                formatted_time = f"({start_time})"
                
                # 合并文本行
                text = ' '.join(text_lines)
                
                # 移除HTML标签（如果有）
                text = re.sub(r'<[^>]+>', '', text)
                
                # 添加到处理后的行列表
                processed_lines.append(f"{formatted_time} {text}")
    
    # 写入处理后的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in processed_lines:
            f.write(line + '\n')
    
    print(f"处理后的字幕已保存到: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='下载YouTube视频字幕')
    parser.add_argument('--url', required=True, help='YouTube视频URL')
    args = parser.parse_args()
    
    print(f"开始下载YouTube字幕: {args.url}")
    result = download_subtitles(args.url)
    
    if result:
        print(f"字幕下载并处理成功: {result}")
    else:
        print("字幕下载失败")
        exit(1)

if __name__ == "__main__":
    main()