#!/usr/bin/env python3
import argparse
import os
import glob
import re
from yt_dlp import YoutubeDL

# ========== 可调参数 ==========
DEBUG = True          # 打开详细调试输出
SENTENCE_END = ".!?"    # 只以句号结尾分句（按需求）。如需 ?! 一起分句，可改为 ".!?"
# =============================

# 正则：cue 头（起止时间）
CUE_HEADER_RE = re.compile(
    r'^(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})'
)

# 正则：逐词时间戳 <HH:MM:SS.mmm>
TS_TAG_RE = re.compile(r'<(\d{2}:\d{2}:\d{2}\.\d{3})>')

# 正则：清理 <c> 或 <c.xxx> 样式标签
C_TAG_RE = re.compile(r'</?c(?:\.[^>]*)?>', re.IGNORECASE)


def vtt_to_sentences(vtt_text: str, debug: bool = False) -> str:
    """
    将带逐词时间戳的 YouTube WebVTT 字幕转换为按句（以句号、问号、叹号结尾）文本，句首带起始时间。
    解析策略：
      1) 识别 cue 头，保存该 cue 的起始时间（作为该块默认时间）。
      2) 仅处理包含 <HH:MM:SS.mmm> 的行，避免重复的纯文本行。
      3) 行内将 <timestamp> 替换为哨兵 [[TS:...]] 并移除 <c> 标签；
         扫描 token，遇到 [[TS:...]] 更新"有效时间"；普通词使用当前有效时间。
      4) 累词到遇到以 '.' '!' 或 '?' 结尾的词即成句，句时间=本句第一词的时间。
    """
    lines = vtt_text.splitlines()
    num_lines = len(lines)
    num_cues = 0
    num_lines_with_ts = 0
    num_ts_tags = 0
    num_words = 0

    sentences = []
    current_words = []
    current_sentence_start_time = None

    effective_time = None        # 当前有效词时间
    cue_start_time = None        # 当前 cue 起始时间（备用）

    def flush_sentence():
        nonlocal current_words, current_sentence_start_time
        if not current_words:
            return
        # 组合文本并清理标点前空格
        text = " ".join(current_words)
        text = re.sub(r"\s+([,.;!?])", r"\1", text)   # 去掉标点前多余空格
        text = re.sub(r"\(\s+", "(", text)            # 括号内多余空格
        text = re.sub(r"\s+\)", ")", text)
        start_ts = current_sentence_start_time or cue_start_time or effective_time or "00:00:00.000"
        sentences.append(f"({start_ts}) {text}")
        current_words = []
        current_sentence_start_time = None

    for i, raw_line in enumerate(lines):
        line = raw_line.strip("\ufeff\r\n")

        # cue 头
        m = CUE_HEADER_RE.match(line)
        if m:
            num_cues += 1
            cue_start_time = m.group(1)
            effective_time = cue_start_time  # 初始有效时间设为 cue 起始
            if debug:
                print(f"[cue] line {i+1}: start={cue_start_time}, end={m.group(2)}")
            continue

        # 只处理含逐词时间戳的行
        if not TS_TAG_RE.search(line):
            continue

        num_lines_with_ts += 1

        # 清理 <c> 标签，并把 <timestamp> 变成 [[TS:...]] 哨兵
        s = C_TAG_RE.sub("", line)
        # 统计本行的 timestamp 个数
        ts_in_line = TS_TAG_RE.findall(s)
        num_ts_tags += len(ts_in_line)
        s = TS_TAG_RE.sub(lambda mm: f" [[TS:{mm.group(1)}]] ", s)

        # 发送给模型时保留所有内容，包括非文本标签和时间戳
        # 但在处理逐词时间戳时，只关心时间和词本身
        # 扫描 token
        for token in s.split():
            if token.startswith("[[TS:") and token.endswith("]]"):
                effective_time = token[5:-2]  # 取出 HH:MM:SS.mmm
                continue

            word = token.strip()
            if not word:
                continue

            # 记录首词时间
            if current_sentence_start_time is None:
                current_sentence_start_time = effective_time or cue_start_time

            current_words.append(word)
            num_words += 1

            # 句子结束判定（句号、问号、叹号）
            if SENTENCE_END and word.strip().endswith(tuple(SENTENCE_END)):
                flush_sentence()

    # 文件结束，收尾
    flush_sentence()

    if debug:
        print("========== DEBUG SUMMARY ==========")
        print(f"Total lines           : {num_lines}")
        print(f"Cue headers found     : {num_cues}")
        print(f"Lines with <timestamp>: {num_lines_with_ts}")
        print(f"Timestamp tags found  : {num_ts_tags}")
        print(f"Words collected       : {num_words}")
        print(f"Sentences assembled   : {len(sentences)}")
        if num_ts_tags == 0:
            print("\n[Hint] 没发现逐词 <timestamp> 标签：")
            print("  - 该 VTT 可能不是逐词时间戳版本（只有普通句级字幕）；")
            print("  - 或时间戳格式与 <HH:MM:SS.mmm> 不一致；")
            print("  - 可把 DEBUG=True 并打印前几百字符手动检查。")
        print("===================================\n")

    return "\n".join(sentences)


def download_subtitles(url, cookies_file=None):
    """下载YouTube视频的字幕"""
    # 设置下载选项
    ydl_opts = {
        'writeautomaticsub': True,       # 下载自动生成的字幕
        'skip_download': True,           # 跳过视频下载
        'subtitleslangs': ['en'],       # 下载英文字幕
        'quiet': True,                   # 减少控制台输出
        'outtmpl': 'subtitles/%(title)s.%(ext)s',  # 字幕输出路径模板
        # 添加模拟真实浏览器的选项
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'no_check_certificate': True,    # 跳过证书验证
    }
    
    # 如果提供了cookies文件，添加到选项中
    if cookies_file and os.path.exists(cookies_file):
        ydl_opts['cookiefile'] = cookies_file
        print(f"使用cookies文件: {cookies_file}")
    else:
        print("未提供cookies文件或文件不存在，尝试无cookies下载")

    # 使用 yt-dlp 下载字幕
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"下载失败: {e}")
        return None
    
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
        vtt_content = f.read()
    
    # 使用新的vtt_to_sentences函数处理VTT内容
    processed_content = vtt_to_sentences(vtt_content, debug=DEBUG)
    
    # 写入处理后的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(processed_content)
    
    print(f"处理后的字幕已保存到: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='下载YouTube视频字幕')
    parser.add_argument('--url', required=True, help='YouTube视频URL')
    parser.add_argument('--cookies', help='YouTube cookies文件路径（可选）')
    args = parser.parse_args()
    
    print(f"开始下载YouTube字幕: {args.url}")
    result = download_subtitles(args.url, args.cookies)
    
    if result:
        print(f"字幕下载并处理成功: {result}")
    else:
        print("字幕下载失败")
        exit(1)

if __name__ == "__main__":
    main()