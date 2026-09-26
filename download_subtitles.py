#!/usr/bin/env python3
import argparse
import os
import glob
import re
import shutil
import sys
import time
import subprocess
from yt_dlp import YoutubeDL

# ========== 可调参数 ==========
DEBUG = True          # 打开详细调试输出
SENTENCE_END = ".!?"    # 只以句号结尾分句（按需求）。如需 ?! 一起分句，可改为 ".!?"
# =============================

# ---- 缩写识别：末尾句点不应触发断句 ----
# 1) 首字母缩写型：U.S.A. / L.I. / U.S. / A.B.C.（单大写字母+句点重复，整词无小写字母）
_INITIALS_RE = re.compile(r'^[A-Z]\.([A-Z]\.)*$')

# 2) 常见缩写词（小写形式，含句点）
_ABBREV_SET = {
    # 称谓
    'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sr.', 'jr.', 'st.',
    # 常见拉丁/英文缩写
    'vs.', 'etc.', 'e.g.', 'i.e.', 'inc.', 'ltd.', 'co.', 'no.',
    'corp.', 'bros.', 'fig.', 'vol.', 'pp.', 'ed.', 'rev.', 'al.',
    'approx.', 'dept.', 'est.', 'ph.d.', 'b.a.', 'm.a.',
    # 月份
    'jan.', 'feb.', 'mar.', 'apr.', 'jun.', 'jul.', 'aug.',
    'sept.', 'oct.', 'nov.', 'dec.',
    # 时间
    'a.m.', 'p.m.',
}


def _is_abbreviation(word: str) -> bool:
    """判断 word 末尾的句点是否属于缩写（不应触发断句）。"""
    w = word.strip()
    if not w:
        return False
    if _INITIALS_RE.match(w):        # L.I. / U.S.A. / U.S.
        return True
    if w.lower() in _ABBREV_SET:     # Mr. / Dr. / e.g. / a.m.
        return True
    return False


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
            # 缩写词末尾的句点（如 L.I. / U.S.A. / Mr. / Dr. / e.g.）不触发断句
            if SENTENCE_END and word.strip().endswith(tuple(SENTENCE_END)):
                if not _is_abbreviation(word):
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


# ========== 字幕语言候选与重试参数 ==========
# 为什么必须优先用 "<lang>-orig"（原生 ASR 轨）而不是 "en"：
#   1) yt-dlp 的 --sub-format 默认是 'best'，其实现为 `formats[-1]`，即取该语言
#      列表里的【最后一条】。而 "en" 这个 key 会把原生 ASR 轨和大量机器翻译轨
#      （tlang=en，lang=xx_XX，variant=timing-optimized）都收进来，最后一条常常
#      正是翻译轨。YouTube 对「翻译字幕」接口限流极严，会直接返回 429。
#   2) "en-orig" 只对应唯一的原生英文 ASR 轨（kind=asr, lang=en, 无 tlang），
#      稳定可下载，并且带逐词时间戳 <HH:MM:SS.mmm>，正是本项目分句所需的格式。
# 参考：https://github.com/yt-dlp/yt-dlp/issues/13831
SUB_LANG_CANDIDATES = ('en-orig', 'en')

# 429/网络抖动重试：每个候选语言最多尝试次数与首次退避秒数（指数退避）
SUB_RETRY_ATTEMPTS = 3
SUB_RETRY_BASE_DELAY = 15
# ===========================================


def _clean_vtt_dir():
    """清掉 subtitles/ 下遗留的 .vtt，避免上一次尝试的文件被 glob 误取。"""
    for f in glob.glob('subtitles/*.vtt'):
        try:
            os.remove(f)
        except OSError:
            pass


def _ytdlp_prefix():
    """返回调用 yt-dlp 的命令前缀。

    优先用 PATH 上的 yt-dlp 可执行文件；找不到时回退到 `python -m yt_dlp`。
    本文件顶部已经 `from yt_dlp import YoutubeDL`，因此模块一定可用，
    回退路径在任何环境下都能工作（也避免 CI 里 PATH 没配好导致 WinError 2）。
    """
    exe = shutil.which('yt-dlp')
    if exe:
        return [exe]
    return [sys.executable, '-m', 'yt_dlp']


def _build_subtitle_cmd(url, sub_lang, cookies_file=None):
    """构造单次 yt-dlp 字幕下载命令。"""
    # 注意：cookies 必须插在「完整前缀」之后，不能写死 cmd[1:1]，
    # 否则前缀为 [python, -m, yt_dlp] 时会被插到 python 和 -m 之间。
    cmd = _ytdlp_prefix()
    if cookies_file and os.path.exists(cookies_file):
        cmd += ['--cookies', cookies_file]
    cmd += [
        '--extractor-args', 'youtube:player_client=default,-web_safari',
        # Explicitly point to the Deno path we set up
        '--extractor-args', 'youtube:js_runtime=/usr/local/bin/deno',
        '--remote-components', 'ejs:github',
        '--no-playlist',
        '--write-auto-sub',
        '--skip-download',
        '--sub-langs', sub_lang,
        '-o', 'subtitles/%(title)s.%(ext)s',
        url
    ]
    return cmd


def _pick_vtt_file(sub_lang):
    """从 subtitles/ 中挑出本次下载的 vtt，避免多文件时 glob[0] 取错。"""
    files = glob.glob('subtitles/*.vtt')
    if not files:
        return None
    exact = [f for f in files if f.endswith(f'.{sub_lang}.vtt')]
    if exact:
        return exact[0]
    partial = [f for f in files if f'.{sub_lang}.' in os.path.basename(f)]
    if partial:
        return partial[0]
    return files[0]


def _finalize_vtt(original_vtt_file):
    """把下载到的 vtt 统一改名为 subtitles/word_level.vtt 并解析成句子。

    解析不出任何句子（例如拿到的是非逐词时间戳字幕）时返回 None，
    以便外层继续尝试下一个候选语言。
    """
    new_vtt_file = 'subtitles/word_level.vtt'
    if os.path.abspath(original_vtt_file) != os.path.abspath(new_vtt_file):
        if os.path.exists(new_vtt_file):
            os.remove(new_vtt_file)
        os.rename(original_vtt_file, new_vtt_file)
        print(f"重命名 '{original_vtt_file}' 为 '{new_vtt_file}'")
    # 清掉同批次残留的其他 vtt，避免污染后续 glob
    for f in glob.glob('subtitles/*.vtt'):
        if os.path.abspath(f) != os.path.abspath(new_vtt_file):
            os.remove(f)

    processed_file = process_vtt_file(new_vtt_file)

    with open(processed_file, 'r', encoding='utf-8') as f:
        sentence_count = sum(1 for line in f if line.strip())
    if sentence_count == 0:
        print(f"警告: {new_vtt_file} 未解析出任何句子（可能不是逐词时间戳字幕），视为失败")
        return None

    return new_vtt_file


def download_subtitles(url, cookies_file=None):
    """下载YouTube视频的字幕。

    按 SUB_LANG_CANDIDATES 顺序尝试候选语言，每个候选遇到 429/Too Many Requests
    时做指数退避重试；全部失败则返回 None。
    """
    if cookies_file and os.path.exists(cookies_file):
        print(f"使用cookies文件: {cookies_file}")
    else:
        print("未提供cookies文件或文件不存在，尝试无cookies下载")

    for sub_lang in SUB_LANG_CANDIDATES:
        cmd = _build_subtitle_cmd(url, sub_lang, cookies_file)

        for attempt in range(1, SUB_RETRY_ATTEMPTS + 1):
            if attempt > 1:
                delay = SUB_RETRY_BASE_DELAY * (2 ** (attempt - 2))
                print(f"[重试] 语言 {sub_lang!r} 第 {attempt}/{SUB_RETRY_ATTEMPTS} 次，等待 {delay}s ...")
                time.sleep(delay)

            _clean_vtt_dir()
            print(f"执行命令: {' '.join(cmd)}")
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True,
                    encoding='utf-8', errors='ignore'
                )
            except Exception as e:
                # 这类错误（找不到可执行文件等）重试也没用，直接换下一个候选语言
                print(f"下载失败（无法执行 yt-dlp）: {e}")
                break

            if result.returncode == 0:
                vtt_file = _pick_vtt_file(sub_lang)
                if not vtt_file:
                    print(f"命令成功但未找到 .vtt 文件（sub_lang={sub_lang!r}）")
                    break
                finalized = _finalize_vtt(vtt_file)
                if finalized:
                    print(f"使用候选语言 {sub_lang!r} 成功")
                    return finalized
                # 拿到了文件但解析不出句子，换下一个候选语言
                break

            err = (result.stderr or '').strip()
            print(f"下载失败 (sub_lang={sub_lang!r}, 第 {attempt}/{SUB_RETRY_ATTEMPTS} 次): {err}")
            # 只有限流类错误才值得重试；其他错误直接换下一个候选语言
            if '429' not in err and 'Too Many Requests' not in err:
                break

        print(f"候选语言 {sub_lang!r} 未能获取可用字幕，尝试下一个候选")

    print("所有候选语言均下载失败。")
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
    print("\n========== 完整原文字幕 ==========")
    print(processed_content)
    print("==================================\n")
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