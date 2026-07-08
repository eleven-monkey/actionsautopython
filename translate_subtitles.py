#!/usr/bin/env python3
import argparse
import os
import re
import time
import json
import sys
from random import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 定义系统提示词常量
SYSTEM_PROMPT = """# Role: 专业字幕翻译官

## 任务
将外文字幕逐行翻译为中文，严格保留每行的时间戳和行数。

## 示例

输入：
(00:00:01.000) Hello everyone, welcome to my channel.
(00:00:03.500) Today we're going to talk about AI.
(00:00:06.200) Let's get started.

输出：
(00:00:01.000) 大家好，欢迎来到我的频道。
(00:00:03.500) 今天我们聊聊人工智能。
(00:00:06.200) 我们开始吧。

---

## 规则
1. 时间戳必须原样保留，不得修改数字、格式或符号。
2. 只翻译时间戳之后的正文内容为中文。
3. 每行时间戳与原文一一对应，不得合并、拆分或调换顺序。
4. 准确传达原意，译文符合中文表达习惯，通顺自然。
5. 不要添加任何解释性文字、注释或说明。
6. 保持原文的语气风格（如风趣幽默、严肃中立等）。
"""

# 线程锁用于保护共享资源
progress_lock = threading.Lock()
completed_count = 0
total_count = 0

# ---- 格式校验相关常量（从视频克隆配音/src 引入并适配本项目） ----
# 严格行格式：(<ts>) 文本 （本项目中的字幕行通常以括号包裹的时间戳开头，不强制要求 [Speaker XX]）
# 为了兼容本项目实际格式（不强制 [Speaker XX]），行校验只要求以时间戳开头。
LINE_PATTERN = re.compile(r'^[\(（]\s*(\d{1,3}:\d{2}(:\d{2})?(\.\d{1,3})?)\s*[\)）]\s*(.+)$')
TS_PATTERN = re.compile(r'[\(（]\s*\d{1,3}:\d{2}(:\d{2})?(\.\d{1,3})?\s*[\)）]')


def load_api_config(config_file):
    """从JSON文件加载API配置"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 验证必要的配置项
        required_fields = ['url', 'model_name']
        for field in required_fields:
            if not config.get(field):
                raise ValueError(f"配置文件中缺少必要字段: {field}")

        return config
    except FileNotFoundError:
        print(f"错误: 找不到配置文件 {config_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"错误: 配置文件 {config_file} 格式不正确")
        sys.exit(1)
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)


def read_file(file_path):
    """读取文件内容并按行分割"""
    with open(file_path, 'r', encoding='utf-8') as file:
        paragraphs = [line.strip() for line in file if line.strip()]
    return paragraphs


def segment_paragraphs(paragraphs, segment_size=3, keep_timestamps=False):
    """将段落分组"""
    segments = []
    current_segment = []
    current_segment_char_count = 0
    max_segment_chars = 2000  # 定义每个段落的最大字符数

    # 更新时间戳模式以包含(00:00.16)格式
    timestamp_pattern = re.compile(r'[\(（]\d{1,3}:\d{2}(:\d{2})?(\.\d{1,3})?[\)）]')

    for paragraph in paragraphs:
        # 计算当前段落的字符数（如果不保留时间戳则不包括时间戳）
        paragraph_char_count = len(paragraph)
        if not keep_timestamps:
            paragraph_char_count = len(timestamp_pattern.sub('', paragraph))

        # 检查添加下一段是否会超过最大字符限制或段落大小
        if (len(current_segment) >= segment_size) or \
           (current_segment_char_count + paragraph_char_count > max_segment_chars and current_segment):
            segments.append("\n".join(current_segment))
            current_segment = [paragraph]
            current_segment_char_count = paragraph_char_count
        else:
            current_segment.append(paragraph)
            current_segment_char_count += paragraph_char_count

    # 添加最后一段
    if current_segment:
        segments.append("\n".join(current_segment))

    return segments


def contains_chinese(text):
    """检查文本是否包含任何中文字符（使用Unicode范围）"""
    chinese_pattern = re.compile(r'[\u4E00-\u9FFF\uF900-\uFAFF\u3400-\u4DBF]')
    return bool(chinese_pattern.search(text))


def _extract_text_after_ts(line):
    """从合规行 '(ts) text' 中提取 text 部分。"""
    m = LINE_PATTERN.match(line.strip())
    if m:
        return m.group(4)
    return ""


def _extract_ts(line):
    """从合规行 '(ts) text' 中提取 ts 字符串部分（不含括号）。"""
    m = LINE_PATTERN.match(line.strip())
    if m:
        return m.group(1)
    return None


def _extract_orig_ts_list(segment_text):
    """从原文段落提取所有时间戳字符串，按出现顺序。"""
    ts_list = []
    for line in segment_text.splitlines():
        line = line.strip()
        if not line:
            continue
        ts = _extract_ts(line)
        if ts is not None:
            ts_list.append(ts)
    return ts_list


def is_valid_translation_format(text, orig_ts_list=None):
    """校验翻译结果格式。

    基础校验：
      - 每行符合 (HH:MM:SS[.mmm]) 文本
      - 时间戳严格递增
    严格校验（当传入 orig_ts_list 时）：
      - 译文时间戳必须是 orig_ts_list 的子序列
      - 顺序与原文一致：译文前 k 行时间戳必须能在 orig_ts_list 中找到
        一段连续子序列与之对应（允许少 1 行合并断句）
      - 时间戳必须**逐字相同**（LLM 改数字是常见错误，必须拒掉）

    返回 (ok: bool, err: str, trans_ts_list: list[str] | None)。
    """
    if not text or not text.strip():
        return False, "文本为空", None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False, "没有有效行", None

    trans_ts_list = []
    prev_ts = None
    for i, line in enumerate(lines, 1):
        m = LINE_PATTERN.match(line)
        if not m:
            return False, f"第{i}行格式不正确: {line[:80]}", None
        ts = m.group(1)
        if prev_ts is not None and ts <= prev_ts:
            return False, f"第{i}行时间戳未递增: {ts} <= {prev_ts}", None
        trans_ts_list.append(ts)
        prev_ts = ts

    # 严格对照原文时间戳序列
    if orig_ts_list is not None:
        # 允许译文比原文少 1 行（LLM 合并断句）
        if len(trans_ts_list) > len(orig_ts_list):
            return False, (
                f"译文行数({len(trans_ts_list)})超过原文({len(orig_ts_list)})，"
                f"LLM 可能插入了伪造时间戳"
            ), trans_ts_list
        if len(trans_ts_list) < len(orig_ts_list) - 1:
            return False, (
                f"译文行数({len(trans_ts_list)})比原文({len(orig_ts_list)})少超过 1 行，"
                f"LLM 漏译了过多行"
            ), trans_ts_list
        # 译文时间戳必须 == orig_ts_list 前 k 项的逐字相同
        for i, tts in enumerate(trans_ts_list):
            if tts != orig_ts_list[i]:
                return False, (
                    f"第{i+1}行时间戳与原文不符: 译文={tts} != 原文={orig_ts_list[i]}，"
                    f"LLM 修改了时间戳数字"
                ), trans_ts_list

    return True, "格式正确", trans_ts_list


def normalize_translation(text):
    """纠错：把 [Speaker XX) 这种开口[闭口)的不匹配括号统一修复为 [Speaker XX]；
    同时规整掉行尾/行首多余空白。
    """
    if not text:
        return text
    # 修复 [xxx) → [xxx]
    text = re.sub(r'\[([^\]\n]+?)\)', r'[\1]', text)
    # 修复 (xxx] → (xxx) （对称修复）
    text = re.sub(r'[\(（]([^\[\]\(\)\n]+?)[\]）]', r'(\1)', text)
    return text


# ---- 本地 llama 兜底翻译（tencent/Hy-MT2-1.8B-GGUF）----
# 仅在 API 翻译失败时启用，作为本地兜底。
# 依赖安装（CPU 预编译版，节约编译时间）：
#   pip install llama-cpp-python --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
#   pip install huggingface_hub
_LLM = None
_LLM_LOCK = threading.Lock()           # 保护模型加载（单例）
_LLM_INFER_LOCK = threading.Lock()     # 保护推理（llama-cpp-python 非线程安全）
_LLM_MODEL_REPO = "tencent/Hy-MT2-1.8B-GGUF"
_LLM_MODEL_FILE = "Hy-MT2-1.8B-Q4_K_M.gguf"


def _get_local_llm():
    """懒加载本地 llama-cpp-python 模型（线程安全单例）。加载失败返回 None。"""
    global _LLM
    if _LLM is not None:
        return _LLM
    with _LLM_LOCK:
        if _LLM is not None:
            return _LLM
        try:
            from llama_cpp import Llama
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            print(f"[本地LLM] 缺少依赖（{e}），无法启用本地兜底。请安装：\n"
                  f"  pip install llama-cpp-python --no-cache-dir "
                  f"--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu\n"
                  f"  pip install huggingface_hub")
            return None
        try:
            print(f"[本地LLM] 正在下载/加载 {_LLM_MODEL_REPO}/{_LLM_MODEL_FILE} ...")
            model_path = hf_hub_download(repo_id=_LLM_MODEL_REPO, filename=_LLM_MODEL_FILE)
            _LLM = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_threads=max(1, (os.cpu_count() or 2) // 2),
                verbose=False,
            )
            print("[本地LLM] 加载完成")
        except Exception as e:
            print(f"[本地LLM] 加载失败: {e}")
            return None
    return _LLM


def translate_with_local_llama(text, orig_ts_list=None):
    """使用本地 llama 模型翻译段落。成功返回译文文本，失败返回 None。

    会做与 API 相同的格式校验；不通过则尝试一次正则纠错，仍不通过则返回 None。
    """
    llm = _get_local_llm()
    if llm is None:
        return None
    try:
        # llama-cpp-python 的 Llama 实例非线程安全，推理需串行
        with _LLM_INFER_LOCK:
            resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_tokens=4000,
                temperature=0.3,
                top_p=0.7,
            )
        content = resp["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[本地LLM] 翻译出错: {e}")
        return None

    content = filter_think_tags(content or "")
    if not content or not contains_chinese(content):
        return None

    ok, _, _ = is_valid_translation_format(content, orig_ts_list=orig_ts_list)
    if ok:
        return content
    # 尝试一次正则纠错
    normalized = normalize_translation(content)
    ok2, _, _ = is_valid_translation_format(normalized, orig_ts_list=orig_ts_list)
    if ok2:
        return normalized
    return None


def translate_text_worker(segment_data, api_config, max_retries=5):
    """并行翻译工作函数。翻译流程：
      1. API 正常 → 走 API
      2. API 返回 5xx → 立即停止重试 → 本地 llama 兜底
      3. API 其他错误 → 重试 5 次 → 纠错 + 再试 1 次 → 仍失败则本地 llama 兜底
      4. 本地兜底也失败 → 逐行原文兜底（保底）

    返回 (idx, translated, original_seg, normalized_flag)。

    segment_data 接受 (idx, text) 或 (idx, text, orig_ts_list)。
    """
    global completed_count, total_count

    if len(segment_data) == 3:
        segment_index, text, orig_ts_list = segment_data
    else:
        segment_index, text = segment_data
        orig_ts_list = _extract_orig_ts_list(text)

    url = api_config['url']
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_config.get('api_key', '')}" if api_config.get('api_key') else None
    }
    headers = {k: v for k, v in headers.items() if v is not None}

    base_data = {
        "model": api_config.get('model_name', 'default'),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        "stream": False,
        "max_tokens": 4000,
        "stop": None,
        "temperature": 0.3,
        "top_p": 0.7,
        "n": 1,
        "response_format": {"type": "text"},
    }

    last_translated = None   # 记录最后一次原始返回，用于纠错阶段
    last_normalized = None   # 记录纠错后的文本
    last_format_err = None   # 记录最后一次的格式错误信息，用于重试时反馈给模型
    hit_5xx = False          # 标记是否因服务端 5xx 错误立即退出重试

    def _build_data(retry_count):
        """根据重试次数构造请求数据；后续重试附带更明确的提示。"""
        data = json.loads(json.dumps(base_data))  # 深拷贝
        if retry_count >= 1 and last_format_err:
            reminder = (
                '\n\n【格式要求】上一轮输出存在问题，请重新输出：\n'
                f'- 问题：{last_format_err}\n'
                '- 时间戳必须与原文**完全相同**（逐字一致），禁止修改、编造、省略或调换顺序。\n'
                '- 行数应与原文相同；若因断句合并，可以比原文少 1 行。\n'
                '- 只输出翻译后的中文文本，不要添加解释、注释或任何额外内容。'
            )
            data["messages"][1]["content"] = f"{text}{reminder}"
        return data

    for retry_count in range(max_retries):
        try:
            if retry_count > 0:
                delay = 1 * (2 ** (retry_count - 1)) + (random() * 0.5)
                time.sleep(delay)

            data = _build_data(retry_count)
            response = requests.post(url, json=data, headers=headers, proxies={"http": None, "https": None})
            response.raise_for_status()
            result = response.json()

            translated_content = None
            try:
                translated_content = result['choices'][0]['message']['content']
            except (KeyError, IndexError, TypeError):
                pass

            translated_content = filter_think_tags(translated_content or "")
            last_translated = translated_content

            if not translated_content:
                print(f"[段落 {segment_index + 1}] 翻译失败或内容为空 (尝试次数: {retry_count + 1}/{max_retries})")
                last_format_err = "返回内容为空"
                continue

            if not contains_chinese(translated_content):
                print(f"[段落 {segment_index + 1}] 翻译内容未包含中文 (尝试次数: {retry_count + 1}/{max_retries})")
                last_format_err = "译文未包含中文"
                continue

            # 格式校验：行格式 + 时间戳递增 + 严格对照原文时间戳
            ok, err, _ = is_valid_translation_format(translated_content, orig_ts_list=orig_ts_list)
            if not ok:
                print(f"[段落 {segment_index + 1}] 格式校验失败 (尝试次数: {retry_count + 1}/{max_retries}): {err}")
                last_format_err = err
                continue

            with progress_lock:
                completed_count += 1
                flag = "成功"
                print(f"[进度 {completed_count}/{total_count}] 段落 {segment_index + 1} {flag} (尝试次数: {retry_count + 1})")
                if completed_count % 5 == 0 or completed_count == total_count:
                    print(f"状态码: {response.status_code}")
                    print(f"模型: {result.get('model', '未返回model信息')}")
                    preview = translated_content[:200] + ("..." if len(translated_content) > 200 else "")
                    print(f"翻译内容预览: {preview}")
                    print()
            return segment_index, translated_content, text, False

        except requests.exceptions.HTTPError as http_err:
            error_text = ""
            try:
                error_text = response.text
            except Exception:
                pass

            # 5xx：服务端错误，立即停止重试，直接转本地 llama 兜底
            if 500 <= response.status_code < 600:
                print(f"[段落 {segment_index + 1}] 服务端错误 {response.status_code}，"
                      f"立即停止重试，转本地 llama 兜底")
                if error_text:
                    print(f"  响应: {error_text[:300]}")
                last_format_err = f"HTTP {response.status_code}"
                hit_5xx = True
                break

            print(f"[段落 {segment_index + 1}] HTTP错误 (尝试次数: {retry_count + 1}/{max_retries})")
            if response.status_code == 502:
                print(f"网关错误 (502): {error_text}")
            else:
                print(f"HTTP错误: {http_err}, 响应: {error_text}")
            last_format_err = f"HTTP {response.status_code}"
            continue

        except requests.exceptions.RequestException as err:
            print(f"[段落 {segment_index + 1}] 请求错误 (尝试次数: {retry_count + 1}/{max_retries}): {err}")
            last_format_err = f"请求错误: {err}"
            continue
        except Exception as e:
            print(f"[段落 {segment_index + 1}] 其他错误 (尝试次数: {retry_count + 1}/{max_retries}): {e}")
            last_format_err = f"其他错误: {e}"
            continue

    # ---- 纠错阶段（仅非 5xx 退出时执行）----
    # 5xx 直接跳过纠错，进入本地 llama 兜底
    if not hit_5xx:
        # 5 次都失败：先对最后一次返回纠错
        if last_translated:
            normalized = normalize_translation(last_translated)
            if normalized != last_translated:
                last_normalized = normalized
                ok, err, _ = is_valid_translation_format(normalized, orig_ts_list=orig_ts_list)
                if ok:
                    with progress_lock:
                        completed_count += 1
                    print(f"[段落 {segment_index + 1}] 纠错后格式通过（无需再请求）")
                    return segment_index, normalized, text, True
                print(f"[段落 {segment_index + 1}] 纠错后仍不通过: {err}，再请求 1 次让模型自修")
            else:
                print(f"[段落 {segment_index + 1}] {max_retries} 次重试后仍未通过且无可纠错项，再请求 1 次")

        # 再请求 1 次，让模型自己修复（同时附带更明确的提示）
        try:
            time.sleep(1)
            data = _build_data(max_retries)  # 附带上 last_format_err
            response = requests.post(url, json=data, headers=headers, proxies={"http": None, "https": None}, timeout=60)
            response.raise_for_status()
            result = response.json()
            translated = None
            try:
                translated = result['choices'][0]['message']['content']
            except (KeyError, IndexError, TypeError):
                pass
            translated = filter_think_tags(translated or "")
            last_translated = translated

            if translated and contains_chinese(translated):
                ok, err, _ = is_valid_translation_format(translated, orig_ts_list=orig_ts_list)
                if ok:
                    with progress_lock:
                        completed_count += 1
                    print(f"[段落 {segment_index + 1}] 纠错阶段成功（模型自修）")
                    return segment_index, translated, text, True
                # 再纠错一次
                normalized = normalize_translation(translated)
                ok2, _, _ = is_valid_translation_format(normalized, orig_ts_list=orig_ts_list)
                if ok2:
                    with progress_lock:
                        completed_count += 1
                    print(f"[段落 {segment_index + 1}] 纠错阶段成功（正则修复+模型自修）")
                    return segment_index, normalized, text, True
                last_normalized = normalized
        except Exception as e:
            print(f"[段落 {segment_index + 1}] 纠错阶段请求出错: {e}")

    # ---- 本地 llama 兜底 ----
    print(f"[段落 {segment_index + 1}] 转入本地 llama 兜底翻译")
    llama_translated = translate_with_local_llama(text, orig_ts_list=orig_ts_list)
    if llama_translated:
        with progress_lock:
            completed_count += 1
        print(f"[进度 {completed_count}/{total_count}] 段落 {segment_index + 1} 本地 llama 兜底成功")
        return segment_index, llama_translated, text, True

    # ---- 本地兜底也失败：逐行原文兜底（保底）----
    with progress_lock:
        completed_count += 1
        print(f"[进度 {completed_count}/{total_count}] 段落 {segment_index + 1} 本地 llama 兜底失败，逐行原文兜底")

    # 返回最后一次可用文本，让 main()/merge_segment_results 逐行兜底到原文
    if last_normalized:
        return segment_index, last_normalized, text, True
    return segment_index, last_translated, text, True  # 可能为 None


def filter_think_tags(text):
    """过滤掉文本中的<think></think>标签及其中的内容"""
    pattern = r'<think>.*?</think>'
    filtered_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return filtered_text


def remove_timestamps(text):
    """从文本中移除时间戳，如(HH:MM:SS.mmm)或(MM:SS)"""
    pattern = r'[\(（]\s*(\d{1,2}:)?\d{2}(:\d{2}\.\d{1,3})?\s*[\)）]'
    text_without_timestamps = re.sub(pattern, '', text)
    return text_without_timestamps


def _ts_to_seconds(ts):
    """把时间戳字符串转成秒数。

    支持 HH:MM:SS.mmm / MM:SS.mmm / MM:SS 等任意段数。
    例：'1:23:45.678' -> 5025.678；'00:12.500' -> 12.5
    """
    parts = ts.split(':')
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + float(p)
    return seconds


MAX_SPEAKING_WPM = 440  # 语速阈值（字/分钟），超过则与相邻较慢行合并


def _merge_two_lines(line_a, line_b):
    """合并两行：保留 line_a 的时间戳，拼接两行文本。

    line_a 是时间戳较早的行（吸收者），line_b 是被吸收的行。
    返回合并后的单行，格式 '(ts) textA textB'。
    """
    ts_a = _extract_ts(line_a)
    text_a = _extract_text_after_ts(line_a)
    if not text_a:
        text_a = remove_timestamps(line_a).strip()
    text_b = _extract_text_after_ts(line_b)
    if not text_b:
        text_b = remove_timestamps(line_b).strip()
    merged_text = f"{text_a} {text_b}".strip()
    if ts_a:
        return f"({ts_a}) {merged_text}"
    return merged_text


def merge_fast_speaking_lines(lines, max_wpm=MAX_SPEAKING_WPM):
    """合并语速超过 max_wpm 的行，与相邻语速较慢者合并以降速。

    策略：
    - 逐行计算语速 = 字数 / (下一行时间戳 - 本行时间戳) * 60
    - 找到第一行超阈值的行 i，检查其左右邻居的语速
    - 选邻居中语速较慢（且比自己慢）的方向合并：
        - 合并左邻居：i 吸收进 i-1，保留 i-1 的时间戳
        - 合并右邻居：i 吸收 i+1，保留 i 的时间戳
    - 合并后重新计算所有行语速（因为时长结构变了），重复
    - 若某行超阈值但两邻居都不比自己慢，标记跳过，继续找下一个

    无时间戳的行无法计算语速，自然跳过。
    每次合并让总行数减 1，因此循环必然终止。

    返回合并后的新行列表（不修改原列表）。
    """
    lines = list(lines)
    merge_count = 0
    skipped = set()

    while True:
        n = len(lines)
        # 解析每行的时间戳（秒）和字数
        ts_secs = []
        chars = []
        for line in lines:
            ts = _extract_ts(line)
            ts_sec = None
            if ts:
                try:
                    ts_sec = _ts_to_seconds(ts)
                except (ValueError, IndexError):
                    ts_sec = None
            ts_secs.append(ts_sec)
            text_only = remove_timestamps(line)
            chars.append(len(re.sub(r'\s', '', text_only)))

        # 计算每行语速（最后一行无下一行，不计）
        wpms = [None] * n
        for i in range(n - 1):
            if ts_secs[i] is not None and ts_secs[i + 1] is not None:
                dur = ts_secs[i + 1] - ts_secs[i]
                if dur > 0 and chars[i] > 0:
                    wpms[i] = chars[i] / dur * 60

        # 找第一个可处理的超阈值行
        target = None
        merge_dir = None
        for i in range(n):
            if i in skipped or wpms[i] is None or wpms[i] <= max_wpm:
                continue
            left_wpm = wpms[i - 1] if i > 0 else None
            right_wpm = wpms[i + 1] if i < n - 1 else None
            # 在比自己慢的邻居中选最慢的
            best = None
            if left_wpm is not None and left_wpm < wpms[i]:
                best = ('left', left_wpm)
            if right_wpm is not None and right_wpm < wpms[i]:
                if best is None or right_wpm < best[1]:
                    best = ('right', right_wpm)
            if best is not None:
                target = i
                merge_dir = best[0]
                break
            else:
                # 两邻居都不比自己慢，合并无法降速，跳过
                skipped.add(i)

        if target is None:
            break

        # 执行合并，行号变化后重新评估所有行
        skipped.clear()
        if merge_dir == 'left':
            lines[target - 1] = _merge_two_lines(lines[target - 1], lines[target])
            del lines[target]
        else:
            lines[target] = _merge_two_lines(lines[target], lines[target + 1])
            del lines[target + 1]
        merge_count += 1

    if merge_count > 0:
        print(f"[语速优化] 合并了 {merge_count} 个语速过快的行（阈值 {max_wpm} 字/分）")
    return lines


def estimate_speaking_rates(lines):
    """根据每行时间戳与下一行时间戳，估算每行语速（字/分钟）。

    语速 = 本句字数 / (下一句时间戳 - 本句时间戳) * 60。
    最后一行没有“下一句”，不计语速；无时间戳的行也跳过。
    字数按去掉时间戳后的非空白字符数计算（中文按字、英文按字母，标点计入）。

    返回新的行列表，在每行末尾追加 '  [语速: X字/分]'。
    """
    # 先把每行的时间戳转成秒（无时间戳记为 None）
    ts_seconds = []
    for line in lines:
        ts = _extract_ts(line)
        if ts:
            try:
                ts_seconds.append(_ts_to_seconds(ts))
            except (ValueError, IndexError):
                ts_seconds.append(None)
        else:
            ts_seconds.append(None)

    out_lines = []
    n = len(lines)
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        # 字数：去掉时间戳后的非空白字符数
        text_only = remove_timestamps(line)
        char_count = len(re.sub(r'\s', '', text_only))

        annotated = False
        if i < n - 1 and ts_seconds[i] is not None and ts_seconds[i + 1] is not None:
            duration = ts_seconds[i + 1] - ts_seconds[i]
            if duration > 0 and char_count > 0:
                wpm = char_count / duration * 60
                out_lines.append(f"{line_stripped}  [语速: {wpm:.0f}字/分]")
                annotated = True
        if not annotated:
            out_lines.append(line_stripped)
    return out_lines


def clean_translation_content(content):
    """清理翻译内容中的多余字符（只清理字符和行内空白，不合并换行）。

    注意：必须保持原始换行结构，否则会破坏按行组织的字幕格式。
    """
    content_cleaned = content.replace('&gt;', '').replace('>>', '').replace('> ', '').replace('&nbsp;', '').replace('_', '').replace('＞', '').replace('[音乐]', '')

    # 额外清理一些可能影响TTS的字符
    content_cleaned = content_cleaned.replace('&lt;', '').replace('&amp;', '').replace('&quot;', '').replace('--', '—')

    # 清理行内多余空格（保留换行符）
    cleaned_lines = [' '.join(line.split()) for line in content_cleaned.splitlines()]
    content_cleaned = '\n'.join(cleaned_lines)

    return content_cleaned


def clean_translation_line(line):
    """对单行做字符清洗 + 行内空白规整。"""
    line = line.replace('&gt;', '').replace('>>', '').replace('> ', '').replace('&nbsp;', '').replace('_', '').replace('＞', '').replace('[音乐]', '')
    line = line.replace('&lt;', '').replace('&amp;', '').replace('&quot;', '').replace('--', '—')
    return ' '.join(line.split())


def _debug_print_paragraph(segment_index, original_seg, translated_raw, translated_clean, trans_lines):
    """打印段落原文和译文（仅在 --show_progress 或出现回退时启用）。"""
    print(f"\n----- [调试] 段落 {segment_index + 1} -----")
    print(f"  原文 ({len([l for l in original_seg.splitlines() if l.strip()])} 行):")
    for i, l in enumerate([x for x in original_seg.splitlines() if x.strip()], 1):
        print(f"    {i:>2}: {l}")
    print(f"  原始模型返回 (前 500 字符):")
    print(f"    {repr((translated_raw or '')[:500])}")
    print(f"  清洗后译文 ({len(trans_lines)} 行):")
    for i, l in enumerate(trans_lines, 1):
        print(f"    {i:>2}: {l}")


def merge_segment_results(segment_index, translated, original_seg, keep_timestamps, debug=False):
    """合并单个段落的翻译结果，失败时逐行/整段回退到原文。

    返回 (out_lines, fallback_count)。
    """
    original_lines = [l.strip() for l in original_seg.splitlines() if l.strip()]
    out_lines = []
    fallback_count = 0

    if not translated:
        # 段落整体失败：每行用原文兜底
        fallback_count += len(original_lines)
        print(f"  [翻译回退整段] 段落 {segment_index + 1}（译文为 None）")
        if debug:
            _debug_print_paragraph(segment_index, original_seg, translated, "", [])
        for line in original_lines:
            out_lines.append(line)
        return out_lines, fallback_count

    # 清洗：先按行 split，逐行清洗（避免把多行合并为 1 行）
    raw = filter_think_tags(translated)
    trans_lines = [
        clean_translation_line(l)
        for l in raw.splitlines()
        if l.strip()
    ]

    # 行数容差：YouTube 自动字幕断句不稳定，模型可能把相邻两行合成一句
    # 因此译文比原文少 1 行也算通过；其他不一致再整段回退到原文
    ALLOWED_LINE_DIFF = 1

    def _emit_line(tl):
        return remove_timestamps(tl) if not keep_timestamps else tl

    if len(trans_lines) == len(original_lines):
        # 逐行校验
        for tl, ol in zip(trans_lines, original_lines):
            # 抽取正文并校验
            txt = _extract_text_after_ts(tl) if LINE_PATTERN.match(tl) else tl
            ok, _, _ = is_valid_translation_format(tl)
            if ok and contains_chinese(txt):
                out_lines.append(_emit_line(tl))
            else:
                fallback_count += 1
                print(f"  [翻译回退] 段落 {segment_index + 1} 行: {ol[:60]}")
                out_lines.append(_emit_line(ol))
    elif len(trans_lines) == len(original_lines) - ALLOWED_LINE_DIFF:
        # 少 1 行：直接采纳译文（断句合并是可接受的）
        joined = "\n".join(trans_lines)
        ok, err, _ = is_valid_translation_format(joined)
        if ok:
            print(f"  [翻译少1行通过] 段落 {segment_index + 1}（原{len(original_lines)}行 → 译{len(trans_lines)}行，模型合并断句）")
            for tl in trans_lines:
                out_lines.append(_emit_line(tl))
        else:
            # 译文本身格式不通过，整段回退
            print(f"  [翻译回退整段] 段落 {segment_index + 1}（译文格式校验失败: {err}）")
            fallback_count += len(original_lines)
            for line in original_lines:
                out_lines.append(_emit_line(line))
    else:
        # 行数不一致且超出容差：整段用原文兜底（保守策略，确保对齐）
        print(f"  [翻译回退整段] 段落 {segment_index + 1}（译{len(trans_lines)}行/原{len(original_lines)}行，差值超出 ±{ALLOWED_LINE_DIFF}）")
        fallback_count += len(original_lines)
        if debug:
            _debug_print_paragraph(segment_index, original_seg, translated, "\n".join(trans_lines), trans_lines)
        for line in original_lines:
            out_lines.append(_emit_line(line))

    return out_lines, fallback_count


def save_translation(translated_segments, output_path, keep_timestamps=False):
    """将翻译结果保存到文件。translated_segments 是已经合并好的纯文本行列表。"""
    with open(output_path, 'w', encoding='utf-8') as file:
        for segment in translated_segments:
            # 过滤掉思考标签
            filtered_segment = filter_think_tags(segment)
            # 清理多余字符
            cleaned_segment = clean_translation_content(filtered_segment)
            # 如果不保留时间戳，则移除时间戳
            if not keep_timestamps:
                cleaned_segment = remove_timestamps(cleaned_segment)
            if cleaned_segment:
                file.write(cleaned_segment + "\n\n")


def main():
    parser = argparse.ArgumentParser(description='翻译字幕文件（支持并行处理）')
    parser.add_argument('--api_config_file', required=True, help='API配置文件路径')
    parser.add_argument('--segment_size', type=int, default=3, help='分段大小')
    parser.add_argument('--keep_timestamps', action='store_true', help='保留时间戳')
    parser.add_argument('--max_workers', type=int, default=5, help='最大并行工作线程数')
    parser.add_argument('--show_progress', action='store_true', help='显示详细进度信息')

    args = parser.parse_args()

    # 加载API配置
    api_config = load_api_config(args.api_config_file)

    # 读取处理后的字幕文件
    input_file = 'subtitles/word_level_processed.txt'
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        sys.exit(1)

    # 读取文件内容
    paragraphs = read_file(input_file)

    # 分段处理
    segments = segment_paragraphs(paragraphs, segment_size=args.segment_size, keep_timestamps=args.keep_timestamps)

    print(f"开始并行翻译任务...")
    print(f"总共 {len(segments)} 个段落，使用 {args.max_workers} 个并行工作线程")
    print(f"预计可提升性能 {min(args.max_workers, len(segments))} 倍")

    start_time = time.time()

    # 并行翻译所有段落
    segment_results = {}  # idx -> (translated, original_seg, normalized_flag)

    # 准备任务数据：(索引, 文本) 元组
    tasks = [(i, segment) for i, segment in enumerate(segments)]

    global completed_count, total_count
    completed_count = 0
    total_count = len(segments)

    # 使用线程池进行并行处理
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 提交所有翻译任务
        future_to_index = {
            executor.submit(translate_text_worker, task_data, api_config): task_data[0]
            for task_data in tasks
        }

        # 收集结果
        for future in as_completed(future_to_index):
            try:
                idx, translated_text, original_seg, normalized = future.result()
                segment_results[idx] = (translated_text, original_seg, normalized)
            except Exception as e:
                segment_index = future_to_index[future]
                print(f"[段落 {segment_index + 1}] 任务执行失败: {e}")
                segment_results[segment_index] = (None, segments[segment_index], False)

    end_time = time.time()
    total_time = end_time - start_time

    # 按原始顺序整理翻译结果，并逐行兜底
    translated_lines = []
    total_fallback_lines = 0
    failed_count = 0
    for i in range(len(segments)):
        translated_text, original_seg, normalized = segment_results.get(i, (None, segments[i], False))
        if not translated_text:
            failed_count += 1
            print(f"[警告] 段落 {i + 1} 翻译失败，整段回退到原文")
        out_lines, fb = merge_segment_results(i, translated_text, original_seg, keep_timestamps=args.keep_timestamps, debug=args.show_progress)
        total_fallback_lines += fb
        translated_lines.extend(out_lines)

    # 合并语速过快的行（固定阈值 440 字/分，默认开启）
    original_line_count = len(translated_lines)
    translated_lines = merge_fast_speaking_lines(translated_lines)
    merged_line_count = original_line_count - len(translated_lines)

    # 保存翻译结果
    output_file = os.path.splitext(input_file)[0] + '_translated.txt'
    save_translation(translated_lines, output_file, keep_timestamps=args.keep_timestamps)

    print(f"\n=== 翻译任务完成 ===")
    print(f"总段落数: {len(segments)}")
    print(f"成功翻译: {len(segments) - failed_count} 个段落")
    print(f"失败段落: {failed_count} 个")
    print(f"兜底行数: {total_fallback_lines}")
    if merged_line_count > 0:
        print(f"语速合并: {original_line_count} 行 → {len(translated_lines)} 行（合并 {merged_line_count} 行）")
    print(f"并行工作线程: {args.max_workers} 个")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均每段耗时: {total_time/len(segments):.2f} 秒")
    print(f"结果已保存到: {output_file}")
    print("\n========== 完整译文字幕 ==========")
    rate_lines = estimate_speaking_rates(translated_lines)
    print('\n'.join(rate_lines))
    print("==================================\n")


if __name__ == "__main__":
    main()
