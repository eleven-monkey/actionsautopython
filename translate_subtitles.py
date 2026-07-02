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
SYSTEM_PROMPT = """# Role: 专业翻译官
## Profile
- author: LangGPT优化中心
- version: 2.1
- language: 中英双语
- description: 专注于文本精准转换的AI翻译专家，擅长处理技术文档和日常对话场景

## Background
用户在跨国协作、技术文档处理、社交媒体互动等场景中，需要将外文内容准确转化为中文，同时保持特殊格式元素完整

## Skills
1. 多语言文本解析与重构能力
2. 时间戳识别与格式保留技术
3. 语义通顺度校验算法
4. 格式控制与冗余内容过滤

## Goals
1. 实现原文语义的精准转换
2. 保持时间戳等特殊格式元素
3. 确保输出结果自然流畅
4. 排除非翻译内容添加

## Constraints
1. 禁止添加解释性文字
2. 禁用注释或说明性符号
3. 保留原始时间戳格式（如(12:34)或(HH:MM:SS.mmm)）
4. 不处理非文本元素（如图片/表格）
5. 禁止使用工具调用（tool_calls）功能，禁止调用外部翻译api进行翻译

## Workflow
1. 接收输入内容，检测语言类型
2. 识别并标记特殊格式元素
3. 执行语义转换：
   - 日常用语：采用口语化表达
   - 技术术语：使用标准化译法
4. 输出纯翻译结果

## OutputFormat
仅返回符合以下要求的翻译文本：
1. 中文书面语表达
2. 保留原始段落结构
3. 时间戳保持(MM:SS)或(HH:MM:SS)或(HH:MM:SS.mmm)格式
4. 无任何附加符号或说明
5. 尽量只要中文，不要中英文夹杂。

## Initialization
请提供需要翻译的文本内容，我将严格遵守上述规则进行处理。"""

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


def is_valid_translation_format(text):
    """校验翻译结果格式：每行符合 (HH:MM:SS[.mmm]) 文本，且时间戳严格递增。

    返回 (ok: bool, err: str)。
    """
    if not text or not text.strip():
        return False, "文本为空"
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return False, "没有有效行"
    prev_ts = None
    for i, line in enumerate(lines, 1):
        if not LINE_PATTERN.match(line):
            return False, f"第{i}行格式不正确: {line[:80]}"
        # 提取时间戳，检查严格递增
        ts_match = LINE_PATTERN.match(line)
        if ts_match:
            ts = ts_match.group(1)
            if prev_ts is not None and ts <= prev_ts:
                return False, f"第{i}行时间戳未递增: {ts} <= {prev_ts}"
            prev_ts = ts
    return True, "格式正确"


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


def translate_text_worker(segment_data, api_config, max_retries=5):
    """并行翻译工作函数，引入：5 次重试 + 纠错阶段（再请求 1 次让模型自修）+
    逐行兜底。返回 (idx, translated, original_seg, normalized_flag)。
    """
    global completed_count, total_count

    segment_index, text = segment_data
    url = api_config['url']
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_config.get('api_key', '')}" if api_config.get('api_key') else None
    }
    headers = {k: v for k, v in headers.items() if v is not None}

    data = {
        "model": api_config.get('model_name', 'default'),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f" {text}"}
        ],
        "stream": False,
        "max_tokens": 4000,
        "stop": None,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.4,
        "n": 1,
        "response_format": {"type": "text"},
    }

    last_translated = None   # 记录最后一次原始返回，用于纠错阶段
    last_normalized = None   # 记录纠错后的文本

    for retry_count in range(max_retries):
        try:
            if retry_count > 0:
                delay = 1 * (2 ** (retry_count - 1)) + (random() * 0.5)
                time.sleep(delay)

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
                continue

            if not contains_chinese(translated_content):
                print(f"[段落 {segment_index + 1}] 翻译内容未包含中文 (尝试次数: {retry_count + 1}/{max_retries})")
                continue

            # 格式校验：行格式 + 时间戳递增
            ok, err = is_valid_translation_format(translated_content)
            if not ok:
                print(f"[段落 {segment_index + 1}] 格式校验失败 (尝试次数: {retry_count + 1}/{max_retries}): {err}")
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

            print(f"[段落 {segment_index + 1}] HTTP错误 (尝试次数: {retry_count + 1}/{max_retries})")
            if response.status_code == 502:
                print(f"网关错误 (502): {error_text}")
            else:
                print(f"HTTP错误: {http_err}, 响应: {error_text}")
            continue

        except requests.exceptions.RequestException as err:
            print(f"[段落 {segment_index + 1}] 请求错误 (尝试次数: {retry_count + 1}/{max_retries}): {err}")
            continue
        except Exception as e:
            print(f"[段落 {segment_index + 1}] 其他错误 (尝试次数: {retry_count + 1}/{max_retries}): {e}")
            continue

    # 5 次都失败：进入纠错阶段
    # 先对最后一次返回纠错
    if last_translated:
        normalized = normalize_translation(last_translated)
        if normalized != last_translated:
            last_normalized = normalized
            ok, err = is_valid_translation_format(normalized)
            if ok:
                with progress_lock:
                    completed_count += 1
                print(f"[段落 {segment_index + 1}] 纠错后格式通过（无需再请求）")
                return segment_index, normalized, text, True
            print(f"[段落 {segment_index + 1}] 纠错后仍不通过: {err}，再请求 1 次让模型自修")
        else:
            print(f"[段落 {segment_index + 1}] 5 次重试后仍未通过且无可纠错项，再请求 1 次")

    # 再请求 1 次，让模型自己修复
    try:
        time.sleep(1)
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
            ok, err = is_valid_translation_format(translated)
            if ok:
                with progress_lock:
                    completed_count += 1
                print(f"[段落 {segment_index + 1}] 纠错阶段成功（模型自修）")
                return segment_index, translated, text, True
            # 再纠错一次
            normalized = normalize_translation(translated)
            ok2, _ = is_valid_translation_format(normalized)
            if ok2:
                with progress_lock:
                    completed_count += 1
                print(f"[段落 {segment_index + 1}] 纠错阶段成功（正则修复+模型自修）")
                return segment_index, normalized, text, True
            last_normalized = normalized
    except Exception as e:
        print(f"[段落 {segment_index + 1}] 纠错阶段请求出错: {e}")

    with progress_lock:
        completed_count += 1
        print(f"[进度 {completed_count}/{total_count}] 段落 {segment_index + 1} 达到最大重试次数({max_retries})，翻译失败")

    # 全部失败：返回最后一次纠错后的文本（让 main() 逐行兜底）
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


def clean_translation_content(content):
    """清理翻译内容中的多余字符"""
    content_cleaned = content.replace('&gt;', '').replace('>>', '').replace('> ', '').replace('&nbsp;', '').replace('_', '').replace('＞', '').replace('[音乐]', '')

    # 额外清理一些可能影响TTS的字符
    content_cleaned = content_cleaned.replace('&lt;', '').replace('&amp;', '').replace('&quot;', '').replace('--', '—')

    # 清理多余的空格和换行
    content_cleaned = ' '.join(content_cleaned.split())

    return content_cleaned


def merge_segment_results(segment_index, translated, original_seg, keep_timestamps):
    """合并单个段落的翻译结果，失败时逐行/整段回退到原文。

    返回 (out_lines, fallback_count)。
    """
    original_lines = [l.strip() for l in original_seg.splitlines() if l.strip()]
    out_lines = []
    fallback_count = 0

    if not translated:
        # 段落整体失败：每行用原文兜底
        fallback_count += len(original_lines)
        for line in original_lines:
            out_lines.append(line)
        return out_lines, fallback_count

    # 清洗模型返回
    translated_clean = filter_think_tags(translated)
    translated_clean = clean_translation_content(translated_clean)
    trans_lines = [l.strip() for l in translated_clean.splitlines() if l.strip()]

    # 行数容差：YouTube 自动字幕断句不稳定，模型可能把相邻两行合成一句
    # 因此译文比原文少 1 行也算通过；其他不一致再整段回退到原文
    ALLOWED_LINE_DIFF = 1

    if len(trans_lines) == len(original_lines):
        # 逐行校验
        for tl, ol in zip(trans_lines, original_lines):
            # 抽取正文并校验
            txt = _extract_text_after_ts(tl) if LINE_PATTERN.match(tl) else tl
            ok, _ = is_valid_translation_format(tl)
            if ok and contains_chinese(txt):
                if not keep_timestamps:
                    out_lines.append(remove_timestamps(tl))
                else:
                    out_lines.append(tl)
            else:
                fallback_count += 1
                print(f"  [翻译回退] 段落 {segment_index + 1} 行: {ol[:60]}")
                if not keep_timestamps:
                    out_lines.append(remove_timestamps(ol))
                else:
                    out_lines.append(ol)
    elif len(trans_lines) == len(original_lines) - ALLOWED_LINE_DIFF:
        # 少 1 行：直接采纳译文（断句合并是可接受的）
        ok, err = is_valid_translation_format(translated_clean)
        if ok:
            print(f"  [翻译少1行通过] 段落 {segment_index + 1}（原{len(original_lines)}行 → 译{len(trans_lines)}行，模型合并断句）")
            for tl in trans_lines:
                if not keep_timestamps:
                    out_lines.append(remove_timestamps(tl))
                else:
                    out_lines.append(tl)
        else:
            # 译文本身格式不通过，整段回退
            print(f"  [翻译回退整段] 段落 {segment_index + 1}（译文格式校验失败: {err}）")
            fallback_count += len(original_lines)
            for line in original_lines:
                if not keep_timestamps:
                    out_lines.append(remove_timestamps(line))
                else:
                    out_lines.append(line)
    else:
        # 行数不一致且超出容差：整段用原文兜底（保守策略，确保对齐）
        print(f"  [翻译回退整段] 段落 {segment_index + 1}（译{len(trans_lines)}行/原{len(original_lines)}行，差值超出 ±{ALLOWED_LINE_DIFF}）")
        fallback_count += len(original_lines)
        for line in original_lines:
            if not keep_timestamps:
                out_lines.append(remove_timestamps(line))
            else:
                out_lines.append(line)

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
        out_lines, fb = merge_segment_results(i, translated_text, original_seg, keep_timestamps=args.keep_timestamps)
        total_fallback_lines += fb
        translated_lines.extend(out_lines)

    # 保存翻译结果
    output_file = os.path.splitext(input_file)[0] + '_translated.txt'
    save_translation(translated_lines, output_file, keep_timestamps=args.keep_timestamps)

    print(f"\n=== 翻译任务完成 ===")
    print(f"总段落数: {len(segments)}")
    print(f"成功翻译: {len(segments) - failed_count} 个段落")
    print(f"失败段落: {failed_count} 个")
    print(f"兜底行数: {total_fallback_lines}")
    print(f"并行工作线程: {args.max_workers} 个")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均每段耗时: {total_time/len(segments):.2f} 秒")
    print(f"结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
