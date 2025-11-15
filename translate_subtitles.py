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
from functools import partial
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
3. 保留原始时间戳格式（如(12:34)）
4. 不处理非文本元素（如图片/表格）
5. 禁止使用工具调用（tool_calls）功能，禁止调用外部翻译api进行翻译

## Workflow
1. 接收输入内容，检测语言类型
2. 识别并标记特殊格式元素
3. 执行语义转换：
   - 日常用语：采用口语化表达
   - 技术术语：使用标准化译法
5. 输出纯翻译结果

## OutputFormat
仅返回符合以下要求的翻译文本：
1. 中文书面语表达
2. 保留原始段落结构
3. 时间戳保持(MM:SS)或(HH:MM:SS)格式
4. 无任何附加符号或说明
4. 尽量只要中文，不要中英文夹杂。

## Initialization
请提供需要翻译的文本内容，我将严格遵守上述规则进行处理。"""

# 线程锁用于保护共享资源
progress_lock = threading.Lock()
completed_count = 0
total_count = 0

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
    # 大多数常见中文字符的Unicode范围（简体和繁体）
    # CJK统一表意文字：U+4E00到U+9FFF
    # CJK兼容表意文字：U+F900到U+FAFF
    # CJK统一表意文字扩展A：U+3400到U+4DBF
    # 如需要可添加其他相关范围
    chinese_pattern = re.compile(r'[\u4E00-\u9FFF\uF900-\uFAFF\u3400-\u4DBF]')
    return bool(chinese_pattern.search(text))

def translate_text_worker(segment_data, api_config, max_retries=5):
    """
    工作者函数，用于并行翻译文本
    segment_data: (index, text) 元组
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

    retry_count = 0
    base_delay = 1  # 基础延迟时间（秒）

    while retry_count < max_retries:
        try:
            if retry_count > 0:
                delay = base_delay * (2 ** (retry_count - 1)) + (random() * 0.5)
                time.sleep(delay)

            response = requests.post(url, json=data, headers=headers, proxies={"http": None, "https": None})
            response.raise_for_status()
            result = response.json()

            translated_content = None
            try:
                translated_content = result['choices'][0]['message']['content']
            except (KeyError, IndexError, TypeError):
                pass

            if translated_content and contains_chinese(translated_content):
                # 使用线程锁来安全地更新进度
                with progress_lock:
                    completed_count += 1
                    print(f"[进度 {completed_count}/{total_count}] 段落 {segment_index + 1} 翻译成功! (尝试次数: {retry_count + 1})")
                    if completed_count % 5 == 0 or completed_count == total_count:  # 每5个或最后一个显示详细信息
                        print(f"状态码: {response.status_code}")
                        print(f"模型: {result.get('model', '未返回model信息')}")
                        preview = translated_content[:200] + ("..." if len(translated_content) > 200 else "")
                        print(f"翻译内容预览: {preview}")
                        print()
                return segment_index, translated_content
            elif translated_content and not contains_chinese(translated_content):
                print(f"[段落 {segment_index + 1}] 翻译内容未包含中文 (尝试次数: {retry_count + 1}/{max_retries})")
                retry_count += 1
                continue
            else:
                print(f"[段落 {segment_index + 1}] 翻译失败或内容为空 (尝试次数: {retry_count + 1}/{max_retries})")
                retry_count += 1
                continue

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
            retry_count += 1
            continue

        except requests.exceptions.RequestException as err:
            print(f"[段落 {segment_index + 1}] 请求错误 (尝试次数: {retry_count + 1}/{max_retries}): {err}")
            retry_count += 1
            continue
        except Exception as e:
            print(f"[段落 {segment_index + 1}] 其他错误 (尝试次数: {retry_count + 1}/{max_retries}): {e}")
            retry_count += 1
            continue

    with progress_lock:
        completed_count += 1
        print(f"[进度 {completed_count}/{total_count}] 段落 {segment_index + 1} 达到最大重试次数({max_retries})，翻译失败")
    
    return segment_index, None

def filter_think_tags(text):
    """过滤掉文本中的<think></think>标签及其中的内容"""
    # 使用正则表达式匹配<think>标签及其内容
    pattern = r'<think>.*?</think>'
    # Use re.DOTALL flag to make . match any character including newline
    filtered_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return filtered_text

def remove_timestamps(text):
    """从文本中移除时间戳，如(HH:MM:SS.mmm)或(MM:SS)"""
    # 模式匹配时间戳，如(HH:MM:SS.mmm)或(MM:SS)
    pattern = r'\((\d{1,2}:)?\d{2}(:\d{2}\.\d{2})?\)'
    text_without_timestamps = re.sub(pattern, '', text)
    return text_without_timestamps

def clean_translation_content(content):
    """清理翻译内容中的多余字符"""
    # Remove '>', '>>' and '& trash;' characters
    content_cleaned = content.replace('&gt;', '').replace('>>', '').replace('> ', '').replace('&nbsp;','').replace('_','').replace('＞','')
    
    # 额外清理一些可能影响TTS的字符
    content_cleaned = content_cleaned.replace('&lt;', '').replace('&amp;', '').replace('&quot;', '').replace('--', '—')
    
    # 清理多余的空格和换行
    content_cleaned = ' '.join(content_cleaned.split())
    
    return content_cleaned

def save_translation(translated_segments, output_path, keep_timestamps=False):
    """将翻译结果保存到文件"""
    with open(output_path, 'w', encoding='utf-8') as file:
        for segment in translated_segments:
            # 过滤掉思考标签
            filtered_segment = filter_think_tags(segment)
            # 清理多余字符
            cleaned_segment = clean_translation_content(filtered_segment)
            # 如果不保留时间戳，则移除时间戳
            if not keep_timestamps:
                cleaned_segment = remove_timestamps(cleaned_segment)
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
    translated_results = {}
    
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
                segment_index, translated_text = future.result()
                if translated_text:
                    translated_results[segment_index] = translated_text
            except Exception as e:
                segment_index = future_to_index[future]
                print(f"[段落 {segment_index + 1}] 任务执行失败: {e}")
                translated_results[segment_index] = None
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 按原始顺序整理翻译结果
    translated_segments = []
    failed_count = 0
    for i in range(len(segments)):
        translated_text = translated_results.get(i)
        if translated_text:
            translated_segments.append(translated_text)
        else:
            failed_count += 1
            print(f"[警告] 段落 {i + 1} 翻译失败")
    
    # 保存翻译结果
    output_file = os.path.splitext(input_file)[0] + '_translated.txt'
    save_translation(translated_segments, output_file, keep_timestamps=args.keep_timestamps)
    
    print(f"\n=== 翻译任务完成 ===")
    print(f"总段落数: {len(segments)}")
    print(f"成功翻译: {len(translated_segments)} 个段落")
    print(f"失败段落: {failed_count} 个")
    print(f"并行工作线程: {args.max_workers} 个")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均每段耗时: {total_time/len(segments):.2f} 秒")
    if len(translated_segments) > 0:
        print(f"性能提升: 相比串行处理，预计提升 {min(args.max_workers, len(segments))} 倍")
    print(f"结果已保存到: {output_file}")

if __name__ == "__main__":
    main()