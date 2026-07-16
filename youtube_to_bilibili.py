#!/usr/bin/env python3
"""
YouTube视频下载、音轨替换并上传到B站的脚本
适用于GitHub Actions环境

支持两阶段运行：
  - prepare 阶段：下载视频、替换音轨、下载/压缩封面、生成上传配置
  - upload  阶段：仅读取产物目录中的 final_video.mp4 / cover.jpeg / upload_config.pkl，执行上传

通过 --skip-prepare / --skip-upload 控制两阶段独立执行，便于失败时只重试上传步骤。
"""

import os
import sys
import argparse
import glob
import subprocess
import pickle
import time
import json
import re
import threading
from random import random
import requests
from pathlib import Path
from typing import Optional, Dict, Any

# 导入所需的库
try:
    import yt_dlp
    from PIL import Image
    # 还原为原始的bilibili_api导入
    from bilibili_api import sync, video_uploader, Credential
except ImportError as e:
    print(f"错误: 缺少必要的库: {e}")
    print("请运行: pip install yt-dlp pillow bilibili-api-python")
    sys.exit(1)


# =====================================================================
# 公共工具：与 translate_subtitles.py 保持一致，避免重复造轮子
# =====================================================================
def contains_chinese(text: str) -> bool:
    """检查文本是否包含任何中文字符（Unicode 范围）。"""
    if not text:
        return False
    chinese_pattern = re.compile(r'[\u4E00-\u9FFF\uF900-\uFAFF\u3400-\u4DBF]')
    return bool(chinese_pattern.search(text))


def filter_think_tags(text: str) -> str:
    """过滤掉文本中的 <think></think> 标签及其内容。"""
    if not text:
        return text
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)


# =====================================================================
# 本地 llama 兜底（与 translate_subtitles.py 共享同一份 HF 缓存）
# 仅在标题翻译 / 标签生成 API 调用失败时启用，作为最后一道防线
# =====================================================================
_LLM = None
_LLM_LOCK = threading.Lock()           # 保护模型加载（单例）
_LLM_INFER_LOCK = threading.Lock()     # 保护推理（llama-cpp-python 非线程安全）
_LLM_MODEL_REPO = "tencent/Hy-MT2-1.8B-GGUF"
_LLM_MODEL_FILE = "Hy-MT2-1.8B-Q4_K_M.gguf"

# 标题翻译专用 prompt（与 generate_upload_config 内联 prompt 保持一致）
TITLE_SYSTEM_PROMPT = """# role
爆款视频up主

## 任务
将英文标题翻译成吸引眼球的爆款视频中文标题。

## 输出格式
翻译后的中文标题

## 输出内容要求
不要给出选项，直接给出翻译后的中文标题
"""

# 标签生成专用 prompt（与 generate_tags_by_ai 内联 prompt 保持一致）
TAGS_SYSTEM_PROMPT = """# role
视频内容分析师

## 任务
根据视频标题生成3-5个相关的中文标签，用逗号分隔。

## 输出格式
标签1,标签2,标签3

## 输出内容要求
1. 只输出标签，不要其他文字
2. 标签要简洁明了
3. 标签要与视频内容相关
"""


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


def _call_local_llm_simple(text: str, system_prompt: str, max_tokens: int = 256) -> Optional[str]:
    """本地 llama 简单调用：单条输入 + 系统提示词，返回清洗后的文本，失败返回 None。

    适用于标题/标签这类短文本生成场景（非字幕时间戳格式）。
    """
    llm = _get_local_llm()
    if llm is None:
        return None
    try:
        with _LLM_INFER_LOCK:
            resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
                top_p=0.7,
            )
        content = resp["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[本地LLM] 推理出错: {e}")
        return None
    content = filter_think_tags(content or "")
    return content.strip() if content else None


def _compute_retry_delay(retry_count: int, is_5xx: bool = False) -> float:
    """计算第 retry_count 次重试前的退避秒数（指数退避 + 抖动）。

    5xx 用更长 base（3s），给服务端恢复时间；其他错误用 1s。
    例（5xx）：retry_count=1→3s, 2→6s, 3→12s, 4→24s, 5→48s（+ 0~0.5s 抖动）
    例（其他）：retry_count=1→1s, 2→2s, 3→4s, 4→8s, 5→16s（+ 0~0.5s 抖动）
    """
    base = 3 if is_5xx else 1
    return base * (2 ** (retry_count - 1)) + (random() * 0.5)


def load_api_config(config_file: str) -> Dict[str, Any]:
    """
    从JSON文件加载API配置；若文件不存在或字段缺失，则允许使用环境变量作为回退。
    返回字典，可能包含键：url, api_key, model_name
    """
    config = {}
    if not config_file:
        return config

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f) or {}
    except FileNotFoundError:
        print(f"警告: 找不到配置文件 {config_file}，将尝试从环境变量读取API配置")
        config = {}
    except json.JSONDecodeError:
        print(f"错误: 配置文件 {config_file} 格式不正确，将尝试从环境变量读取API配置")
        config = {}
    except Exception as e:
        print(f"读取配置文件时发生意外错误: {e}，将尝试从环境变量读取API配置")
        config = {}

    # 不强制要求所有字段；优先使用文件里的设置，缺失时由环境变量补充
    final_config = {
        'url': config.get('url') or os.environ.get('AI_API_URL') or os.environ.get('AI_URL') or '',
        'api_key': config.get('api_key') or os.environ.get('AI_API_KEY') or os.environ.get('AI_KEY') or '',
        'model_name': config.get('model_name') or os.environ.get('AI_MODEL') or config.get('model') or ''
    }

    return final_config


def clean_existing_files(file_pattern: str) -> None:
    """清理已存在的文件"""
    files = glob.glob(file_pattern)
    if files:
        print(f"找到 {len(files)} 个匹配 {file_pattern} 的文件，正在删除...")
        for file in files:
            os.remove(file)
        print("文件删除完成")
    else:
        print(f"没有找到匹配 {file_pattern} 的文件")


def download_youtube_video(url: str, output_path: str, cookies_file: str = None) -> bool:
    """下载YouTube视频（仅视频流），使用 subprocess 调用 yt-dlp"""
    # 尝试多种格式选择器，从最具体到最通用
    format_selectors = [
        'bestvideo[height<=1080]',  # 首选：最高1080p的视频流
        'bestvideo[height<=720]',   # 备选：最高720p的视频流
        'bestvideo',               # 最后备选：任何可用的视频流
        'best[height<=1080]',      # 如果单独视频流不可用，尝试包含音频的
        'best[height<=720]',       # 较低分辨率版本
        'best'                     # 最后的备选方案
    ]

    for format_selector in format_selectors:
        print(f"尝试使用格式选择器: {format_selector}")

        cmd = [
            'yt-dlp',
            '--extractor-args', 'youtube:player_client=default,-web_safari',
            '--extractor-args', 'youtube:js_runtime=/usr/local/bin/deno',
            '--remote-components', 'ejs:github',
            '--no-playlist',
            '-f', format_selector,
            '-o', output_path,
            url
        ]

        # 添加cookies支持
        if cookies_file and os.path.exists(cookies_file):
            cmd.insert(1, '--cookies')
            cmd.insert(2, cookies_file)
            print(f"使用cookies文件: {cookies_file}")

        print(f"正在从URL下载视频: {url}")
        print(f"执行命令: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            if result.returncode == 0:
                print(f"视频已成功下载到 {output_path}")
                return True
            else:
                print(f"使用格式 {format_selector} 下载失败: {result.stderr}")
                if "Requested format is not available" in result.stderr:
                    continue
                else:
                    return False
        except Exception as e:
            print(f"下载视频时发生意外错误: {e}")
            return False

    # 所有格式选择器都失败了
    print("所有格式选择器都失败了，无法下载视频")
    return False


def replace_audio_track(video_path: str, audio_path: str, output_path: str) -> bool:
    """使用ffmpeg替换视频的音轨"""
    if not os.path.exists(audio_path):
        print(f"错误: 找不到音频文件 {audio_path}")
        return False

    print(f"正在替换 {video_path} 的音轨为 {audio_path}")
    try:
        # ffmpeg命令: -i input_video -i input_audio -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 output_video
        subprocess.run([
            'ffmpeg', '-y', '-i', video_path, '-i', audio_path,
            '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
            output_path
        ], check=True)
        print(f"音轨替换完成。最终视频已保存到 {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg处理错误: {e}")
        return False
    except FileNotFoundError:
        print("错误: 找不到ffmpeg命令。请确保已安装ffmpeg。")
        return False
    except Exception as e:
        print(f"替换音轨时发生意外错误: {e}")
        return False


def download_thumbnail(url: str, output_path: str, cookies_file: str = None) -> bool:
    """下载YouTube视频的缩略图，使用 subprocess 调用 yt-dlp"""
    cmd = [
        'yt-dlp',
        '--extractor-args', 'youtube:player_client=default,-web_safari',
        '--extractor-args', 'youtube:js_runtime=/usr/local/bin/deno',
        '--remote-components', 'ejs:github',
        '--no-playlist',
        '--skip-download',
        '--write-thumbnail',
        '-o', output_path,
        url
    ]

    # 添加cookies支持
    if cookies_file and os.path.exists(cookies_file):
        cmd.insert(1, '--cookies')
        cmd.insert(2, cookies_file)

    print(f"正在下载URL的缩略图: {url}")
    print(f"执行命令: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if result.returncode == 0:
            print(f"缩略图已成功下载") # yt-dlp 会自动添加后缀，output_path 只是模板
            return True
        else:
            print(f"缩略图下载错误: {result.stderr}")
            return False
    except Exception as e:
        print(f"下载缩略图时发生意外错误: {e}")
        return False


def convert_and_compress_to_jpeg(input_path: str, output_path: str, target_size_kb: int = 50) -> bool:
    """将图像转换为JPEG格式并调整质量，使其小于目标大小"""
    if not os.path.exists(input_path):
        print(f"错误: 找不到输入文件 {input_path}")
        return False

    try:
        with Image.open(input_path) as img:
            # 确保图像在保存为JPEG之前是RGB模式
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # 初始质量和大小检查
            quality = 90
            img.save(output_path, 'jpeg', quality=quality)
            current_size_kb = os.path.getsize(output_path) / 1024

            # 如有必要，调整质量
            while current_size_kb > target_size_kb and quality > 4:
                quality -= 5
                img.save(output_path, 'jpeg', quality=quality)
                current_size_kb = os.path.getsize(output_path) / 1024
                print(f"当前大小: {current_size_kb:.2f} KB，质量: {quality}")

            if current_size_kb <= target_size_kb:
                print(f"成功转换并压缩到 {output_path}，大小为 {current_size_kb:.2f} KB")
            else:
                print(f"警告: 无法将 {input_path} 压缩到 {target_size_kb} KB 以下。最终大小为 {current_size_kb:.2f} KB。")
            return True
    except Exception as e:
        print(f"转换过程中发生错误: {e}")
        return False

def generate_tags_by_ai(title: str, api_config: Dict[str, Any]) -> str:
    """使用AI根据视频标题生成相关标签。优先使用api_config中的 model_name/api_key/url，缺失则使用环境变量。

    失败兜底链（与 translate_text_worker 对齐）：
      1. API 正常 → 走 API
      2. API 返回 5xx → 立即停止重试 → 本地 llama 兜底
      3. API 其他错误 / 内容无效 → 重试 5 次（指数退避）→ 本地 llama 兜底
      4. 本地兜底也失败 → 返回默认标签 "科普"（保底）
    """
    DEFAULT_TAGS = "科普"
    MAX_RETRIES = 5

    model = api_config.get('model_name') or os.environ.get('AI_MODEL') or "THUDM/GLM-4-9B-0414"
    api_url = api_config.get('url') or os.environ.get('AI_API_URL') or os.environ.get('AI_URL') or ''
    api_key = api_config.get('api_key') or os.environ.get('AI_API_KEY') or os.environ.get('AI_KEY') or ''

    system_prompt = """
# role
视频内容分析师

## 任务
根据视频标题生成3-5个相关的中文标签，用逗号分隔。

## 输出格式
标签1,标签2,标签3

## 输出内容要求
1. 只输出标签，不要其他文字
2. 标签要简洁明了
3. 标签要与视频内容相关
"""

    if not api_url or not api_key:
        print("警告: AI api_url 或 api_key 未配置（api_config 或 环境变量），将返回默认标签")
        return DEFAULT_TAGS

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": title}
        ]
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # ---- 阶段 1：API 调用 + 重试（5xx 也重试，用更长退避）----
    last_err = None
    last_was_5xx = False  # 上一轮是否 5xx，决定下次重试的退避 base
    for retry_count in range(MAX_RETRIES):
        # 退避：非首次时根据上一轮错误类型选 base
        if retry_count > 0:
            time.sleep(_compute_retry_delay(retry_count, is_5xx=last_was_5xx))
        last_was_5xx = False

        try:
            response = requests.post(api_url, json=payload, headers=headers, timeout=30)

            # 5xx：服务端错误，继续重试（用更长退避），不立即转 llama
            if 500 <= response.status_code < 600:
                err_text = (response.text or "")[:300]
                print(f"生成标签 API 服务端错误 {response.status_code}，"
                      f"将在下轮用更长退避重试 (尝试次数: {retry_count + 1}/{MAX_RETRIES}): {err_text}")
                last_err = f"HTTP {response.status_code}"
                last_was_5xx = True
                continue

            response.raise_for_status()
            response_data = response.json()

            # 兼容常见的返回结构
            tags = None
            try:
                tags = response_data['choices'][0]['message']['content'].strip()
            except Exception:
                try:
                    tags = response_data.get('data', [])[0].get('content', '').strip()
                except Exception:
                    tags = None

            if not tags:
                last_err = "AI 返回中未找到标签内容"
                print(f"生成标签失败或内容为空 (尝试次数: {retry_count + 1}/{MAX_RETRIES})")
                continue

            # 简单校验：至少要有 1 个非空标签项
            tag_list = [t.strip() for t in tags.split(',') if t.strip()]
            if not tag_list:
                last_err = "返回内容解析后无有效标签"
                print(f"生成标签无有效项 (尝试次数: {retry_count + 1}/{MAX_RETRIES})")
                continue

            print(f"生成标签成功 (尝试次数: {retry_count + 1}/{MAX_RETRIES}): {tags}")
            return tags

        except requests.exceptions.RequestException as err:
            print(f"生成标签 API 请求错误 (尝试次数: {retry_count + 1}/{MAX_RETRIES}): {err}")
            last_err = f"请求错误: {err}"
            continue
        except Exception as e:
            print(f"生成标签其他错误 (尝试次数: {retry_count + 1}/{MAX_RETRIES}): {e}")
            last_err = f"其他错误: {e}"
            continue

    if last_err:
        print(f"生成标签 {MAX_RETRIES} 次重试后仍未通过（{last_err}），转本地 llama 兜底")

    # ---- 阶段 2：本地 llama 兜底 ----
    llama_tags = _call_local_llm_simple(title, TAGS_SYSTEM_PROMPT, max_tokens=128)
    if llama_tags:
        tag_list = [t.strip() for t in llama_tags.split(',') if t.strip()]
        if tag_list:
            print(f"生成标签本地 llama 兜底成功: {llama_tags}")
            return llama_tags
        print(f"生成标签本地 llama 兜底返回内容无有效标签: {llama_tags!r}")

    # ---- 阶段 3：终极兜底 ----
    print(f"生成标签完全失败（API + llama 均不可用），使用默认标签 {DEFAULT_TAGS!r}")
    return DEFAULT_TAGS

def generate_upload_config(youtube_url: str, api_config_file: str, output_path: str, cookies_file: str = None) -> Dict[str, Any]:
    """生成上传配置文件，包括翻译标题。优先使用 JSON 文件中的 model/url/api_key，否则回退到环境变量。"""
    # 使用load_api_config函数读取API配置（支持回退到环境变量）
    api_config = load_api_config(api_config_file) or {}
    if not api_config:
        print("警告: 未从文件或环境中加载到AI配置，AI翻译将会回退到原始标题。")

    # 获取YouTube视频标题
    cmd = [
        'yt-dlp',
        '--extractor-args', 'youtube:player_client=default,-web_safari',
        '--extractor-args', 'youtube:js_runtime=/usr/local/bin/deno',
        '--remote-components', 'ejs:github',
        '--no-playlist',
        '--dump-json',
        '--skip-download',
        youtube_url
    ] # Note: first arg was renamed to url in this scope, wait, function arg is youtube_url

    # Check function arguments: def generate_upload_config(youtube_url: str, ...
    cmd[-1] = youtube_url

    # cookies支持
    if cookies_file and os.path.exists(cookies_file):
        cmd.insert(1, '--cookies')
        cmd.insert(2, cookies_file)
        print(f"获取标题时使用cookies文件: {cookies_file}")

    try:
        print(f"获取标题命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if result.returncode == 0:
            info_dict = json.loads(result.stdout)
            title = info_dict.get('title', None)
            print(f"视频标题: {title}")
        else:
            print(f"获取视频标题失败 (yt-dlp error): {result.stderr}")
            return {}
    except Exception as e:
        print(f"获取视频标题失败: {e}")
        return {}

    model = api_config.get('model_name') or os.environ.get('AI_MODEL') or "THUDM/GLM-4-9B-0414"
    api_url = api_config.get('url') or os.environ.get('AI_API_URL') or os.environ.get('AI_URL') or ''
    api_key = api_config.get('api_key') or os.environ.get('AI_API_KEY') or os.environ.get('AI_KEY') or ''

    system_prompt = """
# role
爆款视频up主

## 任务
将英文标题翻译成吸引眼球的爆款视频中文标题。

## 输出格式
翻译后的中文标题

## 输出内容要求
不要给出选项，直接给出翻译后的中文标题
"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": title}
        ]
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    translated_title = None
    if not api_url or not api_key:
        print("警告: AI api_url 或 api_key 未配置（api_config 或 环境变量），将使用原始标题作为翻译结果")
        translated_title = title
    else:
        # ---- 阶段 1：API 调用 + 重试（5xx 也重试，用更长退避）----
        # 失败兜底链（与 translate_text_worker 对齐）：
        #   1. API 正常 → 走 API
        #   2. API 返回 5xx → 继续重试（用更长退避给服务端恢复时间）
        #   3. API 其他错误 / 内容无效 → 重试 5 次（指数退避）
        #   4. 重试 5 次仍失败 → 本地 llama 兜底
        #   5. 本地兜底也失败 → 使用原始英文标题（保底）
        MAX_RETRIES = 5
        MAX_TITLE_LEN = 100  # 标题正常不超过 100 字符，超过视为模型复读/异常输出
        last_err = None
        last_was_5xx = False  # 上一轮是否 5xx，决定下次重试的退避 base
        for retry_count in range(MAX_RETRIES):
            # 退避：非首次时根据上一轮错误类型选 base（5xx→3s，其他→1s）
            if retry_count > 0:
                time.sleep(_compute_retry_delay(retry_count, is_5xx=last_was_5xx))
            last_was_5xx = False

            try:
                response = requests.post(api_url, json=payload, headers=headers, timeout=30)

                # 5xx：服务端错误，继续重试（用更长退避）
                if 500 <= response.status_code < 600:
                    err_text = (response.text or "")[:300]
                    print(f"翻译标题 API 服务端错误 {response.status_code}，"
                          f"将在下轮用更长退避重试 (尝试次数: {retry_count + 1}/{MAX_RETRIES}): {err_text}")
                    last_err = f"HTTP {response.status_code}"
                    last_was_5xx = True
                    continue

                response.raise_for_status()
                response_data = response.json()

                # 兼容常见的返回结构
                try:
                    translated_title_with_markdown = response_data['choices'][0]['message']['content']
                except Exception:
                    # 兼容不同返回格式
                    translated_title_with_markdown = (
                        response_data.get('data', [])[0].get('content', '')
                        if response_data.get('data') else ''
                    )

                if not translated_title_with_markdown:
                    last_err = "AI 返回中未找到标题内容"
                    print(f"翻译标题失败或内容为空 (尝试次数: {retry_count + 1}/{MAX_RETRIES})")
                    continue

                candidate = translated_title_with_markdown.replace('**', '').strip()

                # 校验 1：必须包含中文
                if not contains_chinese(candidate):
                    last_err = "译文未包含中文"
                    print(f"翻译标题未包含中文 (尝试次数: {retry_count + 1}/{MAX_RETRIES}): {candidate[:80]!r}")
                    continue

                # 校验 2：长度合理（标题一般不超过 100 字符，过长视为异常）
                if len(candidate) > MAX_TITLE_LEN:
                    last_err = f"译文过长 ({len(candidate)} > {MAX_TITLE_LEN})"
                    print(f"翻译标题过长 (尝试次数: {retry_count + 1}/{MAX_RETRIES}): "
                          f"{len(candidate)} 字符")
                    continue

                print(f"翻译标题成功 (尝试次数: {retry_count + 1}/{MAX_RETRIES}): {candidate}")
                translated_title = candidate
                break  # 成功，退出重试

            except requests.exceptions.RequestException as err:
                print(f"翻译标题 API 请求错误 (尝试次数: {retry_count + 1}/{MAX_RETRIES}): {err}")
                last_err = f"请求错误: {err}"
                continue
            except Exception as e:
                print(f"翻译标题其他错误 (尝试次数: {retry_count + 1}/{MAX_RETRIES}): {e}")
                last_err = f"其他错误: {e}"
                continue

        # ---- 阶段 2：本地 llama 兜底（仅在 API 全部失败时启用）----
        if translated_title is None:
            if last_err:
                print(f"翻译标题 {MAX_RETRIES} 次重试后仍未通过（{last_err}），转本地 llama 兜底")

            llama_title = _call_local_llm_simple(title, TITLE_SYSTEM_PROMPT, max_tokens=128)
            if llama_title and contains_chinese(llama_title) and len(llama_title) <= MAX_TITLE_LEN:
                llama_title = llama_title.replace('**', '').strip()
                print(f"翻译标题本地 llama 兜底成功: {llama_title}")
                translated_title = llama_title
            else:
                # ---- 阶段 3：终极兜底（原始英文标题 + 上游会加 (中配) 前缀）----
                print(f"翻译标题完全失败（API + llama 均不可用），使用原始英文标题兜底: {title!r}")
                translated_title = title

    # 使用AI生成标签
    tags = generate_tags_by_ai(title, api_config)
    print(f"生成的标签: {tags}")

    tags_list = [tag.strip() for tag in tags.split(',') if tag.strip()] if tags else []

    # 准备要pickle的数据
    upload_data = {
        'title_desc': '(中配)' + (translated_title if translated_title else title),
        'tags': tags_list or ['默认标签']
    }

    # 序列化并保存到文件
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(upload_data, f)
        print(f"\n配置已保存到 {output_path}")
        print("保存的数据:")
        print(upload_data)
        return upload_data
    except Exception as e:
        print(f"保存配置时出错: {e}")
        return {}


async def upload_to_bilibili(video_path: str, cover_path: str, config: Dict[str, Any],
                           credential: Credential, max_retries: int = 6) -> bool:
    """上传视频到B站"""
    title_desc = config.get('title_desc', '默认标题')
    tags = config.get('tags', ['默认标签'])

    # 还原为原始的video_uploader调用
    vu_meta = video_uploader.VideoMeta(
        tid=130,  # 音乐分类
        title=title_desc,
        tags=tags,
        desc=title_desc,
        cover=cover_path,
        no_reprint=True,
    )

    page = video_uploader.VideoUploaderPage(
        path=video_path,
        title=title_desc,
        description=title_desc,
    )

    uploader = video_uploader.VideoUploader(
        [page], vu_meta, credential, line=video_uploader.Lines.QN
    )  # 选择七牛线路，不选则自动测速选择最优线路

    @uploader.on("__ALL__")
    async def ev(data):
        print(data)

    retry_count = 0
    while retry_count < max_retries:
        try:
            await uploader.start()
            print("视频上传成功完成。")
            return True
        except Exception as e:
            retry_count += 1
            print(f"视频上传错误: {e}")
            if retry_count < max_retries:
                delay = 10 * (2 ** (retry_count - 1))  # 指数退避延迟
                print(f"重试上传 (尝试 {retry_count}/{max_retries})。等待 {delay} 秒...")
                time.sleep(delay)
            else:
                print(f"达到最大上传重试次数 ({max_retries})。上传失败。")
                return False


def upload_to_dailymotion(video_path: str, config: Dict[str, Any]) -> bool:
    """上传视频到Dailymotion"""
    # 从环境变量读取Dailymotion凭证
    dm_api_key = os.environ.get('DM_API_KEY', '')
    dm_api_secret = os.environ.get('DM_API_SECRET', '')
    dm_username = os.environ.get('DM_USERNAME', '')
    dm_password = os.environ.get('DM_PASSWORD', '')

    if not all([dm_api_key, dm_api_secret, dm_username, dm_password]):
        print("错误: 未设置Dailymotion所需的环境变量（DM_API_KEY, DM_API_SECRET, DM_USERNAME, DM_PASSWORD）")
        return False

    try:
        import dailymotion
    except ImportError:
        print("错误: 缺少dailymotion库，请运行: pip install dailymotion")
        return False

    title_desc = config.get('title_desc', '默认标题')

    try:
        d = dailymotion.Dailymotion()
        d.set_grant_type(
            'password',
            api_key=dm_api_key,
            api_secret=dm_api_secret,
            scope=['manage_videos'],
            info={'username': dm_username, 'password': dm_password}
        )
        print(f"正在上传视频到Dailymotion: {video_path}")
        url = d.upload(video_path)
        d.post(
            '/me/videos',
            {
                'url': url,
                'title': title_desc,
                'is_created_for_kids': 'false',
                'published': 'true',
                'channel': 'news'
            }
        )
        print("视频已成功上传到Dailymotion")
        return True
    except Exception as e:
        print(f"上传到Dailymotion时出错: {e}")
        return False


# ===================== 两阶段运行辅助函数 =====================

def run_prepare_stage(args, github_workspace: str,
                      downloaded_video_path: str,
                      final_video_path: str,
                      thumbnail_path: str,
                      compressed_thumbnail_path: str,
                      upload_config_path: str,
                      translated_audio_path: str,
                      api_config_file_path: str,
                      cookies_file_path: Optional[str]) -> bool:
    """
    执行 prepare 阶段：生成上传配置 + 下载视频 + 替换音轨 + 下载/压缩封面。
    产物统一输出到 args.output_dir，便于后续作为 artifact 打包。
    """
    # 检查翻译后的音频文件是否存在（prepare 阶段必需）
    if not os.path.exists(translated_audio_path):
        print(f"错误: 找不到翻译后的音频文件 {translated_audio_path}")
        print(f"当前工作目录: {os.getcwd()}")
        parent_dir = os.path.dirname(translated_audio_path)
        if os.path.exists(parent_dir):
            print(f"文件所在目录 '{parent_dir}' 的内容: {os.listdir(parent_dir)}")
        else:
            print(f"文件所在目录 '{parent_dir}' 本身不存在。")
        return False

    # 清理 prepare 工作目录中已有的临时文件
    clean_existing_files(os.path.join(args.work_dir, '*.mp4'))
    clean_existing_files(os.path.join(args.work_dir, 'cover.*'))

    # 步骤1: 生成上传配置（包括翻译标题）
    print("=" * 60)
    print("[prepare] 步骤 1/5: 生成上传配置（AI 翻译标题 + 生成 tags）")
    print("=" * 60)
    upload_config = generate_upload_config(
        args.url, api_config_file_path, upload_config_path, cookies_file_path
    )
    if not upload_config:
        print("[prepare] 生成上传配置失败，终止流程")
        return False

    # 步骤2: 下载YouTube视频
    print("=" * 60)
    print("[prepare] 步骤 2/5: 下载 YouTube 视频")
    print("=" * 60)
    if not download_youtube_video(args.url, downloaded_video_path, cookies_file_path):
        print("[prepare] 视频下载失败，终止流程")
        return False

    # 步骤3: 替换音轨
    print("=" * 60)
    print("[prepare] 步骤 3/5: 替换音轨 (原声 -> 中配 TTS)")
    print("=" * 60)
    if not replace_audio_track(downloaded_video_path, translated_audio_path, final_video_path):
        print("[prepare] 音轨替换失败，终止流程")
        return False

    # 步骤4: 下载缩略图
    print("=" * 60)
    print("[prepare] 步骤 4/5: 下载 YouTube 缩略图")
    print("=" * 60)
    if not download_thumbnail(args.url, thumbnail_path, cookies_file_path):
        print("[prepare] 缩略图下载失败，终止流程")
        return False

    # 步骤5: 压缩缩略图
    print("=" * 60)
    print("[prepare] 步骤 5/5: 压缩缩略图为 cover.jpeg")
    print("=" * 60)
    thumbnail_files = glob.glob(os.path.join(args.work_dir, 'cover.*'))
    if not thumbnail_files:
        print("[prepare] 找不到下载的缩略图文件")
        return False
    if not convert_and_compress_to_jpeg(thumbnail_files[0], compressed_thumbnail_path):
        print("[prepare] 缩略图压缩失败，终止流程")
        return False

    # 清理中间临时文件，保留最终产物
    try:
        if os.path.exists(downloaded_video_path):
            os.remove(downloaded_video_path)
            print(f"已清理中间文件: {downloaded_video_path}")
        for f in thumbnail_files:
            if os.path.exists(f):
                os.remove(f)
        print(f"已清理原始缩略图文件")
    except Exception as e:
        print(f"清理临时文件时出错（不影响主流程）: {e}")

    print("=" * 60)
    print(f"[prepare] 准备阶段完成，产物目录: {args.output_dir}")
    for name in ['final_video.mp4', 'cover.jpeg', 'upload_config.pkl']:
        p = os.path.join(args.output_dir, name)
        if os.path.exists(p):
            size = os.path.getsize(p)
            print(f"  - {name}: {size / (1024*1024):.2f} MB")
    print("=" * 60)
    return True


def run_upload_stage(args, final_video_path: str, compressed_thumbnail_path: str,
                     upload_config_path: str) -> int:
    """
    执行 upload 阶段：仅读取产物目录中的 final_video.mp4 / cover.jpeg / upload_config.pkl，跑上传。
    返回 0 表示全部成功，1 表示至少一个目标失败。
    """
    # 校验产物
    for label, p in [('final_video.mp4', final_video_path),
                     ('cover.jpeg', compressed_thumbnail_path),
                     ('upload_config.pkl', upload_config_path)]:
        if not os.path.exists(p):
            print(f"错误: [upload] 缺少产物文件 {label}: {p}")
            print("请确认 prepare 阶段是否已成功完成，或 --output_dir 是否正确指向产物目录。")
            return 1

    # 加载上传配置
    try:
        with open(upload_config_path, 'rb') as f:
            upload_config = pickle.load(f)
        print(f"[upload] 已加载上传配置: {upload_config}")
    except Exception as e:
        print(f"错误: [upload] 加载 upload_config.pkl 失败: {e}")
        return 1

    upload_success = True

    if args.upload_bilibili:
        print("=" * 60)
        print("[upload] 开始上传到 B 站")
        print("=" * 60)
        sessdata = os.environ.get('BILIBILI_SESSDATA', '')
        bili_jct = os.environ.get('BILIBILI_JCT', '')
        buvid3 = ''  # 硬编码为空字符串

        if not sessdata or not bili_jct:
            print("错误: 未设置BILIBILI_SESSDATA或BILIBILI_JCT环境变量")
            upload_success = False
        else:
            credential = Credential(
                sessdata=sessdata,
                bili_jct=bili_jct,
                buvid3=buvid3
            )
            if not sync(upload_to_bilibili(final_video_path, compressed_thumbnail_path, upload_config, credential)):
                print("[upload] B 站视频上传失败")
                upload_success = False
            else:
                print("[upload] B 站视频上传成功")

    if args.upload_dailymotion:
        print("=" * 60)
        print("[upload] 开始上传到 Dailymotion")
        print("=" * 60)
        if not upload_to_dailymotion(final_video_path, upload_config):
            print("[upload] Dailymotion 视频上传失败")
            upload_success = False
        else:
            print("[upload] Dailymotion 视频上传成功")

    if not args.upload_bilibili and not args.upload_dailymotion:
        print("警告: [upload] 未指定任何上传目标(--upload-bilibili / --upload-dailymotion)，跳过上传")

    return 0 if upload_success else 1


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YouTube视频下载、音轨替换并上传')
    parser.add_argument('--url', required=True, help='YouTube视频URL')
    parser.add_argument('--api_config', required=False, default='',
                        help='API配置文件路径 (upload 阶段不需要，可省略)')
    parser.add_argument('--work_dir', default='/tmp', help='prepare 阶段工作目录（用于下载/解临时文件）')
    parser.add_argument('--output_dir', default='./output',
                        help='prepare 产物输出目录 / upload 读取目录（默认 ./output）')
    parser.add_argument('--cookies', help='YouTube cookies文件路径')
    parser.add_argument('--upload-bilibili', action='store_true', help='上传到B站')
    parser.add_argument('--upload-dailymotion', action='store_true', help='上传到Dailymotion')
    parser.add_argument('--skip-prepare', action='store_true',
                        help='跳过 prepare 阶段，仅运行 upload 阶段（需 --output_dir 中已有产物）')
    parser.add_argument('--skip-upload', action='store_true',
                        help='跳过 upload 阶段，仅运行 prepare 阶段')

    args = parser.parse_args()

    # 参数互斥校验
    if args.skip_prepare and args.skip_upload:
        print("错误: --skip-prepare 和 --skip-upload 不能同时使用")
        return 1
    if args.skip_prepare and not (args.upload_bilibili or args.upload_dailymotion):
        print("错误: --skip-prepare 时必须指定至少一个上传目标 (--upload-bilibili / --upload-dailymotion)")
        return 1
    if not args.skip_prepare and not args.skip_upload and not (args.upload_bilibili or args.upload_dailymotion):
        print("错误: 未指定任何上传目标，请添加 --upload-bilibili 或 --upload-dailymotion")
        return 1
    # prepare 阶段强制要求 --api_config（生成翻译标题/tags 用）
    if not args.skip_prepare and not args.api_config:
        print("错误: prepare 阶段必须指定 --api_config (用于 AI 翻译标题/生成 tags)")
        return 1

    # --- GitHub 工作空间目录 ---
    github_workspace = os.environ.get('GITHUB_WORKSPACE')
    if not github_workspace:
        # 如果不在 GitHub Actions 环境中，则假设是当前目录
        print("警告: 未在GitHub Actions环境中运行，将使用当前目录作为根目录。")
        github_workspace = '.'

    # --- 关键路径（绝对路径） ---
    translated_audio_path = os.path.join(github_workspace, 'subtitles', 'word_level_processed_translated.mp3')
    api_config_file_path = os.path.join(github_workspace, args.api_config) if args.api_config else ''
    cookies_file_path = os.path.join(github_workspace, args.cookies) if args.cookies else None

    print(f"翻译后音频预期路径: {translated_audio_path}")
    if api_config_file_path:
        print(f"API 配置文件预期路径: {api_config_file_path}")
    if cookies_file_path:
        print(f"cookies 文件预期路径: {cookies_file_path}")

    # --- 准备目录 ---
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    # prepare 阶段的中间文件全部在 work_dir；最终产物复制到 output_dir

    # prepare 阶段的中间文件路径
    downloaded_video_path = os.path.join(args.work_dir, 'downloaded_video.mp4')
    thumbnail_path = os.path.join(args.work_dir, 'cover.%(ext)s')

    # 产物目录中的最终文件路径（prepare 写入，upload 读取）
    final_video_path = os.path.join(args.output_dir, 'final_video.mp4')
    compressed_thumbnail_path = os.path.join(args.output_dir, 'cover.jpeg')
    upload_config_path = os.path.join(args.output_dir, 'upload_config.pkl')

    # ============ prepare 阶段 ============
    if not args.skip_prepare:
        # prepare 阶段需要切到 work_dir，方便 yt-dlp 写入 cover.* 在固定位置
        original_cwd = os.getcwd()
        os.chdir(args.work_dir)
        try:
            ok = run_prepare_stage(
                args, github_workspace,
                downloaded_video_path, final_video_path,
                thumbnail_path, compressed_thumbnail_path,
                upload_config_path,
                translated_audio_path, api_config_file_path,
                cookies_file_path
            )
        finally:
            os.chdir(original_cwd)
        if not ok:
            return 1
    else:
        print("[main] --skip-prepare 已指定，跳过 prepare 阶段")

    # ============ upload 阶段 ============
    if args.skip_upload:
        print("[main] --skip-upload 已指定，跳过 upload 阶段")
        return 0

    rc = run_upload_stage(
        args, final_video_path, compressed_thumbnail_path, upload_config_path
    )
    if rc == 0:
        print("[main] 整个流程成功完成")
    else:
        print("[main] 上传阶段失败")
    return rc


if __name__ == "__main__":
    sys.exit(main())
