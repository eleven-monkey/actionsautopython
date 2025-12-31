#!/usr/bin/env python3
"""
YouTube视频下载、音轨替换并上传到B站的脚本
适用于GitHub Actions环境
"""

import os
import sys
import argparse
import glob
import subprocess
import pickle
import time
import json
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
    """下载YouTube视频（仅视频流）"""
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
        
        ydl_opts = {
            'format': format_selector,
            'outtmpl': output_path,
            'noplaylist': True,  # 不下载播放列表
        }

        # 添加cookies支持
        if cookies_file and os.path.exists(cookies_file):
            ydl_opts['cookiefile'] = cookies_file
            print(f"使用cookies文件: {cookies_file}")

        print(f"正在从URL下载视频: {url}")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print(f"视频已成功下载到 {output_path}")
            return True
        except yt_dlp.DownloadError as e:
            print(f"使用格式 {format_selector} 下载失败: {e}")
            if "Requested format is not available" in str(e):
                # 如果格式不可用，尝试下一个格式选择器
                continue
            else:
                # 如果是其他错误，直接返回失败
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
    """下载YouTube视频的缩略图"""
    ydl_opts = {
        'skip_download': True,  # 仅提取信息，不下载视频/音频
        'writethumbnail': True,  # 写入缩略图
        'outtmpl': output_path,
        'noplaylist': True,  # 不处理播放列表
        'extractor-args': "youtube:player_js_version=actual"
    }

    # 添加cookies支持
    if cookies_file and os.path.exists(cookies_file):
        ydl_opts['cookiefile'] = cookies_file

    print(f"正在下载URL的缩略图: {url}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
        print(f"缩略图已成功下载到 {output_path}")
        return True
    except yt_dlp.DownloadError as e:
        print(f"缩略图下载错误: {e}")
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
    """使用AI根据视频标题生成相关标签。优先使用api_config中的 model_name/api_key/url，缺失则使用环境变量。"""
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
    
    if not api_url or not api_key:
        print("警告: AI api_url 或 api_key 未配置（api_config 或 环境变量），将返回默认标签")
        return "科普"

    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        response_data = response.json()
        # 兼容常见的返回结构
        tags = None
        try:
            tags = response_data['choices'][0]['message']['content'].strip()
        except Exception:
            # 其它模型/接口的可能字段
            try:
                tags = response_data.get('data', [])[0].get('content', '').strip()
            except Exception:
                tags = None

        if not tags:
            print("警告: AI 返回中未找到标签内容，使用默认标签")
            return "科普"

        return tags
    except Exception as e:
        print(f"生成标签时出错: {e}")
        return "科普"  # 返回默认标签

def generate_upload_config(youtube_url: str, api_config_file: str, output_path: str, cookies_file: str = None) -> Dict[str, Any]:
    """生成上传配置文件，包括翻译标题。优先使用 JSON 文件中的 model/url/api_key，否则回退到环境变量。"""
    # 使用load_api_config函数读取API配置（支持回退到环境变量）
    api_config = load_api_config(api_config_file) or {}
    if not api_config:
        print("警告: 未从文件或环境中加载到AI配置，AI翻译将会回退到原始标题。")

    # 获取YouTube视频标题
    ydl_opts = {
        'skip_download': True,  # 跳过下载
        'noplaylist': True,
    }
    
    # cookies支持
    if cookies_file and os.path.exists(cookies_file):
        ydl_opts['cookiefile'] = cookies_file
        print(f"获取标题时使用cookies文件: {cookies_file}")
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            title = info_dict.get('title', None)
            print(f"视频标题: {title}")
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
        try:
            response = requests.post(api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            response_data = response.json()
            try:
                translated_title_with_markdown = response_data['choices'][0]['message']['content']
            except Exception:
                # 兼容不同返回格式
                translated_title_with_markdown = response_data.get('data', [])[0].get('content', title) if response_data.get('data') else title

            translated_title = translated_title_with_markdown.replace('**', '').strip() if translated_title_with_markdown else title
        except Exception as e:
            print(f"翻译标题时出错: {e}，将使用原始标题作为翻译结果")
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


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YouTube视频下载、音轨替换并上传到B站')
    parser.add_argument('--url', required=True, help='YouTube视频URL')
    parser.add_argument('--api_config', required=True, help='API配置文件路径')
    parser.add_argument('--work_dir', default='/tmp', help='工作目录')
    parser.add_argument('--cookies', help='YouTube cookies文件路径')
    
    args = parser.parse_args()
    
    # --- 关键修正：获取GitHub工作空间目录 ---
    github_workspace = os.environ.get('GITHUB_WORKSPACE')
    if not github_workspace:
        # 如果不在 GitHub Actions 环境中，则假设是当前目录
        print("警告: 未在GitHub Actions环境中运行，将使用当前目录作为根目录。")
        github_workspace = '.'

    # --- 关键修正：获取所有文件的绝对路径 ---
    translated_audio_path = os.path.join(github_workspace, 'subtitles', 'word_level_processed_translated.mp3')
    api_config_file_path = os.path.join(github_workspace, args.api_config)
    
    if args.cookies:
        cookies_file_path = os.path.join(github_workspace, args.cookies)
    else:
        cookies_file_path = None
    
    print(f"正在查找翻译后的音频文件，预期路径: {translated_audio_path}")
    print(f"正在查找API配置文件，预期路径: {api_config_file_path}")
    if cookies_file_path:
        print(f"正在查找cookies文件，预期路径: {cookies_file_path}")
    
    # 确保工作目录存在
    os.makedirs(args.work_dir, exist_ok=True)
    os.chdir(args.work_dir)
    
    # 定义文件路径
    downloaded_video_path = os.path.join(args.work_dir, 'downloaded_video.mp4')
    final_video_path = os.path.join(args.work_dir, 'final_video.mp4')
    thumbnail_path = os.path.join(args.work_dir, 'cover.%(ext)s')
    compressed_thumbnail_path = os.path.join(args.work_dir, 'cover.jpeg')
    upload_config_path = os.path.join(args.work_dir, 'upload_config.pkl')
    
    # 检查翻译后的音频文件是否存在
    if not os.path.exists(translated_audio_path):
        print(f"错误: 找不到翻译后的音频文件 {translated_audio_path}")
        # 打印调试信息
        print(f"当前工作目录: {os.getcwd()}")
        parent_dir = os.path.dirname(translated_audio_path)
        if os.path.exists(parent_dir):
            print(f"文件所在目录 '{parent_dir}' 的内容: {os.listdir(parent_dir)}")
        else:
            print(f"文件所在目录 '{parent_dir}' 本身不存在。")
        return 1
    
    # 清理已存在的文件
    clean_existing_files('*.mp4')
    clean_existing_files('cover.*')
    
    # 步骤1: 生成上传配置（包括翻译标题）
    # --- 关键修正：传递cookies文件路径 ---
    upload_config = generate_upload_config(args.url, api_config_file_path, upload_config_path, cookies_file_path)
    if not upload_config:
        print("生成上传配置失败，终止流程")
        return 1
    
    # 步骤2: 下载YouTube视频
    if not download_youtube_video(args.url, downloaded_video_path, cookies_file_path):
        print("视频下载失败，终止流程")
        return 1
    
    # 步骤3: 替换音轨
    if not replace_audio_track(downloaded_video_path, translated_audio_path, final_video_path):
        print("音轨替换失败，终止流程")
        return 1
    
    # 步骤4: 下载缩略图
    if not download_thumbnail(args.url, thumbnail_path, cookies_file_path):
        print("缩略图下载失败，终止流程")
        return 1
    
    # 步骤5: 压缩缩略图
    # 查找实际下载的缩略图文件（可能有不同的扩展名）
    thumbnail_files = glob.glob('cover.*')
    if not thumbnail_files:
        print("找不到下载的缩略图文件")
        return 1
    
    if not convert_and_compress_to_jpeg(thumbnail_files[0], compressed_thumbnail_path):
        print("缩略图压缩失败，终止流程")
        return 1
    
    # 步骤6: 上传到B站
    # 从环境变量中读取B站凭证
    sessdata = os.environ.get('BILIBILI_SESSDATA', '')
    bili_jct = os.environ.get('BILIBILI_JCT', '')
    buvid3 = ''  # 硬编码为空字符串
    
    if not sessdata or not bili_jct:
        print("错误: 未设置BILIBILI_SESSDATA或BILIBILI_JCT环境变量")
        return 1
    
    credential = Credential(
        sessdata=sessdata,
        bili_jct=bili_jct,
        buvid3=buvid3
    )
    
    # 还原为原始的sync调用
    if not sync(upload_to_bilibili(final_video_path, compressed_thumbnail_path, upload_config, credential)):
        print("视频上传失败")
        return 1
    
    print("整个流程成功完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
