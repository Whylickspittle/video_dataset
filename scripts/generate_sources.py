#!/usr/bin/env python3
"""
自动生成 sources.txt

用法：
    # 方式1: 搜索关键词获取视频链接
    python3 generate_sources.py --search "4K nature relaxing" --count 20

    # 方式2: 从 YouTube Playlist 提取所有视频
    python3 generate_sources.py --playlist "PLxxx..." --output sources.txt

    # 方式3: 从频道获取最新视频
    python3 generate_sources.py --channel "UCxxx..." --count 30

    # 方式4: 直接指定多个关键词批量搜索
    python3 generate_sources.py --multi-search "nature,wildlife,ocean" --count 10
"""

import argparse
import json
import subprocess
import sys
from urllib.parse import parse_qs, urlparse


def run_ytdlp(*args, timeout: int = 60):
    """运行 yt-dlp 命令。"""
    cmd = ["yt-dlp", "--no-check-certificates"] + list(args)
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )
    if result.returncode != 0:
        print(f"[WARN] yt-dlp error: {result.stderr[:200]}", file=sys.stderr)
    return result.stdout.strip(), result.returncode


def search_videos(query: str, count: int = 20) -> list[str]:
    """搜索 YouTube 视频，返回视频链接列表。"""
    print(f"[INFO] Searching: '{query}' (target: {count})")
    stdout, rc = run_ytdlp(
        "ytsearch{}:{}".format(count, query),
        "--flat-playlist",
        "--print", "%(webpage_url)s",
        "--playlist-end", str(count),
    )
    if rc != 0 or not stdout:
        print(f"[ERROR] Search failed for: {query}")
        return []

    urls = [line.strip() for line in stdout.splitlines() if line.strip().startswith("http")]
    print(f"[INFO] Found {len(urls)} videos")
    return urls


def extract_from_playlist(playlist_url: str, count: int = 50) -> list[str]:
    """从 Playlist 提取视频链接。"""
    print(f"[INFO] Extracting from playlist: {playlist_url}")
    stdout, rc = run_ytdlp(
        playlist_url,
        "--flat-playlist",
        "--print", "%(webpage_url)s",
        "--playlist-end", str(count),
    )
    if rc != 0 or not stdout:
        return []

    urls = [line.strip() for line in stdout.splitlines() if line.strip().startswith("http")]
    print(f"[INFO] Found {len(urls)} videos from playlist")
    return urls


def extract_from_channel(channel_url: str, count: int = 30) -> list[str]:
    """从频道提取视频链接。"""
    print(f"[INFO] Extracting from channel: {channel_url}")
    # 频道视频通常需要获取 uploads playlist
    stdout, rc = run_ytdlp(
        channel_url,
        "--flat-playlist",
        "--print", "%(webpage_url)s",
        "--playlist-end", str(count),
    )
    if rc != 0 or not stdout:
        return []

    urls = [line.strip() for line in stdout.splitlines() if line.strip().startswith("http")]
    print(f"[INFO] Found {len(urls)} videos from channel")
    return urls


def filter_hd_videos(urls: list[str], min_height: int = 1080) -> list[str]:
    """
    过滤高清视频。逐个检查视频分辨率。
    注意：这会调用 yt-dlp 获取每个视频的信息，比较慢。
    """
    filtered = []
    print(f"[INFO] Filtering {len(urls)} videos for HD ({min_height}p+)...")
    for i, url in enumerate(urls):
        stdout, rc = run_ytdlp(
            "--dump-json", "--no-check-certificates",
            url,
            timeout=30,
        )
        if rc != 0 or not stdout:
            continue
        try:
            info = json.loads(stdout)
            height = info.get("height", 0)
            duration = info.get("duration", 0)
            # 过滤条件：分辨率达标 + 时长大于 20 秒（排除 Shorts）
            if height >= min_height and duration >= 20:
                print(f"  [{i+1}/{len(urls)}] OK: {info.get('title', '?')[:40]} ({height}p, {duration//60}min)")
                filtered.append(url)
            else:
                print(f"  [{i+1}/{len(urls)}] SKIP: height={height}, duration={duration}s")
        except Exception as exc:
            print(f"  [{i+1}/{len(urls)}] ERROR: {exc}")
    return filtered


def deduplicate(urls: list[str]) -> list[str]:
    """去重（基于 video ID）。"""
    seen = set()
    result = []
    for url in urls:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if host == "youtu.be":
            vid = parsed.path.strip("/").split("?")[0]
        elif "youtube.com" in host:
            query = parse_qs(parsed.query)
            vid = query.get("v", [parsed.path])[-1]
        else:
            vid = url
        if vid and vid not in seen:
            seen.add(vid)
            result.append(url)
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate sources.txt for miner")
    parser.add_argument("--search", default="", help="Search query, e.g. '4K nature relaxing'")
    parser.add_argument("--playlist", default="", help="YouTube playlist URL")
    parser.add_argument("--channel", default="", help="YouTube channel URL")
    parser.add_argument("--multi-search", default="", help="Multiple queries separated by comma")
    parser.add_argument("--count", type=int, default=20, help="Max videos per source")
    parser.add_argument("--filter-hd", action="store_true", help="Filter HD videos (slow)")
    parser.add_argument("--min-height", type=int, default=1080, help="Min resolution")
    parser.add_argument("--output", default="sources.txt", help="Output file")
    args = parser.parse_args()

    all_urls = []

    # 方式1: 搜索
    if args.search:
        all_urls.extend(search_videos(args.search, args.count))

    # 方式2: Playlist
    if args.playlist:
        all_urls.extend(extract_from_playlist(args.playlist, args.count))

    # 方式3: 频道
    if args.channel:
        all_urls.extend(extract_from_channel(args.channel, args.count))

    # 方式4: 多关键词批量搜索
    if args.multi_search:
        queries = [q.strip() for q in args.multi_search.split(",")]
        for q in queries:
            all_urls.extend(search_videos(q, args.count))

    if not all_urls:
        print("[ERROR] No URLs found. Please provide --search, --playlist, --channel or --multi-search")
        sys.exit(1)

    # 去重
    all_urls = deduplicate(all_urls)
    print(f"[INFO] Total unique URLs: {len(all_urls)}")

    # HD 过滤（可选，很慢）
    if args.filter_hd:
        all_urls = filter_hd_videos(all_urls, args.min_height)
        print(f"[INFO] HD videos after filter: {len(all_urls)}")

    # 写入文件
    with open(args.output, "w") as f:
        for url in all_urls:
            f.write(url + "\n")

    print(f"[INFO] Saved {len(all_urls)} URLs to {args.output}")
    print("[INFO] Preview:")
    for url in all_urls[:5]:
        print(f"  {url}")
    if len(all_urls) > 5:
        print(f"  ... and {len(all_urls) - 5} more")


if __name__ == "__main__":
    main()
