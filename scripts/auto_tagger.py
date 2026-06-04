#!/usr/bin/env python3
"""
自动分类标注工具（基于 Caption 关键词规则）

用法：
    # 方式1: 给单个 caption 打标签
    python3 auto_tagger.py --caption "A drone flying over snow covered mountains at sunset"

    # 方式2: 给整个 dataset.parquet 打标签
    python3 auto_tagger.py --parquet ./test_out/dataset.parquet --output tagged.parquet

    # 方式3: 只打印分类规则参考
    python3 auto_tagger.py --show-rules
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow.parquet as pq
import pyarrow as pa


# ═══════════════════════════════════════════════════════════════════════════
# 严格的分类体系定义
# ═══════════════════════════════════════════════════════════════════════════

# 一级分类（互斥，每个切片只能属于一个）
CATEGORIES = {
    "nature": "自然综合场景（无法归入更具体分类时）",
    "landscape": "风景地貌（山脉、沙漠、冰川、峡谷）",
    "wildlife": "野生动物（鸟类、哺乳动物、海洋生物）",
    "ocean": "海洋水域（海浪、海滩、珊瑚、深海）",
    "sky": "天空气象（云、日落、星空、极光）",
    "forest": "森林树木（雨林、秋林、竹林、松林）",
    "water": "淡水景观（瀑布、河流、湖泊、溪流）",
}

# 二级分类（每个一级分类下的子类，互斥）
SUBCATEGORIES = {
    # nature 的子类
    "nature": ["meadow", "valley", "canyon", "general"],
    # landscape 的子类
    "landscape": ["mountain", "desert", "glacier", "volcano", "field", "hill"],
    # wildlife 的子类
    "wildlife": ["bird", "mammal", "marine_life", "insect", "reptile"],
    # ocean 的子类
    "ocean": ["beach", "wave", "coral", "deep_sea", "tide", "coastline"],
    # sky 的子类
    "sky": ["sunset", "sunrise", "cloud", "storm", "star", "rainbow", "aurora"],
    # forest 的子类
    "forest": ["rainforest", "autumn", "bamboo", "pine", "jungle", "winter_forest"],
    # water 的子类
    "water": ["waterfall", "river", "lake", "stream", "hot_spring"],
}

# 标签体系（多选，一个切片可以有多个标签）
TAG_RULES = {
    # 时间
    "dawn": ["dawn", "early morning", "misty morning", "morning mist"],
    "day": ["daytime", "sunny day", "bright day", "clear day", "day"],
    "dusk": ["dusk", "twilight", "evening", "late afternoon"],
    "night": ["night", "dark", "midnight", "nighttime"],
    "golden_hour": ["golden hour", "golden light", "warm light", "golden glow"],
    "blue_hour": ["blue hour", "blue light", "cool blue"],

    # 季节
    "spring": ["spring", "blossom", "cherry blossom", "wildflower", "green meadow"],
    "summer": ["summer", "lush green", "tropical", "palm tree", "sunny beach"],
    "autumn": ["autumn", "fall", "fall colors", "autumn leaves", "orange leaves", "golden leaves"],
    "winter": ["winter", "snow", "snowy", "snow covered", "ice", "frozen", "icicle"],

    # 拍摄方式
    "drone": ["drone", "aerial", "bird eye", "from above", "overhead", "flying over"],
    "timelapse": ["timelapse", "time lapse", "time-lapse", "speed up", "fast motion"],
    "macro": ["macro", "close up", "close-up", "detail", "micro"],
    "underwater": ["underwater", "under water", "beneath the surface", "submerged"],
    "static_tripod": ["static", "fixed", "stationary", "tripod"],
    "handheld": ["handheld", "walking", "following", "tracking"],

    # 天气
    "sunny": ["sunny", "clear sky", "bright sunshine"],
    "cloudy": ["cloudy", "overcast", "cloud cover"],
    "rainy": ["rain", "rainy", "pouring", "drizzle", "raindrop"],
    "foggy": ["fog", "foggy", "mist", "misty", "haze", "hazy"],
    "snowy": ["snow", "snowy", "snowfall", "blizzard"],
    "stormy": ["storm", "stormy", "thunder", "lightning", "hurricane"],

    # 光线
    "backlight": ["backlight", "backlit", "silhouette", "against the sun"],
    "sidelight": ["sidelight", "side light", "rim light"],
    "soft_light": ["soft light", "diffused light", "gentle light"],
    "harsh_light": ["harsh light", "strong light", "direct sunlight"],

    # 运动
    "slow_motion": ["slow motion", "slowly", "gentle movement"],
    "fast_motion": ["fast", "rapid", "swift", "quick movement"],
    "panning": ["pan", "panning", "horizontal movement"],
    "tilt": ["tilt", "tilting", "vertical movement"],
    "zoom": ["zoom", "zooming", "push in", "pull out"],
}

# 一级分类判定规则（基于关键词，优先级从高到低）
CATEGORY_RULES = [
    ("water", ["waterfall", "river", "lake", "stream", "creek", "brook", "hot spring", "geyser"]),
    ("wildlife", ["animal", "bird", "eagle", "deer", "bear", "wolf", "fox", "lion", "tiger", "elephant", "dolphin", "whale", "shark", "fish", "turtle", "butterfly", "bee", "insect"]),
    ("ocean", ["ocean", "sea", "beach", "wave", "coast", "shore", "tide", "coral", "reef", "underwater", "marine"]),
    ("forest", ["forest", "wood", "jungle", "rainforest", "tree", "pine", "bamboo", "redwood", "sequoia", "grove"]),
    ("sky", ["sky", "cloud", "sunset", "sunrise", "star", "galaxy", "milky way", "aurora", "rainbow", "storm", "thunder"]),
    ("landscape", ["mountain", "peak", "summit", "valley", "canyon", "cliff", "mesa", "plateau", "desert", "dune", "savanna", "tundra", "glacier", "iceberg", "volcano"]),
    ("nature", ["nature", "meadow", "field", "prairie", "grassland", "flower", "garden", "natural"]),
]

# 二级分类判定规则（在确定一级分类后，匹配子类）
SUBCATEGORY_KEYWORDS = {
    # landscape
    "mountain": ["mountain", "peak", "summit", "ridge", "alp", "alpine", "rocky mountain", "snowy mountain"],
    "desert": ["desert", "dune", "sand", "arid", "sahara", "gobi"],
    "glacier": ["glacier", "ice field", "ice cap", "frozen landscape"],
    "volcano": ["volcano", "lava", "eruption", "crater", "magma"],
    "field": ["field", "farmland", "crop", "wheat", "rice paddy"],
    "hill": ["hill", "rolling hill", "foothill"],

    # wildlife
    "bird": ["bird", "eagle", "hawk", "owl", "parrot", "penguin", "flamingo", "swan", "flock"],
    "mammal": ["deer", "bear", "wolf", "fox", "lion", "tiger", "elephant", "giraffe", "zebra", "monkey", "ape"],
    "marine_life": ["dolphin", "whale", "shark", "seal", "sea lion", "otter", "jellyfish", "octopus"],
    "insect": ["butterfly", "bee", "dragonfly", "ant", "beetle"],
    "reptile": ["snake", "lizard", "crocodile", "turtle", "tortoise"],

    # ocean
    "beach": ["beach", "shore", "coast", "sandy", "palm tree", "tropical beach"],
    "wave": ["wave", "surf", "crashing wave", "tidal wave"],
    "coral": ["coral", "reef", "coral reef"],
    "deep_sea": ["deep sea", "abyss", "open ocean"],
    "tide": ["tide", "tidal", "low tide", "high tide"],
    "coastline": ["coastline", "coastal", "seaside", "shoreline"],

    # sky
    "sunset": ["sunset", "sun down", "evening glow", "twilight"],
    "sunrise": ["sunrise", "sun up", "dawn", "daybreak", "morning sun"],
    "cloud": ["cloud", "cloudy", "cumulus", "stratus", "storm cloud"],
    "storm": ["storm", "thunderstorm", "hurricane", "typhoon", "tornado"],
    "star": ["star", "galaxy", "milky way", "nebula", "starry", "night sky"],
    "rainbow": ["rainbow"],
    "aurora": ["aurora", "northern light", "southern light"],

    # forest
    "rainforest": ["rainforest", "tropical forest", "jungle"],
    "autumn": ["autumn forest", "fall forest", "autumn wood", "fall wood"],
    "bamboo": ["bamboo", "bamboo forest"],
    "pine": ["pine", "pine forest", "conifer", "evergreen"],
    "jungle": ["jungle", "dense forest"],
    "winter_forest": ["snowy forest", "winter wood", "snow covered tree"],

    # water
    "waterfall": ["waterfall", "cascade", "falls", "cataract"],
    "river": ["river", "stream", "brook", "creek", "flowing water"],
    "lake": ["lake", "pond", "reservoir", "lagoon"],
    "stream": ["stream", "creek", "brook", "rivulet"],
    "hot_spring": ["hot spring", "geyser", "thermal"],

    # nature fallback
    "meadow": ["meadow", "grassland", "prairie", "pasture"],
    "valley": ["valley", "vale", "gorge", "ravine"],
    "canyon": ["canyon", "ravine", "glen"],
    "general": ["nature", "scenic", "landscape", "outdoor"],
}


def normalize_text(text: str) -> str:
    """标准化文本用于匹配。"""
    return text.lower().strip()


def classify_by_caption(caption: str) -> Tuple[str, str, List[str]]:
    """
    根据 caption 自动分类。

    返回: (category, subcategory, tags)
    """
    text = normalize_text(caption)

    # 1. 判定一级分类（按优先级匹配）
    category = "nature"  # 默认
    for cat, keywords in CATEGORY_RULES:
        if any(kw in text for kw in keywords):
            category = cat
            break

    # 2. 判定二级分类
    subcategory = "general"  # 默认
    if category in SUBCATEGORIES:
        candidates = SUBCATEGORIES[category]
        for sub in candidates:
            keywords = SUBCATEGORY_KEYWORDS.get(sub, [])
            if any(kw in text for kw in keywords):
                subcategory = sub
                break

    # 3. 提取标签（多选）
    tags = []
    for tag, keywords in TAG_RULES.items():
        if any(kw in text for kw in keywords):
            tags.append(tag)

    return category, subcategory, tags


def tag_parquet(input_path: str, output_path: str):
    """给 parquet 文件添加分类列。"""
    table = pq.read_table(input_path)
    df = table.to_pandas()

    if "caption" not in df.columns:
        print("[ERROR] parquet 中没有 caption 列")
        sys.exit(1)

    categories = []
    subcategories = []
    tags_list = []

    for caption in df["caption"]:
        cat, sub, tags = classify_by_caption(str(caption))
        categories.append(cat)
        subcategories.append(sub)
        tags_list.append(json.dumps(tags))

    df["category"] = categories
    df["subcategory"] = subcategories
    df["tags"] = tags_list

    new_table = pa.Table.from_pandas(df)
    pq.write_table(new_table, output_path)

    print(f"[INFO] Tagged {len(df)} clips")
    print(f"[INFO] Saved to {output_path}")

    # 打印分布统计
    print("\nCategory distribution:")
    for cat, count in df["category"].value_counts().items():
        print(f"  {cat}: {count}")

    print("\nTop tags:")
    all_tags = []
    for t in tags_list:
        all_tags.extend(json.loads(t))
    from collections import Counter
    for tag, count in Counter(all_tags).most_common(10):
        print(f"  {tag}: {count}")


def show_rules():
    """打印分类规则参考。"""
    print("=" * 60)
    print("SN70 数据集分类体系")
    print("=" * 60)

    print("\n【一级分类】(互斥)")
    for cat, desc in CATEGORIES.items():
        print(f"  {cat:12s} - {desc}")

    print("\n【二级分类】(每个一级分类下的子类)")
    for cat, subs in SUBCATEGORIES.items():
        print(f"  {cat}: {', '.join(subs)}")

    print("\n【标签体系】(多选)")
    tag_groups = {
        "时间": ["dawn", "day", "dusk", "night", "golden_hour", "blue_hour"],
        "季节": ["spring", "summer", "autumn", "winter"],
        "拍摄方式": ["drone", "timelapse", "macro", "underwater", "static_tripod", "handheld"],
        "天气": ["sunny", "cloudy", "rainy", "foggy", "snowy", "stormy"],
        "光线": ["backlight", "sidelight", "soft_light", "harsh_light"],
        "运动": ["slow_motion", "fast_motion", "panning", "tilt", "zoom"],
    }
    for group, tags in tag_groups.items():
        print(f"  {group}: {', '.join(tags)}")


def main():
    parser = argparse.ArgumentParser(description="Auto-tagger for SN70 dataset")
    parser.add_argument("--caption", default="", help="Single caption to classify")
    parser.add_argument("--parquet", default="", help="Parquet file to tag")
    parser.add_argument("--output", default="tagged.parquet", help="Output parquet path")
    parser.add_argument("--show-rules", action="store_true", help="Show classification rules")
    args = parser.parse_args()

    if args.show_rules:
        show_rules()
        return

    if args.caption:
        cat, sub, tags = classify_by_caption(args.caption)
        print(f"Caption: {args.caption}")
        print(f"Category:    {cat}")
        print(f"Subcategory: {sub}")
        print(f"Tags:        {tags}")
        return

    if args.parquet:
        tag_parquet(args.parquet, args.output)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
