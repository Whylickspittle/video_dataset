#!/usr/bin/env python3
"""
矿工切片管理数据库初始化脚本。

创建 SQLite 数据库，用于管理：
- 视频源信息（含三方链接）
- 切片记录
- 分类标签
- Interval 提交历史
- 全局去重索引
"""

import sqlite3
from pathlib import Path


def init_database(db_path: str = "miner.db") -> sqlite3.Connection:
    """初始化矿工管理数据库，创建所有表和索引。"""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # ── 1. 视频源表 ──
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            -- 核心标识
            source_video_id TEXT NOT NULL,              -- YouTube 视频 ID
            source_video_url TEXT NOT NULL,             -- YouTube 原始链接
            source_platform TEXT DEFAULT 'youtube',     -- 平台：youtube/pexels/vimeo/self 等

            -- 三方链接（用户自定义备份/镜像）
            third_party_url TEXT,                       -- 备用下载链接
            third_party_platform TEXT,                  -- 备用平台名称
            local_backup_path TEXT,                     -- 本地备份路径

            -- 元数据（下载前预检获取）
            title TEXT,
            channel TEXT,
            description TEXT,
            upload_date TEXT,                           -- YYYY-MM-DD

            -- 技术指标
            src_width INTEGER,
            src_height INTEGER,
            src_fps REAL,
            src_tbr INTEGER,                            -- 码率 kbps
            src_duration REAL,                          -- 总时长（秒）
            src_vcodec TEXT,
            src_acodec TEXT,
            src_filesize INTEGER,                       -- 估算文件大小（字节）

            -- 内容分类
            category TEXT DEFAULT 'nature',             -- 主分类：nature/landscape/wildlife/ocean/sky/urban
            subcategory TEXT,                           -- 子分类：forest/mountain/waterfall/beach/desert/snow
            tags TEXT,                                  -- 标签 JSON: ["drone", "aerial", "timelapse"]

            -- 质量评分（用户自评，0-1）
            visual_quality REAL,                        -- 画面清晰度
            motion_quality REAL,                        -- 运动流畅度
            lighting_quality REAL,                      -- 光线质量

            -- 下载管理
            local_path TEXT,                            -- 本地文件路径
            download_status TEXT DEFAULT 'pending',     -- pending/downloaded/failed/skipped
            download_attempts INTEGER DEFAULT 0,        -- 下载重试次数
            download_error TEXT,                        -- 最后一次错误信息

            -- 使用统计
            total_segments INTEGER,                     -- 可切总段数 = floor(duration / 5.04)
            used_segments INTEGER DEFAULT 0,            -- 已生成多少段
            submitted_segments INTEGER DEFAULT 0,       -- 已提交多少段

            -- 时间戳
            discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            downloaded_at TIMESTAMP,
            last_checked_at TIMESTAMP                   -- 上次验证时间
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_sources_video_id ON sources(source_video_id);
        CREATE INDEX IF NOT EXISTS idx_sources_category ON sources(category, subcategory);
        CREATE INDEX IF NOT EXISTS idx_sources_status ON sources(download_status);
        CREATE INDEX IF NOT EXISTS idx_sources_quality ON sources(visual_quality);
    """)

    # ── 2. 切片表 ──
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS clips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            -- 核心标识
            clip_id TEXT UNIQUE NOT NULL,               -- 确定性哈希：sha256(source_id + start_sec)
            source_id INTEGER NOT NULL,

            -- 所属 interval
            interval_id INTEGER,                        -- 提交到哪个 interval
            batch_id TEXT,                              -- 批次号（用于本地分组管理）

            -- 时间位置
            start_sec REAL NOT NULL,
            end_sec REAL,                               -- start_sec + duration
            duration_sec REAL NOT NULL DEFAULT 5.04,
            segment_index INTEGER,                      -- 该视频第几段（0-based）

            -- 文件路径（本地）
            clip_local_path TEXT,
            frame_local_path TEXT,

            -- SHA256（提交用）
            clip_sha256 TEXT,
            frame_sha256 TEXT,

            -- 实际探测参数（下载后 ffprobe 确认）
            actual_width INTEGER,
            actual_height INTEGER,
            actual_fps REAL,
            actual_num_frames INTEGER,
            actual_duration REAL,

            -- 内容
            caption TEXT,                               -- LLM 生成的描述
            caption_model TEXT,                         -- 使用的模型：gpt-4o-mini/gemini/etc
            caption_confidence REAL,                    -- LLM 置信度（如果有）

            -- 用户自定义元数据
            third_party_url TEXT,                       -- 该切片的三方备份链接
            notes TEXT,                                 -- 用户备注
            tags TEXT,                                  -- 标签 JSON

            -- 质量自评（0-1）
            self_quality_score REAL DEFAULT 0.5,
            visual_score REAL,                          -- 画面评分
            motion_score REAL,                          -- 运动评分
            uniqueness_score REAL,                      -- 独特性评分（与其他片段的差异度）

            -- 提交状态
            submission_status TEXT DEFAULT 'draft',     -- draft/selected/submitted/accepted/rejected
            validator_failures TEXT,                    -- JSON 数组，失败原因
            validator_notes TEXT,                       -- 验证者备注

            -- 时间戳
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            submitted_at TIMESTAMP,
            validated_at TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_clips_source ON clips(source_id);
        CREATE INDEX IF NOT EXISTS idx_clips_status ON clips(submission_status);
        CREATE INDEX IF NOT EXISTS idx_clips_interval ON clips(interval_id);
        CREATE INDEX IF NOT EXISTS idx_clips_batch ON clips(batch_id);
        CREATE INDEX IF NOT EXISTS idx_clips_score ON clips(self_quality_score);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_clips_unique ON clips(source_id, start_sec);
    """)

    # ── 3. Interval 提交表 ──
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS intervals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interval_num INTEGER UNIQUE NOT NULL,       -- interval_id（链上编号）
            miner_hotkey TEXT,

            -- 提交文件路径
            dataset_parquet_path TEXT,
            manifest_json_path TEXT,

            -- 统计
            total_clips INTEGER DEFAULT 400,
            accepted_clips INTEGER,
            rejected_clips INTEGER,

            -- 分类分布（提交时的统计）
            category_distribution TEXT,                 -- JSON: {"nature": 150, "ocean": 100, ...}

            -- Validator 结果
            validator_accepted BOOLEAN,
            validator_score REAL,
            validator_rank INTEGER,                     -- Top-K 排名

            -- 链上状态
            weights_received REAL,                      -- 获得的权重值
            block_submitted INTEGER,                    -- 提交区块高度

            -- 时间戳
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            submitted_at TIMESTAMP,
            result_received_at TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_intervals_num ON intervals(interval_num);
        CREATE INDEX IF NOT EXISTS idx_intervals_score ON intervals(validator_score);
    """)

    # ── 4. 全局去重索引（自己维护，跨周期去重用） ──
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS global_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_video_id TEXT NOT NULL,
            source_video_url TEXT,
            start_sec REAL NOT NULL,
            interval_id INTEGER,
            clip_id TEXT,
            used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_video_id, start_sec)
        );

        CREATE INDEX IF NOT EXISTS idx_global_source ON global_index(source_video_id);
        CREATE INDEX IF NOT EXISTS idx_global_url ON global_index(source_video_url);
    """)

    # ── 5. 分类字典表（可选，用于规范分类） ──
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,                  -- nature/landscape/wildlife/ocean/sky/urban
            description TEXT,
            priority INTEGER DEFAULT 0,                 -- 优先级（用于自动平衡）
            target_ratio REAL DEFAULT 0.166             -- 目标占比（默认 1/6 ≈ 16.6%）
        );

        -- 初始化默认分类
        INSERT OR IGNORE INTO categories (name, description, priority, target_ratio) VALUES
            ('nature', 'General nature scenes', 1, 0.20),
            ('landscape', 'Wide landscape shots', 2, 0.20),
            ('wildlife', 'Animals in natural habitat', 3, 0.15),
            ('ocean', 'Ocean, sea, water bodies', 4, 0.15),
            ('sky', 'Sky, clouds, celestial', 5, 0.15),
            ('forest', 'Forests, woods, trees', 6, 0.15);

        CREATE TABLE IF NOT EXISTS subcategories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            UNIQUE(category_id, name)
        );

        -- 初始化子分类
        INSERT OR IGNORE INTO subcategories (category_id, name, description)
        SELECT id, 'mountain', 'Mountain ranges, peaks, alpine' FROM categories WHERE name = 'landscape'
        UNION ALL SELECT id, 'waterfall', 'Waterfalls, cascades' FROM categories WHERE name = 'nature'
        UNION ALL SELECT id, 'beach', 'Beaches, coastlines' FROM categories WHERE name = 'ocean'
        UNION ALL SELECT id, 'desert', 'Deserts, sand dunes' FROM categories WHERE name = 'landscape'
        UNION ALL SELECT id, 'snow', 'Snow, ice, winter scenes' FROM categories WHERE name = 'nature'
        UNION ALL SELECT id, 'drone', 'Aerial drone footage' FROM categories WHERE name = 'landscape'
        UNION ALL SELECT id, 'timelapse', 'Timelapse sequences' FROM categories WHERE name = 'sky'
        UNION ALL SELECT id, 'sunset', 'Sunset, sunrise, golden hour' FROM categories WHERE name = 'sky';
    """)

    # ── 6. 操作日志表（用于审计和调试） ──
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS operation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            operation TEXT NOT NULL,                    -- download/slice/validate/submit
            entity_type TEXT,                           -- source/clip/interval
            entity_id TEXT,
            status TEXT,                                -- success/failure
            details TEXT,                               -- JSON 详情
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_logs_op ON operation_logs(operation);
        CREATE INDEX IF NOT EXISTS idx_logs_time ON operation_logs(created_at);
    """)

    conn.commit()
    print(f"Database initialized: {db_path}")
    print("Tables created: sources, clips, intervals, global_index, categories, subcategories, operation_logs")
    return conn


def verify_schema(conn: sqlite3.Connection) -> dict:
    """验证数据库结构，返回各表行数。"""
    cursor = conn.cursor()
    tables = ["sources", "clips", "intervals", "global_index",
              "categories", "subcategories", "operation_logs"]
    result = {}
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        result[table] = cursor.fetchone()[0]
    return result


if __name__ == "__main__":
    import sys
    db_file = sys.argv[1] if len(sys.argv) > 1 else "miner.db"
    conn = init_database(db_file)
    counts = verify_schema(conn)
    print("\nTable row counts:")
    for table, count in counts.items():
        print(f"  {table:20s}: {count}")
    conn.close()
    print("\nDone. Run this script again anytime to verify the database.")
