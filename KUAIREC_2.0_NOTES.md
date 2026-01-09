# KuaiRec 2.0 数据格式说明

## 数据文件位置

```
data/KuaiRec 2.0/data/
├── big_matrix.csv              # 全观测交互数据（长格式）
├── small_matrix.csv            # 稀疏交互数据（长格式）
├── user_features.csv           # 用户特征
├── kuairec_caption_category.csv # 视频特征（标题、描述、类别）
├── item_categories.csv         # 视频类别
├── item_daily_features.csv     # 视频每日特征
└── social_network.csv          # 社交网络数据
```

## 数据格式

### big_matrix.csv / small_matrix.csv

**格式**：长格式（不是矩阵格式）

**列**：
- `user_id`: 用户ID
- `video_id`: 视频ID
- `play_duration`: 播放时长（毫秒）
- `video_duration`: 视频总时长（毫秒）
- `time`: 时间戳（字符串）
- `date`: 日期（YYYYMMDD）
- `timestamp`: Unix时间戳
- `watch_ratio`: 观看比例（play_duration / video_duration）

**数据量**：
- big_matrix.csv: ~1250万行
- small_matrix.csv: ~470万行

### kuairec_caption_category.csv

**列**：
- `video_id`: 视频ID
- `manual_cover_text`: 封面文字
- `caption`: 视频标题/简介
- `topic_tag`: 话题标签
- `first_level_category_id/name`: 一级类别
- `second_level_category_id/name`: 二级类别
- `third_level_category_id/name`: 三级类别

### user_features.csv

**列**：
- `user_id`: 用户ID
- `user_active_degree`: 用户活跃度
- `follow_user_num`: 关注数
- `fans_user_num`: 粉丝数
- 以及其他用户特征...

## 预处理说明

脚本会自动：
1. 检测数据格式（长格式 vs 矩阵格式）
2. 将watch_ratio转换为1-5评分
3. 提取视频标题和描述用于BM25/Cross-Encoder
4. 处理大文件（分块读取）

## 使用方法

```bash
python3 data_prep_kuairec.py --kuairec_dir "data/KuaiRec 2.0"
```

脚本会自动：
- 检测`data/`子目录
- 读取big_matrix.csv（或small_matrix.csv）
- 提取视频特征创建videos.csv
- 生成ratings_kuairec.csv

## 注意事项

1. **大文件处理**：big_matrix.csv有1250万行，读取可能需要几分钟
2. **内存需求**：确保有足够内存（建议>8GB）
3. **评分转换**：watch_ratio会映射到1-5评分
   - 0-0.2 → 1
   - 0.2-0.4 → 2
   - 0.4-0.6 → 3
   - 0.6-0.8 → 4
   - 0.8+ → 5




