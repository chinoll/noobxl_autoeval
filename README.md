# NoobXL Auto Evaluation

本仓库用于自动化测试 Stable Diffusion XL (SDXL) 及相关模型的能力，包括画师分类和 danbooru 分类。

---

## 项目简介
本项目通过自动化脚本，对 SDXL 及衍生模型进行批量评测，支持画师分类和 danbooru 分类模型。

---

## 环境依赖
- 推荐 Python 3.10 及以上版本

---

## 快速开始
主评测脚本：`autoeval_pipeline.py`

```bash
python autoeval_pipeline.py \
  --input /path/to/your/dir \
  --artist-class-csv /path/to/your/artist/class/csv \
  --artist-model-checkpoint /path/to/your/artist/model \
  --artist_threshold 0.4 \
  --danbooru-model-name repo_od \
  --answer prompt.json
```

### 参数说明
| 参数 | 说明 |
| ---- | ---- |
| `--input` | 待评测图片文件夹 |
| `--artist-class-csv` | 画师分类标签 CSV 文件 |
| `--artist-model-checkpoint` | 画师分类模型 checkpoint 路径 |
| `--artist_threshold` | 画师分类置信度阈值 |
| `--danbooru-model-name` | danbooru 分类模型的 HuggingFace 仓库名 |
| `--answer` | 评测用的标准答案 JSON |

---

## 支持的模型
- 画师分类：`lsnet_xl_artist_448`
- danbooru：[danbooru 分类模型](https://huggingface.co/spaces/animetimm/dbv4-full-ranklist)