# NoobXL auto eval

该仓库用于自动化测试 SDXL 及相关模型的能力，包括画师分类和 danbooru 分类。

## 项目简介
本项目通过自动化脚本，对 Stable Diffusion XL (SDXL) 及衍生模型进行批量评测，支持画师分类和 danbooru 分类模型。

## 依赖安装
建议使用 Python 3.10 及以上版本。

## 使用方法
主评测脚本：`autoeval_pipeline.py`

```bash
python autoeval_pipeline.py --artist-model-checkpoint /path/to/your/model \
    --artist_threshold <threshold> \
    --input <input_dir> \
    --artist-class-csv /path/to/your/csv \
    --danbooru-model-name <repo_id>
```

参数说明：
- `--artist-model-checkpoint`：画师分类模型的 checkpoint 路径。
- `--artist_threshold`：画师分类置信度阈值。
- `--input`：待评测图片文件夹。
- `--artist-class-csv`：画师分类标签 CSV 文件。
- `--danbooru-model-name`：danbooru 分类模型的 HuggingFace 仓库名。

## 输出结果
- 评测结果将保存在 `output/inference/` 和 `result/` 目录下，格式为 JSON。
- 示例：`result/result.json`

## 使用的模型
- 画师分类：`lsnet_xl_artist_448`
- danbooru：[danbooru 分类模型](https://huggingface.co/spaces/animetimm/dbv4-full-ranklist)
