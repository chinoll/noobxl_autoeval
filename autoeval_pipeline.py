import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models import create_model
from torchvision import transforms as T
import pandas as pd
from huggingface_hub import hf_hub_download

# Ensure custom LSNet artist models are registered with timm
from model import lsnet_artist  # noqa: F401


def get_args_parser():
    parser = argparse.ArgumentParser('Artist Style Inference', add_help=False)
    

    parser.add_argument('--danbooru-model-name', required=False, type=str,
                        help='danbooru model name')
    
    # 输入输出
    parser.add_argument('--input', required=True, type=str,
                        help='Input image path or directory')
    parser.add_argument('--output', default='./output/inference', type=str,
                        help='Output directory')
    parser.add_argument('--answer', required=False, type=str,
                        help='answer')
    
    # 其他参数
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device to use')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size for batch inference')
    
    #画师分类相关参数设置
    parser.add_argument('--artist_threshold', default=0.4, type=float,
                        help='Probability threshold to filter predictions (default: 0.0)')
    parser.add_argument('--artist-class-csv', default=None, type=str,
                        help='Path to class mapping CSV exported during training')
    parser.add_argument('--artist-model-type', default='lsnet_xl_artist_448', type=str,
                        choices=['lsnet_xl_artist_448'],
                        help='Model architecture')
    parser.add_argument('--artist-model-checkpoint', required=False, type=str,
                        help='Path to artist model checkpoint')
    return parser


def load_checkpoint_state(checkpoint_path: str):
    """加载 checkpoint 并返回模型权重"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            return checkpoint['model']
        if 'model_ema' in checkpoint:
            return checkpoint['model_ema']
    return checkpoint


def check_and_get_num_classes(class_mapping: Optional[Dict[int, str]],state_dict) -> int:
    """根据参数、CSV 或 checkpoint 推断类别数"""
    # 优先使用CSV中的类别数
    if class_mapping:
        csv_classes = len(class_mapping)
    # 最后尝试从权重中解析分类头大小
    for key, value in state_dict.items():
        if key.endswith('head.weight') or key.endswith('head.l.weight'):
            weight_classes = value.shape[0]
    if csv_classes != weight_classes:
        raise ValueError("权重的类别与CSV文件中不一致，请检查下载的权重或CSV")
    return csv_classes



def load_artist_model(args, state_dict):
    """加载模型"""
    print(f"Loading model: {args.artist_model_type}")

    model = create_model(
        args.artist_model_type,
        pretrained=False,
        num_classes=args.num_classes,
    )
    model.to(args.device)
    model.eval()
    model.load_state_dict(state_dict, strict=True)
    print(f"Model loaded from {args.artist_model_checkpoint}")
    return model


def load_class_mapping(class_csv_path: Optional[str]) -> Optional[Dict[int, str]]:
    """加载 CSV 类别映射，返回 class_id -> name 的字典"""
    if not class_csv_path:
        return None

    csv_path = Path(class_csv_path)
    with csv_path.open('r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or 'class_id' not in reader.fieldnames or 'class_name' not in reader.fieldnames:
            raise ValueError('CSV 必须包含 class_id 和 class_name 两列。')

        mapping: Dict[int, str] = {}
        for row in reader:
            class_id = int(row['class_id'])
            class_name = row['class_name']
            mapping[class_id] = class_name

    if not mapping:
        raise ValueError(f"CSV {csv_path} 中未找到任何类别映射。")

    return mapping


def preprocess_image(image_path, transform):
    """预处理单张图像"""
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image)
    return tensor.unsqueeze(0)

def process_directory(args, transform, callback_fn, class_mapping,threshold):
    """批量处理目录中的图像"""
    input_dir = Path(args.input)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_paths = [p for p in input_dir.glob('**/*') if p.suffix.lower() in image_extensions]
    print(f"Found {len(image_paths)} images")
    all_results = {}
    
    # 批量处理
    for i in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[i:i + args.batch_size]
        
        # 预处理批次
        batch_tensors = []
        for path in batch_paths:
            try:
                tensor = preprocess_image(path, transform)
                batch_tensors.append(tensor)
            except Exception as e:
                print(f"Error processing {path.name}: {e}")
                continue
        
        if not batch_tensors:
            continue
        
        batch_tensor = torch.cat(batch_tensors, dim=0)
        
        # 推理
        with torch.no_grad():
            batch_tensor = batch_tensor.to(args.device)
            probs = callback_fn(batch_tensor)
            top_probs, top_indices = torch.topk(probs, k=probs.size(-1), dim=-1)
        
        # 保存结果
        for j, path in enumerate(batch_paths):
            result = {'image': path.name}

            img_top_probs = top_probs[j].cpu().numpy()
            img_top_indices = top_indices[j].cpu().numpy()
            
            classifications = []
            for prob, idx in zip(img_top_probs, img_top_indices):
                class_id = int(idx)
                if class_id in class_mapping and prob >= threshold[int(idx)]:
                    class_name = class_mapping[class_id]
                    classifications.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'probability': float(prob)
                    })
                
                result['classification'] = classifications
            all_results[path.name] = result
        print(f"Processed {min(i + args.batch_size, len(image_paths))}/{len(image_paths)} images")
    
    return all_results


def get_artist(args):
    args.input_size = 448
    # args.artist_model_type = 'lsnet_xl_artist_448'
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 加载类别映射
    class_mapping = load_class_mapping(args.artist_class_csv)

    # 加载 checkpoint 并解析类别数
    state_dict = load_checkpoint_state(args.artist_model_checkpoint)
    args.num_classes = check_and_get_num_classes(class_mapping, state_dict)

    # 加载模型
    model = load_artist_model(args, state_dict)
    
    # 创建数据转换
    config = resolve_data_config({'input_size': (3, args.input_size, args.input_size)}, model=model)
    transform = create_transform(**config)
    
    # 判断输入类型
    def process_fn(inputs):
        logits = model(inputs, return_features=False)
        probs = F.softmax(logits, dim=-1)
        return probs
    results = process_directory(args, transform, process_fn,class_mapping, [args.artist_threshold] * len(class_mapping))
    return results

def generate_image_from_sdxl(args):
    pass
def postprocess_result(args,artist_result, danbooru_result):
    merged_results = {}
    df_tags = pd.read_csv(
        hf_hub_download(repo_id=args.danbooru_model_name, repo_type='model', filename='selected_tags.csv'),
        keep_default_na=False
    )
    tags_mapping = {k:int(v) for k,v in zip(df_tags['name'],df_tags['category'])}
    # 以 artist_result 为主遍历
    for img_name, artist_info in artist_result.items():
        image_path = artist_info.get("image", img_name)
        artist_names = [c["class_name"] for c in artist_info.get("classification", [])]
        # 获取 danbooru 标签
        danbooru_info = danbooru_result.get(img_name, {})
        danbooru_names = [c["class_name"] for c in danbooru_info.get("classification", [])]
        merged_results[image_path] = {
            "artist": artist_names,
            "general":[i for i in danbooru_names if tags_mapping[i] == 0],
            "character":[i for i in danbooru_names if tags_mapping[i] == 4],
            "rating":[i for i in danbooru_names if tags_mapping[i] == 9]
        }
    return merged_results

def get_model_score(pred, target_path):
    stats = {
        'artist': {},
        'general': {},
        'character': {},
        'rating': {}
    }
    # 加载目标答案
    with open(target_path, encoding="utf-8") as f:
        target = json.load(f)

    def update_stats(category, label, answer_set):
        if label not in stats[category]:
            stats[category][label] = {'total': 0, 'correct': 0}
        stats[category][label]['total'] += 1
        if label in answer_set:
            stats[category][label]['correct'] += 1

    # 遍历每张图片的预测结果
    for img_name, pred_info in pred.items():
        artist_set = set(target[img_name].get('artist', []))
        general_set = set(target[img_name].get('general', []))
        character_set = set(target[img_name].get('character', []))
        rating_set = set(target[img_name].get('rating', []))
        for label in pred_info.get('artist', []):
            update_stats('artist', label, artist_set)
        for label in pred_info.get('general', []):
            update_stats('general', label, general_set)
        for label in pred_info.get('character', []):
            update_stats('character', label, character_set)
        for label in pred_info.get('rating', []):
            update_stats('rating', label, rating_set)

    # 计算准确率
    score = {
        cat: {label: (info['correct'] / info['total'] if info['total'] > 0 else 0)
              for label, info in stats[cat].items()}
        for cat in stats
    }
    return stats,score

def visualization_results(stats, score):
    """
    可视化每个大分类的平均准确率、加权平均准确率、最低的10个子标签。
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    for cat in ['artist', 'general', 'character', 'rating']:
        print(f"\n分类: {cat}")
        # 平均准确率
        avg_acc = np.mean(list(score[cat].values())) if score[cat] else 0
        print(f"平均准确率: {avg_acc:.4f}")
        # 加权平均准确率
        total = sum(stats[cat][k]['total'] for k in stats[cat])
        weighted_acc = sum(stats[cat][k]['correct'] for k in stats[cat]) / total if total > 0 else 0
        print(f"加权平均准确率: {weighted_acc:.4f}")
        # 最低的10个子标签
        sorted_labels = sorted(score[cat].items(), key=lambda x: x[1])
        print("最低的10个子标签:")
        for label, acc in sorted_labels[:10]:
            print(f"  {label}: {acc:.4f} (样本数: {stats[cat][label]['total']})")
        # 可视化准确率分布
        plt.figure(figsize=(10, 4))
        plt.hist(list(score[cat].values()), bins=30, color='skyblue', edgecolor='black')
        plt.title(f"{cat} 标签准确率分布")
        plt.xlabel("准确率")
        plt.ylabel("标签数量")
        plt.tight_layout()
        plt.show()

def get_danbooru_tags(args):
    model = create_model(f'hf-hub:{args.danbooru_model_name}', pretrained=True).cuda()
    data_config = resolve_data_config(model=model)
    model.eval()
    preprocessor = T.Compose([
        T.Resize(size=data_config['input_size'][-1], antialias=True),
        T.CenterCrop(size=data_config['input_size'][1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
    ])
    df_tags = pd.read_csv(
        hf_hub_download(repo_id=args.danbooru_model_name, repo_type='model', filename='selected_tags.csv'),
        keep_default_na=False
    )
    tags_mapping = {int(k):v for k,v in zip(range(len(df_tags['name'])),df_tags['name'])}

    def process_fn(tensors):
        output = model(tensors)
        prediction = torch.sigmoid(output)
        return prediction

    result = process_directory(args,preprocessor,process_fn,tags_mapping,df_tags['best_threshold'])
    return result

def main(args):
    create_output_dir(args)
    generate_image_from_sdxl(args)
    artist = get_artist(args)
    danbooru = get_danbooru_tags(args)
    result = postprocess_result(args,artist,danbooru)
    if args.answer is not None:
        stats,score = get_model_score(result,args.answer)
        # print(score)
        visualization_results(stats,score)
def create_output_dir(args):
    pass
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Artist Style Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)