import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models import create_model

# Ensure custom LSNet artist models are registered with timm
from model import lsnet_artist  # noqa: F401


def get_args_parser():
    parser = argparse.ArgumentParser('Artist Style Inference', add_help=False)
    
    # 模型参数
    parser.add_argument('--model', default='lsnet_t_artist', type=str,
                        choices=['lsnet_t_artist', 'lsnet_s_artist', 'lsnet_b_artist', 'lsnet_l_artist', 'lsnet_xl_artist', 'lsnet_xl_artist_448'],
                        help='Model architecture')
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='Path to model checkpoint')
    parser.add_argument('--input-size', default=224, type=int,
                        help='Input image size')
    
    # 输入输出
    parser.add_argument('--input', required=True, type=str,
                        help='Input image path or directory')
    parser.add_argument('--output', default='./output/inference', type=str,
                        help='Output directory')
    parser.add_argument('--class-csv', default=None, type=str,
                        help='Path to class mapping CSV exported during training')
    
    # 其他参数
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device to use')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size for batch inference')
    parser.add_argument('--threshold', default=0.0, type=float,
                        help='Probability threshold to filter predictions (default: 0.0)')
    
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



def load_model(args, state_dict):
    """加载模型"""
    print(f"Loading model: {args.model}")

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
    )
    model.to(args.device)
    model.eval()
    model.load_state_dict(state_dict, strict=True)
    print(f"Model loaded from {args.checkpoint}")
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


def classify_image(model, image_tensor, device, class_mapping: Optional[Dict[int, str]] = None, threshold=0.7):
    """对图像进行分类"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        # 使用分类头
        logits = model(image_tensor, return_features=False)
        
        # 计算概率
        probs = F.softmax(logits, dim=-1)
        
        # Top-K结果
        top_probs, top_indices = torch.topk(probs, k=min(5, probs.size(-1)), dim=-1)
        
        results = []
        for prob, idx in zip(top_probs[0].cpu().numpy(), top_indices[0].cpu().numpy()):
            if prob >= threshold:
                class_name = class_mapping.get(int(idx), f"Class {idx}") if class_mapping else f"Class {idx}"
                results.append({
                    'class_id': int(idx),
                    'class_name': class_name,
                    'probability': float(prob)
                })
        
        return results

def process_single_image(args, model, transform, class_mapping: Optional[Dict[int, str]] = None):
    """处理单张图像"""
    image_path = Path(args.input)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return
    
    print(f"\nProcessing: {image_path.name}")
    
    # 预处理
    image_tensor = preprocess_image(image_path, transform)
    
    results = {}
    
    print("\n[Classification Results]")
    classification = classify_image(model, image_tensor, args.device, class_mapping, args.threshold)
    results['classification'] = classification
    
    for i, result in enumerate(classification, 1):
        print(f"{i}. {result['class_name']}: {result['probability']:.4f}")
    return results


def process_directory(args, model, transform, class_mapping: Optional[Dict[int, str]] = None):
    """批量处理目录中的图像"""
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Error: Directory not found: {input_dir}")
        return
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_paths = [p for p in input_dir.glob('**/*') if p.suffix.lower() in image_extensions]
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
    
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
            
            if args.mode in ['classify', 'both']:
                logits = model(batch_tensor, return_features=False)
                probs = F.softmax(logits, dim=-1)
                top_probs, top_indices = torch.topk(probs, k=min(args.top_k, probs.size(-1)), dim=-1)
        
        # 保存结果
        for j, path in enumerate(batch_paths):
            if j >= len(batch_tensors):
                continue
            
            result = {'image': path.name}
            
            if args.mode in ['classify', 'both']:
                # 获取该图像的 top-k 结果
                img_top_probs = top_probs[j].cpu().numpy()
                img_top_indices = top_indices[j].cpu().numpy()
                
                classifications = []
                for prob, idx in zip(img_top_probs, img_top_indices):
                    if prob >= args.threshold:
                        class_id = int(idx)
                        class_name = class_mapping.get(class_id, f"Class {class_id}") if class_mapping else f"Class {class_id}"
                        classifications.append({
                            'class_id': class_id,
                            'class_name': class_name,
                            'probability': float(prob)
                        })
                
                # 如果需要，取前 top_k
                if len(classifications) > args.top_k:
                    classifications = classifications[:args.top_k]
                
                result['classification'] = classifications
            all_results[path.name] = result
        print(f"Processed {min(i + args.batch_size, len(image_paths))}/{len(image_paths)} images")
    
    return all_results


def get_artist(args):
    # 根据模型配置动态设置输入大小
    from model.lsnet_artist import default_cfgs_artist
    if args.model in default_cfgs_artist:
        model_cfg = default_cfgs_artist[args.model]
        configured_input_size = model_cfg.get('input_size', (3, 224, 224))[1]  # 获取高度（假设正方形）
        if args.input_size != configured_input_size:
            args.input_size = configured_input_size
            print(f"Auto-setting input_size to {configured_input_size} for model {args.model} (from config)")
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 加载类别映射
    class_mapping = load_class_mapping(args.class_csv)

    # 加载 checkpoint 并解析类别数
    state_dict = load_checkpoint_state(args.checkpoint)
    args.num_classes = check_and_get_num_classes(class_mapping, state_dict)

    # 加载模型
    model = load_model(args, state_dict)
    
    # 创建数据转换
    config = resolve_data_config({'input_size': (3, args.input_size, args.input_size)}, model=model)
    transform = create_transform(**config)
    
    # 判断输入类型
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 单张图像
        results = process_single_image(args, model, transform, class_mapping)
        
        # 保存结果
        output_file = output_dir / f"{input_path.stem}_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")
        
    elif input_path.is_dir():
        # 目录批量处理
        results = process_directory(args, model, transform, class_mapping)
        
        # 保存结果
        output_file = output_dir / "batch_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")
        
        # 如果是聚类模式，额外保存特征矩阵
        if args.mode in ['cluster', 'both']:
            features_list = []
            image_names = []
            for name, result in results.items():
                if 'features' in result:
                    features_list.append(result['features'])
                    image_names.append(name)
            
            if features_list:
                features_array = np.array(features_list)
                np.save(output_dir / "features.npy", features_array)
                with open(output_dir / "image_names.txt", 'w') as f:
                    f.write('\n'.join(image_names))
                print(f"Feature matrix saved: {output_dir / 'features.npy'}")
                print(f"Feature matrix shape: {features_array.shape}")
    else:
        print(f"Error: Invalid input path: {input_path}")

def main(args):
    get_artist(args)
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Artist Style Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)