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



def load_model(args, state_dict):
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
            if j >= len(batch_tensors):
                continue
            
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
    model = load_model(args, state_dict)
    
    # 创建数据转换
    config = resolve_data_config({'input_size': (3, args.input_size, args.input_size)}, model=model)
    transform = create_transform(**config)
    
    # 判断输入类型
    input_path = Path(args.input)
    def process_fn(inputs):
        logits = model(inputs, return_features=False)
        probs = F.softmax(logits, dim=-1)
        return probs
    if input_path.is_dir():
        # 目录批量处理
        results = process_directory(args, transform, process_fn,class_mapping, [args.artist_threshold] * len(class_mapping))
        # 保存结果
        output_file = output_dir / "batch_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")
    else:
        print(f"Error: Invalid input path: {input_path}")

def generate_image_from_sdxl(args):
    pass
def get_character(args):
    pass

def get_danbooru_tags(args):
    model = create_model(f'hf-hub:{args.danbooru_model_checkpoint}', pretrained=True).cuda()
    data_config = resolve_data_config(model=model)
    model.eval()
    preprocessor = T.Compose([
        T.Resize(size=data_config['input_size'][-1], antialias=True),
        T.CenterCrop(size=data_config['input_size'][1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
    ])
    df_tags = pd.read_csv(
        hf_hub_download(repo_id=args.danbooru_model_checkpoint, repo_type='model', filename='selected_tags.csv'),
        keep_default_na=False
    )
    tags_mapping = {int(k):v for k,v in zip(range(len(df_tags['name'])),df_tags['name'])}

    def process_fn(tensors):
        output = model(tensors)
        prediction = torch.sigmoid(output)
        return prediction

    result = process_directory(args,preprocessor,process_fn,tags_mapping,df_tags['best_threshold'])
    print(result)
def main(args):
    # generate_image_from_sdxl(args)
    get_artist(args)
    get_danbooru_tags(args)
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Artist Style Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)