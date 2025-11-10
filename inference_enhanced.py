# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import argparse
import re

import imageio.v3 as iio
import nncore
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
from PIL import Image

from unipixel.constants import MEM_TOKEN, REF_TOKEN, SEG_TOKEN
from unipixel.dataset.utils import process_vision_info
from unipixel.model.builder import build_model
from unipixel.utils.io import load_image, load_video
from unipixel.utils.transforms import get_sam2_transform
from unipixel.utils.visualizer import draw_mask

BANNER = r"""
=================================================================================
   __    __    __   __    __    ______     __   ___   ___   _______    __
  |  |  |  |  |  \ |  |  |  |  |   _  \   |  |  \  \ /  /  |   ____|  |  |
  |  |  |  |  |   \|  |  |  |  |  |_)  |  |  |   \  V  /   |  |__     |  |
  |  |  |  |  |  . `  |  |  |  |   ___/   |  |    >   <    |   __|    |  |
  |  `--'  |  |  |\   |  |  |  |  |       |  |   /  .  \   |  |____   |  `----.
   \______/   |__| \__|  |__|  | _|       |__|  /__/ \__\  |_______|  |_______|

=================================================================================
"""

INFO = """
\033[1;36m   Examples:\033[0m 
             1. Segmentation: "Please segment the tallest giraffe."
             2. Regional Understanding with mask: Use --refer_mask_path option
             3. Point-based Referring: Use --point_coords and --point_labels options

\033[1;32m Model Path:\033[0m {}
\033[1;32m Media Path:\033[0m {}
\033[1;32m     Prompt:\033[0m {}
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('media_path')
    parser.add_argument('prompt')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--model_path', default='PolyU-ChenLab/UniPixel-3B')
    parser.add_argument('--sample_frames', type=int, default=16)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--dtype', default='bfloat16')
    
    # 支持 <mem> token - Regional Understanding模式
    parser.add_argument('--refer_mask_path', type=str, default=None,
                        help='Path to reference mask image (triggers <mem> token)')
    parser.add_argument('--prompt_frame_idx', type=int, default=0,
                        help='Frame index for mask prompt (used with refer_mask_path)')
    
    # 支持 <ref> token - Point/Box Referring模式  
    parser.add_argument('--point_coords', type=str, default=None,
                        help='Point coordinates: "x1,y1;x2,y2" (normalized 0-1024)')
    parser.add_argument('--point_labels', type=str, default=None,
                        help='Point labels: "1,1" (1=positive, 0=negative, 2=top-left, 3=bottom-right)')
    parser.add_argument('--point_frame_idx', type=int, default=0,
                        help='Frame index for point annotation')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print(BANNER + INFO.format(args.model_path, args.media_path, args.prompt))

    model, processor = build_model(args.model_path, device=args.device, dtype=args.dtype)
    device = next(model.parameters()).device

    sam2_transform = get_sam2_transform(model.config.sam2_image_size)

    # 加载图像或视频
    if any(args.media_path.endswith(k) for k in ('jpg', 'png')):
        frames, images = load_image(args.media_path), [args.media_path]
        is_video = False
    else:
        frames, images = load_video(args.media_path, sample_frames=args.sample_frames)
        is_video = True

    frame_size = frames.shape[1:3]
    
    # ==================== 模式1: Regional Understanding (使用 <mem> token) ====================
    if args.refer_mask_path:
        print(f"\033[1;33m   Mode: Regional Understanding with <mem> token\033[0m")
        
        # 加载参考mask
        mask_img = Image.open(args.refer_mask_path).convert('L')
        mask_array = torch.from_numpy(nncore.pure_array(mask_img)) > 128
        
        # 调整mask大小到帧尺寸
        mask_resized = T.resize(mask_array.unsqueeze(0).unsqueeze(0).float(), frame_size)
        mask_resized = mask_resized.squeeze(0).squeeze(0) > 0.5
        
        # 构建refer_mask张量 [num_frames, num_objects, height, width]
        refer_mask = torch.zeros(frames.size(0), 1, *frame_size, dtype=torch.bool)
        refer_mask[args.prompt_frame_idx, 0] = mask_resized
        
        # 处理refer_mask以匹配模型要求
        if refer_mask.size(0) % 2 != 0:
            refer_mask = torch.cat((refer_mask, refer_mask[-1, None]))
        refer_mask = refer_mask.flatten(1)
        refer_mask = F.max_pool1d(refer_mask.float().transpose(-1, -2), kernel_size=2, stride=2).transpose(-1, -2)
        refer_mask = refer_mask.view(-1, 1, *frame_size)
        
        # 构建prompt - 使用与demo/app.py相同的格式
        if not is_video:
            prefix = f'Here is an image with the following highlighted regions:\n[0]: <{args.prompt_frame_idx + 1}> {MEM_TOKEN}\n'
        else:
            prefix = f'Here is a video with {len(images)} frames denoted as <1> to <{len(images)}>. The highlighted regions are as follows:\n'
            # 找到mask出现的帧
            tids = (refer_mask[:, 0].any(dim=(-1, -2)).nonzero()[:, 0] * 2 + 1).tolist()
            prefix += '[0]: ' + ' '.join([f'<{tid}>-<{tid + 1}> {MEM_TOKEN}' for tid in tids]) + '\n'
        
        prompt_text = prefix + args.prompt
        
    # ==================== 模式2: Point/Box Referring (使用 <ref> token) ====================
    elif args.point_coords and args.point_labels:
        print(f"\033[1;33m   Mode: Point/Box Referring with <ref> token\033[0m")
        
        # 解析坐标和标签
        coords = [[float(x) for x in coord.split(',')] for coord in args.point_coords.split(';')]
        labels = [int(x) for x in args.point_labels.split(',')]
        
        # 转换为张量格式 (必须是SAM2图像尺寸的坐标)
        point_coords = [[torch.tensor([coords], dtype=torch.float32)]]
        point_labels = [[torch.tensor([labels], dtype=torch.int32)]]
        point_frames = [[torch.tensor([args.point_frame_idx], dtype=torch.int32)]]
        
        # 构建prompt - 使用[0] <|ref|>格式
        prompt_text = args.prompt.replace('<region>', f'[0] {REF_TOKEN}')
        if '<region>' not in args.prompt:
            # 如果没有<region>占位符，在prompt开头添加
            prompt_text = f'[0] {REF_TOKEN} ' + args.prompt
        
    # ==================== 模式3: 纯分割 (模型输出 <seg> token) ====================
    else:
        print(f"\033[1;33m   Mode: Pure Segmentation (model generates <seg> token)\033[0m")
        prompt_text = args.prompt

    print(f"\033[1;32m Final Prompt:\033[0m {prompt_text}")

    # 构建消息
    messages = [{
        'role': 'user',
        'content': [{
            'type': 'video',
            'video': images,
            'min_pixels': 128 * 28 * 28,
            'max_pixels': 256 * 28 * 28 * int(args.sample_frames / len(images))
        }, {
            'type': 'text',
            'text': prompt_text
        }]
    }]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)

    images_data, videos, kwargs = process_vision_info(messages, return_video_kwargs=True)

    data = processor(text=[text], images=images_data, videos=videos, return_tensors='pt', **kwargs)

    data['frames'] = [sam2_transform(frames).to(model.sam2.dtype)]
    data['frame_size'] = [frames.shape[1:3]]

    # 根据模式添加额外参数
    if args.refer_mask_path:
        # 调整refer_mask到正确的分辨率 (spatial patch size=14, merge size=2)
        refer_mask_adjusted = T.resize(refer_mask.float(), (data['video_grid_thw'][0][1] * 14, data['video_grid_thw'][0][2] * 14))
        refer_mask_adjusted = F.max_pool2d(refer_mask_adjusted, kernel_size=28, stride=28)
        refer_mask_adjusted = refer_mask_adjusted > 0
        data['refer_mask'] = [refer_mask_adjusted.to(device)]
        
    elif args.point_coords and args.point_labels:
        data['point_coords'] = [[p.to(device) for p in point_coords[0]]]
        data['point_labels'] = [[p.to(device) for p in point_labels[0]]]
        data['point_frames'] = [[p.to(device) for p in point_frames[0]]]

    # 生成
    output_ids = model.generate(
        **data.to(device),
        do_sample=False,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        max_new_tokens=512)

    assert data.input_ids.size(0) == output_ids.size(0) == 1
    output_ids = output_ids[0, data.input_ids.size(1):]

    if output_ids[-1] == processor.tokenizer.eos_token_id:
        output_ids = output_ids[:-1]

    response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
    
    # 高亮显示特殊token
    response_display = response.replace(SEG_TOKEN, f'\033[1;35m{SEG_TOKEN}\033[0m')
    response_display = response_display.replace(MEM_TOKEN, f'\033[1;34m{MEM_TOKEN}\033[0m')
    response_display = response_display.replace(REF_TOKEN, f'\033[1;36m{REF_TOKEN}\033[0m')
    
    print(f'\n\033[1;32m   Response:\033[0m {response_display}')

    # 保存分割结果
    if len(model.seg) >= 1:
        imgs = draw_mask(frames, model.seg)

        nncore.mkdir(args.output_dir)

        path = nncore.join(args.output_dir, f"{nncore.pure_name(args.media_path)}.{'gif' if len(imgs) > 1 else 'png'}")
        print(f'\033[1;32mOutput Path:\033[0m {path}')
        iio.imwrite(path, imgs, duration=100, loop=0)
    else:
        print(f'\n\033[1;33m   No segmentation masks generated.\033[0m')