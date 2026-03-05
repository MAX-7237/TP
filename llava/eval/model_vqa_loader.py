import argparse
import torch
import os
import json
import numpy as np
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    
    # 检查是否使用 actor
    use_attention_actor = getattr(args, 'use_attention_actor', False)
    actor_hidden_dim = getattr(args, 'actor_hidden_dim', 1024)
    actor_num_heads = getattr(args, 'actor_num_heads', 8)
    actor_num_layers = getattr(args, 'actor_num_layers', 1)
    actor_dropout = getattr(args, 'actor_dropout', 0.1)
    actor_top_k = getattr(args, 'actor_top_k', 192)
    actor_tau = getattr(args, 'actor_tau', 1.0)
    actor_ckpt = getattr(args, 'actor_ckpt', None)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        args.model_base, 
        model_name,
        device="cuda",
        use_attention_actor=use_attention_actor,
        actor_hidden_dim=actor_hidden_dim,
        actor_num_heads=actor_num_heads,
        actor_num_layers=actor_num_layers,
        actor_dropout=actor_dropout,
        actor_top_k=actor_top_k,
        actor_tau=actor_tau,
    )
    
    # 在推理模式下初始化 attention_actor（如果配置中启用了 actor 但实例未创建）
    if use_attention_actor:
        mdl = model.get_model()
        if hasattr(mdl, 'attention_actor') and mdl.attention_actor is None:
            from llava.model.dynamicvlm_actor import dynamicvlm_actor
            
            # 获取模型的 hidden_size 作为 text_dim 和 image_dim
            text_dim = model.config.hidden_size
            image_dim = text_dim
            
            print(f"[INFO] Initializing dynamicvlm_actor for inference: hidden_dim={actor_hidden_dim}, num_heads={actor_num_heads}, num_layers={actor_num_layers}, top_k={actor_top_k}")
            mdl.attention_actor = dynamicvlm_actor(
                text_dim=text_dim,
                image_dim=image_dim,
                hidden_dim=actor_hidden_dim,
                num_heads=actor_num_heads,
                num_layers=actor_num_layers,
                dropout=actor_dropout,
                top_k=actor_top_k,
                tau=actor_tau,
            )
            mdl.attention_actor = mdl.attention_actor.to(device="cuda", dtype=torch.float16)
    
    # 加载 actor 权重
    if use_attention_actor and actor_ckpt and actor_ckpt != "None":
        actor_ckpt_path = os.path.expanduser(actor_ckpt)
        if os.path.exists(actor_ckpt_path):
            print(f"[INFO] Loading actor weights from: {actor_ckpt_path}")
            actor_state_dict = torch.load(actor_ckpt_path, map_location='cpu')
            model.get_model().attention_actor.load_state_dict(actor_state_dict, strict=False)
            print(f"[INFO] Actor weights loaded successfully")
        else:
            print(f"[WARNING] Actor checkpoint not found: {actor_ckpt_path}")
    
    # 初始化可视化相关属性
    visualization_output_dir = getattr(args, 'visualization_output_dir', None)
    visualization_plots_save_steps = getattr(args, 'visualization_plots_save_steps', 50)
    
    has_visualization = visualization_output_dir is not None
    
    if has_visualization:
        # 初始化推理模式下的可视化数据收集
        model.get_model()._global_step = 0
        model.get_model()._visualization_save_steps = visualization_plots_save_steps
        model.get_model()._pruning_history_steps = []
        model.get_model()._pruning_history_keep_ratio = []
        print(f"[INFO] Visualization enabled: {visualization_output_dir}")

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # 收集 keep_mask 数据用于可视化
        if has_visualization and use_attention_actor:
            mdl = model.get_model()
            keep_mask_list = getattr(mdl, '_last_keep_mask_list', [])
            
            if len(keep_mask_list) > 0:
                # 记录 keep_ratio
                current_keep_ratio = keep_mask_list[-1].mean().item()
                mdl._global_step += 1
                current_step = mdl._global_step
                
                mdl._pruning_history_steps.append(current_step)
                mdl._pruning_history_keep_ratio.append(current_keep_ratio)
                
                print(f"[Eval Step {current_step}] keep_ratio: {current_keep_ratio:.4f}")
                
                # 保存可视化图像
                if current_step % visualization_plots_save_steps == 0:
                    save_dir = visualization_output_dir
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # 绘制 keep_ratio 曲线
                    try:
                        import matplotlib.pyplot as plt
                        
                        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                        ax.plot(mdl._pruning_history_steps, mdl._pruning_history_keep_ratio, 'b-', linewidth=1.5, alpha=0.5, label='Keep Ratio (Raw)')
                        
                        # 添加平滑曲线
                        if len(mdl._pruning_history_keep_ratio) > 10:
                            # 简单移动平均
                            window = min(10, len(mdl._pruning_history_keep_ratio) // 2)
                            smoothed = np.convolve(mdl._pruning_history_keep_ratio, np.ones(window)/window, mode='valid')
                            ax.plot(mdl._pruning_history_steps[:len(smoothed)], smoothed, 'b-', linewidth=2.5, label=f'Keep Ratio (Smoothed)')
                        
                        ax.set_xlabel('Steps', fontsize=12)
                        ax.set_ylabel('Keep Ratio', fontsize=12)
                        ax.set_title(f'Keep Ratio Over Evaluation (Step {current_step})', fontsize=14)
                        ax.legend(loc='upper right', fontsize=10)
                        ax.grid(True, alpha=0.3)
                        ax.set_ylim([0, 1])
                        
                        # 添加统计信息
                        final_ratio = mdl._pruning_history_keep_ratio[-1]
                        avg_ratio = np.mean(mdl._pruning_history_keep_ratio)
                        stats_text = f'Current: {final_ratio:.4f}\nAverage: {avg_ratio:.4f}'
                        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                        
                        plt.tight_layout()
                        save_path = os.path.join(save_dir, f'keep_ratio_eval_step_{current_step}.png')
                        plt.savefig(save_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        print(f"[Visualization] Saved: {save_path}")
                    except Exception as e:
                        print(f"[WARNING] Visualization failed: {e}")
                
                # 清空 keep_mask_list
                mdl._last_keep_mask_list = []

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()
    
    # 保存最终统计信息
    if has_visualization and use_attention_actor and len(mdl._pruning_history_keep_ratio) > 0:
        stats = {
            'total_samples': len(questions),
            'keep_ratio_stats': {
                'mean': float(np.mean(mdl._pruning_history_keep_ratio)),
                'std': float(np.std(mdl._pruning_history_keep_ratio)),
                'min': float(np.min(mdl._pruning_history_keep_ratio)),
                'max': float(np.max(mdl._pruning_history_keep_ratio)),
                'final': float(mdl._pruning_history_keep_ratio[-1]),
            }
        }
        stats_path = os.path.join(visualization_output_dir, 'eval_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"[INFO] Evaluation stats saved to: {stats_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    
    # Actor 相关参数
    parser.add_argument("--use_attention_actor", action="store_true", help="Use attention actor for pruning")
    parser.add_argument("--actor_hidden_dim", type=int, default=1024, help="Actor hidden dimension")
    parser.add_argument("--actor_num_heads", type=int, default=8, help="Actor number of attention heads")
    parser.add_argument("--actor_num_layers", type=int, default=1, help="Actor number of layers")
    parser.add_argument("--actor_dropout", type=float, default=0.1, help="Actor dropout probability")
    parser.add_argument("--actor_top_k", type=int, default=192, help="Number of tokens to keep")
    parser.add_argument("--actor_tau", type=float, default=1.0, help="Temperature for softmax")
    parser.add_argument("--actor_ckpt", type=str, default=None, help="Actor checkpoint path")
    
    # 可视化参数
    parser.add_argument("--visualization_output_dir", type=str, default=None, help="Directory to save visualizations")
    parser.add_argument("--visualization_plots_save_steps", type=int, default=50, help="Save visualization every N samples")
    
    args = parser.parse_args()

    eval_model(args)
