import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        # 新增：是否收集注意力分数
        self.collect_attention = getattr(args, 'collect_attention', False)
        # 新增：收集倒数第几层的注意力（默认倒数第2层）
        self.attention_layer_idx = getattr(args, 'attention_layer_idx', -2)

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images, return_attention=False):
        if type(images) is list:
            image_features = []
            attention_scores = [] if return_attention else None
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                    output_attentions=return_attention
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)

                if return_attention:
                    # 获取指定层的注意力分数
                    attns = image_forward_out.attentions
                    if attns is not None:
                        # attention_layer_idx 可以是负数（如-2表示倒数第2层）
                        layer_idx = self.attention_layer_idx
                        if layer_idx < 0:
                            layer_idx = len(attns) + layer_idx
                        attn = attns[layer_idx]  # shape: [batch, num_heads, seq_len, seq_len]
                        # 取CLS token（index 0）对其他token的注意力
                        cls_attention = attn[0, :, 0, 1:].cpu()  # [num_heads, seq_len-1]
                        # 对不同注意力头进行归一化后取平均
                        attn_min = cls_attention.min(dim=-1, keepdim=True)[0]  # [num_heads, 1]
                        attn_max = cls_attention.max(dim=-1, keepdim=True)[0]  # [num_heads, 1]
                        attn_range = attn_max - attn_min + 1e-8
                        attn_normalized = (cls_attention - attn_min) / attn_range  # [num_heads, seq_len-1]
                        cls_attention = attn_normalized.mean(dim=0)  # [seq_len-1]
                        attention_scores.append(cls_attention)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
                output_attentions=return_attention
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

            attention_scores = None
            if return_attention:
                attns = image_forward_outs.attentions
                if attns is not None:
                    layer_idx = self.attention_layer_idx
                    if layer_idx < 0:
                        layer_idx = len(attns) + layer_idx
                    attn = attns[layer_idx]  # shape: [batch, num_heads, seq_len, seq_len]
                    # 取CLS token对其他token的注意力 (batch=0, cls=0)
                    attention_scores = attn[:, :, 0, 1:].cpu()  # [batch, num_heads, seq_len-1]

                    # ----------对不同注意力头,先各自进行归一化后，再取平均----------
                    # 对每个注意力头单独归一化: (a - amin) / (amax - amin)
                    attn_min = attention_scores.min(dim=-1, keepdim=True)[0]  # [batch, num_heads, 1]
                    attn_max = attention_scores.max(dim=-1, keepdim=True)[0]  # [batch, num_heads, 1]
                    # 避免除零
                    attn_range = attn_max - attn_min + 1e-8
                    attn_normalized = (attention_scores - attn_min) / attn_range  # [batch, num_heads, seq_len-1]
                    # 对注意力头取平均
                    attention_scores = attn_normalized.mean(dim=1)  # [batch, seq_len-1]

        if return_attention:
            return image_features, attention_scores
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images, return_attention=False):
        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,
            output_attentions=return_attention
        )
        image_features = self.feature_select(image_forward_outs).to(images.dtype)

        attention_scores = None
        if return_attention:
            attns = image_forward_outs.attentions
            if attns is not None:
                layer_idx = self.attention_layer_idx
                if layer_idx < 0:
                    layer_idx = len(attns) + layer_idx
                attn = attns[layer_idx]
                # 取CLS token对其他token的注意力
                attention_scores = attn[:, :, 0, 1:].cpu()

        if return_attention:
            return image_features, attention_scores
        return image_features

    @torch.no_grad()
    def forward(self, images, return_attention=False):
        if type(images) is list:
            image_features = []
            attention_scores = [] if return_attention else None
            for image in images:
                image_feature = self.multiscale_forward(
                    lambda x: self.forward_feature(x, return_attention=False),
                    image.unsqueeze(0),
                    img_sizes=self.s2_scales,
                    max_split_size=self.s2_split_size
                )
                image_features.append(image_feature)

                # S2多尺度模式下，注意力收集比较复杂，暂时返回None
                # 如果需要收集注意力，需要修改s2wrapper支持
                if return_attention:
                    attention_scores.append(None)
        else:
            image_features = self.multiscale_forward(
                lambda x: self.forward_feature(x, return_attention=False),
                images,
                img_sizes=self.s2_scales,
                max_split_size=self.s2_split_size
            )

            if return_attention:
                attention_scores = [None]  # S2模式暂时不支持
            else:
                attention_scores = None

        if return_attention:
            return image_features, attention_scores
        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
