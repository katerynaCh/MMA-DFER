# --------------------------------------------------------
# References:
# timm: https://github.com/huggingface/pytorch-image-models
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch

from timm.models.vision_transformer import VisionTransformer
import torch.nn as nn

class VisionTransformer2(VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, n_seq=196, n_progr=3, n_frames=16, **kwargs):
        super(VisionTransformer2, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        self.n_seq = n_seq
        self.n_progr = n_progr
        self.n_frames = n_frames

        self.latent_dim = 128
       
        self.learnable_prompts_init = nn.Parameter(torch.randn(self.n_progr * (len(self.blocks)//6), 768) * (768 **-0.5))
        self.learnable_prompts_progr = nn.ParameterList([nn.Parameter(torch.randn(self.n_progr, 768) * (768 **-0.5)) for i in range(len(self.blocks)//6)])

        self.audio_proj_pre = nn.ModuleList([nn.Sequential(nn.Linear(768, self.latent_dim), nn.LayerNorm(self.latent_dim)) for i in range(len(self.blocks))])
        self.temporal_pre = nn.ModuleList([nn.Linear(768, self.latent_dim) for i in range(len(self.blocks))])

        self.temporal_pre_norm = nn.ModuleList([nn.LayerNorm(self.latent_dim) for i in range(len(self.blocks))])
        self.temporal_att_post = nn.ModuleList([nn.Sequential(nn.Linear(self.latent_dim, 768), nn.GELU()) for i in range(len(self.blocks))])
        self.all_gate = nn.ParameterList([nn.Parameter(torch.zeros(1)) for i in range(len(self.blocks))])
   
    def forward_block_pre(self, ii, x, B):
        
        if ii == 0:
            x = self.patch_embed(x)
            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x, self.learnable_prompts_init.expand(B, -1, -1)), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)

        x = self.blocks[ii](x)
        return x

    def forward_block_post(self, ii, x, x_t, B):
        x_t = self.temporal_att_post[ii](x_t)
        x = x + nn.functional.tanh(self.all_gate[ii])* x_t.unsqueeze(2).view(B, -1, 768)

        if ii % 6 == 0:
            prompts_progr = self.learnable_prompts_progr[ii//6].expand(B, -1, -1)
            x[:,self.n_seq+1+ii//6*self.n_progr:self.n_seq+1+(ii//6+1)*self.n_progr,:] = x[:,self.n_seq+1+ii//6*self.n_progr:self.n_seq+1+(ii//6+1)*self.n_progr,:] + prompts_progr

        if ii == (len(self.blocks) - 1):
            if self.global_pool:
                x = x[:, 1:, :] # without cls token (N, L=14*14, D=768=16*16*3)
                x = x.mean(dim=1) # global average pooling (N, D=768)
                outcome = self.fc_norm(x) # Layer Normalization (N, D=768)
                return outcome
            else:
                x = self.norm(x)
                outcome = x[:, 0]
                return outcome
        return x #outcome

 
 
    def forward_features(self, x, audio=None):
        B = x.shape[0]
        #print('im shape: ', x.shape)
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        #x = torch.cat((cls_tokens, x), dim=1)
        x = torch.cat((cls_tokens, x, self.learnable_prompts_init.expand(B, -1, -1)), dim=1)
        #print('image_seq: ', x.shape)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for ii, blk in enumerate(self.blocks): 
            prompts_progr = self.learnable_prompts_progr[ii].expand(B, -1, -1)
            #print(prompts_progr.shape, audio.shape)
            #prompts_progr = self.audio_proj_post[ii](prompts_progr + self.audio_proj_pre(audio).repeat_interleave(16,0).unsqueeze(1))
            #prompts_progr = prompts_progr + audio.repeat_interleave(16,0).unsqueeze(1)
            #prompts_progr = self.audio_proj_post[ii](prompts_progr + audio.repeat_interleave(16,0).unsqueeze(1))
            x = blk(x)

            x_t = x[:,0,:].contiguous().view(B // 16, 16, x.shape[-1]) 
            x_t = x_t + self.temporal_pos_embed		
            x_t = self.temporal_pre[ii](x_t)

            qs = self.learnable_q[ii].expand(B // 16, -1, -1)
            qs = self.norm_qs[ii](qs)
            x_t_1, _ = self.context_att[ii](qs, x_t, x_t, need_weights=False)
            #x_t = x_t + nn.functional.tanh(self.context_gate[ii])*x_t_1
 

            x_a = self.audio_proj_pre[ii](audio[ii])                        
            x_t_2, _ = self.audio_att[ii](qs, x_a, x_a, need_weights=False)
            x_t = x_t + nn.functional.tanh(self.audio_gate[ii])*x_t_2 + nn.functional.tanh(self.context_gate[ii])*x_t_1

            #x_t_1 = self.norm_xt_2[ii](x_t)
            #x_t_1 = self.norm_xt[ii](x_t)
            #x_t_1, _ = self.temporal_att[ii](x_t_1, x_t_1, x_t_1, need_weights=False)
            #x_t = x_t + nn.functional.tanh(self.temp_gate[ii])*x_t_1

            x_t = self.temporal_att_post[ii](x_t)

            x[:,1:197,:] = x[:,1:197,:] + nn.functional.tanh(self.all_gate[ii])* x_t.unsqueeze(2).view(B, -1, 768)
            if ii != 11:
                x[:,197+ii*3:197+(ii+1)*3,:] = x[:,197+ii*3:197+(ii+1)*3,:] + prompts_progr

        if self.global_pool:
            x = x[:, 1:, :] # without cls token (N, L=14*14, D=768=16*16*3)
            x = x.mean(dim=1) # global average pooling (N, D=768)
            outcome = self.fc_norm(x) # Layer Normalization (N, D=768)
        else:
            #x = self.norm(x)
            #print('mae', x.shape)
            outcome = x[:, 0]

        return outcome

    # borrow from timm
    def forward(self, x, ret_feature=False):
        x = self.forward_features(x)
        feature = x
        if getattr(self, 'head_dist', None) is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        # return
        if ret_feature:
            return x, feature
        else:
            return x


# setup model archs
VIT_KWARGS_BASE = dict(mlp_ratio=4, qkv_bias=True,
    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))

VIT_KWARGS_PRESETS = {
    'tiny': dict(patch_size=16, embed_dim=192, depth=12, num_heads=3),
    'small': dict(patch_size=16, embed_dim=384, depth=12, num_heads=6),
    'base': dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
    'large': dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
    'huge': dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16),
    'giant': dict(patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11),
    'gigantic': dict(patch_size=14, embed_dim=1664, depth=48, num_heads=16, mlp_ratio=64/13),
}

def create_vit_model(preset=None, creator=None, **kwargs):
    preset = 'base' if preset is None else preset.lower()
    all_kwargs = dict()
    all_kwargs.update(VIT_KWARGS_BASE)
    all_kwargs.update(VIT_KWARGS_PRESETS[preset])
    all_kwargs.update(kwargs)
    if creator is None:
        creator = VisionTransformer2
    return creator(**all_kwargs)

#vit_tiny_patch16 = partial(create_vit_model, preset='tiny')
#vit_small_patch16 = partial(create_vit_model, preset='small')
vit_base_patch16 = partial(create_vit_model, preset='base')
#vit_large_patch16 = partial(create_vit_model, preset='large')
#vit_huge_patch14 = partial(create_vit_model, preset='huge')
#vit_giant_patch14 = partial(create_vit_model, preset='giant')
#vit_gigantic_patch14 = partial(create_vit_model, preset='gigantic')
