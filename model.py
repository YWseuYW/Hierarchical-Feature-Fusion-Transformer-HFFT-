import torch
import torch.nn as nn
import math
from timm.models.layers import DropPath,  trunc_normal_
import pdb
import torch.nn.functional as F




def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    # print(min_value)
    # print(int(v + divisor / 2) // divisor * divisor)
    # pdb.set_trace()
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):#多头注意力
    def __init__(self, dim, hidden_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qk = nn.Linear(dim, hidden_dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

    def forward(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]   # make torchscript happy (cannot use tensor as tuple)
        v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block_Frame(nn.Module):

    def __init__(self, outer_dim, inner_dim,  inner_num_heads, num_words, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.has_inner = inner_dim > 0
        if self.has_inner:
            # Frame Transformer Block
            self.inner_norm1 = norm_layer(inner_dim)
            self.inner_attn = Attention(
                inner_dim, inner_dim, num_heads=inner_num_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.inner_norm2 = norm_layer(inner_dim)
            self.inner_mlp = Mlp(in_features=inner_dim, hidden_features=int(inner_dim * mlp_ratio),
                                 out_features=inner_dim, act_layer=act_layer, drop=drop)
            self.proj_norm1 = norm_layer(num_words * inner_dim)
            self.proj = nn.Linear(num_words * inner_dim, 768, bias=False)
            self.actv = nn.ReLU(inplace=True)
            self.proj_norm2 = norm_layer(768)
            self.proj2 = nn.Linear(768,  outer_dim, bias=False)
            self.proj_norm3 = norm_layer(outer_dim)
            self.score = nn.Softmax2d()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, inner_tokens, outer_tokens):
        if self.has_inner:
            inner_tokens = inner_tokens + self.drop_path(self.inner_attn(self.inner_norm1(inner_tokens))) # B*N, k*k, c
            inner_tokens = inner_tokens + self.drop_path(self.inner_mlp(self.inner_norm2(inner_tokens))) # B*N, k*k, c
            B, N, C = outer_tokens.size()
            # print(inner_tokens.reshape(B, N, -1).shape)#(1,10,384)
            # pdb.set_trace()
            #这个地方就是特征融合层，将帧级特征与段级特征进行融合
            frame_score= self.score(self.proj_norm3(self.actv(self.proj2(self.proj_norm2(self.actv(self.proj(self.proj_norm1(inner_tokens.reshape(B, N, -1)))))))))
            # print(frame_score.shape)
            # print(outer_tokens.shape)
            # print( self.score(self.proj_norm3(self.actv(self.proj2(self.proj_norm2(self.actv(self.proj(self.proj_norm1(inner_tokens.reshape(B, N, -1))))))))))
            # print(outer_tokens[:,:,0])
            fuse=torch.mul(outer_tokens,frame_score)
            # print(fuse[:,:,0])
            # pdb.set_trace()
            outer_tokens = outer_tokens +  fuse # B, N, C 
        return inner_tokens, outer_tokens

class Block_Segment(nn.Module):

    def __init__(self, outer_dim,  outer_num_heads,  mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        # Segment Transformer Block
        self.outer_norm1 = norm_layer(outer_dim)
        self.outer_attn = Attention(
            outer_dim, outer_dim, num_heads=outer_num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.outer_norm2 = norm_layer(outer_dim)
        self.outer_mlp = Mlp(in_features=outer_dim, hidden_features=int(outer_dim * mlp_ratio),
                             out_features=outer_dim, act_layer=act_layer, drop=drop)

    def forward(self, outer_tokens):  
        outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens)))
        outer_tokens = outer_tokens + self.drop_path(self.outer_mlp(self.outer_norm2(outer_tokens)))
        return  outer_tokens

class PatchEmbed(nn.Module):
    """ Image to Visual Word Embedding
    """
    def __init__(self, img_size=[1,7680], patch_size=[1,768], in_chans=1, outer_dim=384, inner_dim=64, inner_stride=[1,128]):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])#10=10*1
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.inner_dim = inner_dim
        self.num_words = math.ceil(patch_size[0] / inner_stride[0]) * math.ceil(patch_size[1] / inner_stride[1])#6=768/128
        # 对于输入的1 * 7680的时序呼吸数据，先执行nn.Unfold的操作：
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)# input_shape (1,1,1,7680)
        # x = self.unfold(x)
        # # output_shape (1,768,10)
        # 这里10=1*10，196表示用1*768的patch可以遍历1*7680次数（1/1 * 7680/768）
        # 然后768=1*768*1，前两个是一个patch的长宽，1是通道。相当于每个patch的pixel(像素)数。
        # 这一步Unfold相当于卷积的“卷”过程用滑窗取出所有的块。
        # print(in_chans, inner_dim,inner_stride)
        self.proj = nn.Conv2d(in_chans, inner_dim, kernel_size=[1,336], padding=[0,144], stride=inner_stride)#进一步划分小Patch

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.unfold(x) # B, Ck2, N 
        # print(x.shape)#(1,768,10)
        # pdb.set_trace()
        x = x.transpose(1, 2).reshape(B * self.num_patches, C, *self.patch_size) # B*N, 1, 1, 768
        # print(x.shape)#(10,1,1,768)
        x = self.proj(x) # B*N, C, 1, 768
        # print(x.shape)#(10,64,1,6)
        # pdb.set_trace()
        x = x.reshape(B * self.num_patches, self.inner_dim, -1).transpose(1, 2) # B*N, 1*6, C
        # print("PatchEmbed后输入特征维度:",x.shape)#(10,6,64)
        # (10,1,1,768) 这个很好理解这是10个patch，每个patch是3通道，长宽是1,768
        # (10,64,1,6)这是卷积后的结果
        # (10,6,64)是变形后的结构
        return x


class HFFT(nn.Module):
    #drop_rate:在segment-level Ptach输入Transformer之前对其做一下dropout 以及 Transformer编码器里面的attention里面的线性映射的dropout和mlp里面的dropout
    #attn_drop_rate:Transformer编码器里面的注意力算完以后做了一个dropout
    #drop_path_rate:Frame-level Patch再Transformer里面经过Attention做了一个dropout，经过MLP又做了一个dropout
    def __init__(self, img_size=[1,7680], patch_size=[1,768], in_chans=1, num_classes=2, outer_dim=384, inner_dim=64,
                 depth_frame=12, depth_seg_emo=12,depth_seg_gender=12,outer_num_heads=6, inner_num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, inner_stride=[1,128]):
        super().__init__()
        outer_dim = make_divisible(outer_dim, outer_num_heads)#384
        inner_dim = make_divisible(inner_dim, inner_num_heads)#64
        self.num_classes = num_classes#2
        self.num_features = self.outer_dim = outer_dim  # num_features for consistency with other models 384

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, outer_dim=outer_dim,
            inner_dim=inner_dim, inner_stride=inner_stride)
        self.num_patches = num_patches = self.patch_embed.num_patches
        num_words = self.patch_embed.num_words
        
        self.proj_norm1 = norm_layer(num_words * inner_dim)
        self.proj = nn.Linear(num_words * inner_dim, outer_dim)
        self.proj_norm2 = norm_layer(outer_dim)
        self.outer_tokens = nn.Parameter(torch.zeros(1, num_patches, outer_dim), requires_grad=False)
        self.outer_pos = nn.Parameter(torch.zeros(1, num_patches, outer_dim))
        self.inner_pos = nn.Parameter(torch.zeros(1, num_words, inner_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)



        dpr_frame = [x.item() for x in torch.linspace(0, drop_path_rate, depth_frame)]  # stochastic depth decay rule
        dpr_seg_emo = [x.item() for x in torch.linspace(0, drop_path_rate, depth_seg_emo)]
        dpr_seg_gender = [x.item() for x in torch.linspace(0, drop_path_rate, depth_seg_gender)]
        blocks_frame,blocks_seg_emo,blocks_seg_gender = [],[],[]

        for i in range(depth_frame):
            blocks_frame.append(Block_Frame(
                outer_dim=outer_dim, inner_dim=inner_dim, inner_num_heads=inner_num_heads,num_words=num_words, 
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr_frame[i], norm_layer=norm_layer))
        self.blocks_frame = nn.ModuleList(blocks_frame)

        for j in range(depth_seg_emo):
            blocks_seg_emo.append(Block_Segment(
                outer_dim=outer_dim,  outer_num_heads=outer_num_heads,  
                mlp_ratio=mlp_ratio,qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, 
                attn_drop=attn_drop_rate, drop_path=dpr_seg_emo[j], norm_layer=norm_layer))
        self.blocks_seg_emo = nn.ModuleList(blocks_seg_emo)

        for k in range(depth_seg_gender):
            blocks_seg_gender.append(Block_Segment(
                outer_dim=outer_dim,  outer_num_heads=outer_num_heads,  
                mlp_ratio=mlp_ratio,qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, 
                attn_drop=attn_drop_rate, drop_path=dpr_seg_gender[k], norm_layer=norm_layer))
        self.blocks_seg_gender = nn.ModuleList(blocks_seg_gender)

        self.norm = norm_layer(outer_dim)

        # Classifier head
        self.head = nn.Linear(outer_dim, num_classes) if num_classes > 0 else nn.Identity()

        # trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.outer_pos, std=.02)
        trunc_normal_(self.inner_pos, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'outer_pos', 'inner_pos', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.outer_dim, num_classes) if num_classes > 0 else nn.Identity()#384-->4

    def forward_features(self, x):
        B = x.shape[0]#batch_size=1
        inner_tokens = self.patch_embed(x) + self.inner_pos # B*N, 1*6, C  #(1*10=10,6,64)  将视觉单词转换为词嵌入
        outer_tokens = self.proj_norm2(self.proj(self.proj_norm1(inner_tokens.reshape(B, self.num_patches, -1))))     #(1,10,384)
        outer_tokens = outer_tokens + self.outer_pos
        outer_tokens = self.pos_drop(outer_tokens)

        for blk in self.blocks_frame:
            inner_tokens, outer_tokens = blk(inner_tokens, outer_tokens)

        for blk2 in self.blocks_seg_emo:
            outer_tokens_emo=blk2(outer_tokens)

        



        outer_tokens_emo = self.norm(outer_tokens_emo)
        
        
      


        # print("inner_tokens(经过frame Transformer后):",inner_tokens.shape)
        # print("outer_tokens_emo(经过segment Transformer后):",outer_tokens_emo.shape)
        # print("outer_tokens_gender(经过segment Transformer后):",outer_tokens_gender.shape)
        return outer_tokens_emo

    def forward(self, x):
        # print(x.shape)
        # pdb.set_trace()
        x_emo = self.forward_features(x)
        # print("经过内外Transformer后的输出cls:",x_emo.shape,x_gen.shape)
        x_emo = self.head(x_emo)
        
        # print("fc以后:",x_emo.shape,x_gen.shape)
        x_emo = torch.mean(x_emo,dim=1)
        
        # print("平均池化以后:",x_emo.shape,x_gen.shape)
        # print(x_emo)
        # print(x_gen)
        # print(dis_loss)
        return x_emo
