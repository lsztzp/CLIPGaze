import torch.nn.functional as F
from PIL import Image
import os
from os.path import isdir
import numpy as np
from torchvision import transforms
import math
from os.path import join, isfile
import torch

def rescaled_pos_emb(model,new_size,token_shape=(14,14)):
    assert len(new_size) == 2

    a = model.positional_embedding[1:].T.view(1, 768, * token_shape)
    b = F.interpolate(a, new_size, mode='bicubic', align_corners=False).squeeze(0).view(768, new_size[0] * new_size[1]).T
    return torch.cat([model.positional_embedding[:1], b])

def forward_multihead_attention(x, b, with_aff=False, attn_mask=None):
    """
    Simplified version of multihead attention (taken from torch source code but without tons of if clauses).
    The mlp and layer norm come from CLIP.
    x: input.
    b: multihead attention module.
    """

    x_ = b.ln_1(x)
    q, k, v = F.linear(x_, b.attn.in_proj_weight, b.attn.in_proj_bias).chunk(3, dim=-1)
    tgt_len, bsz, embed_dim = q.size()

    head_dim = embed_dim // b.attn.num_heads
    scaling = float(head_dim) ** -0.5

    q = q.contiguous().view(tgt_len, bsz * b.attn.num_heads, b.attn.head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * b.attn.num_heads, b.attn.head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * b.attn.num_heads, b.attn.head_dim).transpose(0, 1)

    q = q * scaling

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))  # n_heads * batch_size, tokens^2, tokens^2
    if attn_mask is not None:

        attn_mask_type, attn_mask = attn_mask
        n_heads = attn_output_weights.size(0) // attn_mask.size(0)
        attn_mask = attn_mask.repeat(n_heads, 1)

        if attn_mask_type == 'cls_token':
            # the mask only affects similarities compared to the readout-token.
            attn_output_weights[:, 0, 1:] = attn_output_weights[:, 0, 1:] * attn_mask[None, ...]
            # attn_output_weights[:, 0, 0] = 0*attn_output_weights[:, 0, 0]

        if attn_mask_type == 'all':
            # print(attn_output_weights.shape, attn_mask[:, None].shape)
            attn_output_weights[:, 1:, 1:] = attn_output_weights[:, 1:, 1:] * attn_mask[:, None]

    attn_output_weights = torch.softmax(attn_output_weights, dim=-1)

    attn_output = torch.bmm(attn_output_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = b.attn.out_proj(attn_output)

    x = x + attn_output
    x = x + b.mlp(b.ln_2(x))

    if with_aff:
        return x, attn_output_weights
    else:
        return x

def visual_forward(model, x_inp, extract_layers=(), skip=False, mask=None,token_shape=(14,14)):
    with torch.no_grad():
        # inp_size = x_inp.shape[2:]
        x = model.conv1(x_inp)  # shape = [*, width, grid, grid]

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat([model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                            dtype=x.dtype, device=x.device), x],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]

        standard_n_tokens = 50 if model.conv1.kernel_size[0] == 32 else 197

        if x.shape[1] != standard_n_tokens:
            new_shape = (20, 32)
            x = x + rescaled_pos_emb(model,(new_shape[0], new_shape[1]),token_shape=token_shape).to(x.dtype)[None, :, :]
        else:
            x = x + model.positional_embedding.to(x.dtype)
        x = model.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        activations, affinities = [], []
        for i, res_block in enumerate(model.transformer.resblocks):

            if mask is not None:
                mask_layer, mask_type, mask_tensor = mask
                if mask_layer == i or mask_layer == 'all':
                    # import ipdb; ipdb.set_trace()
                    size = int(math.sqrt(x.shape[0] - 1))

                    attn_mask = (mask_type, F.interpolate(mask_tensor.unsqueeze(1).float(), (size, size)).view(
                        mask_tensor.shape[0], size * size))

                else:
                    attn_mask = None
            else:
                attn_mask = None

            x, aff_per_head = forward_multihead_attention(x, res_block, with_aff=True, attn_mask=attn_mask)

            if i in extract_layers:
                affinities += [aff_per_head]
                activations += [x]

            if len(extract_layers) > 0 and i == max(extract_layers) and skip:
                print('early skip')
                break

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = model.ln_post(x[:, 0, :])

        if model.proj is not None:
            x = x @ model.proj

        return x, activations, affinities


def get_image_embedding(dataset_path, save_path, model, clip_model, device='cuda:0', overwrite=False,token_shape=(14,14)):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((320, 512)),
    ])

    src_path = join(dataset_path, 'images/')
    target_path = save_path + 'featuremaps/'
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    folders = [i for i in os.listdir(src_path) if isdir(join(src_path, i))]

    for folder in folders:
        print(folder)
        if not (os.path.exists(join(target_path, '_'.join(folder.split(' ')))) and os.path.isdir(join(target_path, '_'.join(folder.split(' '))))):
            os.mkdir(join(target_path, '_'.join(folder.split(' '))))
        files = [i for i in os.listdir(join(src_path, folder)) if
                 isfile(join(src_path, folder, i)) and i.endswith('.jpg')]

        for f in files:
            if overwrite == False and os.path.exists(join(target_path, folder, f.replace('jpg', 'pth'))):
                continue
            PIL_image = Image.open(join(src_path, folder, f)).convert("RGB")
            image_ftrs = transform(PIL_image).unsqueeze(0)

            with torch.no_grad():
                visual_q, activations, _ = visual_forward(model, image_ftrs, extract_layers=[3,6,9])

                activations = [x.permute(1,0,2) for x in activations]
                torch.save(activations, join(target_path, '_'.join(folder.split(' ')), f.replace('jpg', 'pth')))


def get_text_embedding(dataset_path, save_path, model):
    src_path = join(dataset_path, 'images/')
    tasks = [' '.join(i.split('_')) for i in os.listdir(src_path) if isdir(join(src_path, i))]
    embed_dict = {}
    for task in tasks:
        # task_promt = 'a photo of a ' + task
        task_promt = task
        print(task_promt)
        text_tokens = clip.tokenize(task_promt)
        cond = model.encode_text(text_tokens)
        embed_dict[task]=cond.squeeze().cpu().detach().numpy()

    with open(join(save_path, 'dataset/embeddings.npy'), 'wb') as f:
        np.save(f, embed_dict, allow_pickle=True)
        f.close()

if __name__=="__main__":
    import clip
    dataset_path='/01-Datasets/01-ScanPath-Datasets/coco_search18/raw/COCOSearch18/'
    save_path='/01-Datasets/01-ScanPath-Datasets/coco_search18/vit-L14-336/'

    device = "cpu"
    version = "ViT-L/14@336px"
    token_shape = {'ViT-B/32': (7, 7), 'ViT-B/16': (14, 14), 'ViT-L/14@336px': (24,24)}[version]
    clip_model, _ = clip.load(version, device=device, jit=False)
    model = clip_model.visual

    get_image_embedding(dataset_path, save_path, model, device='cuda:0', overwrite=False,token_shape=token_shape,clip_model=clip_model)
    get_text_embedding(dataset_path, save_path, clip_model)