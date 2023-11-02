import torch
import numpy as np
import clip
import os 

def prepare_vocab(opt, matcontent, zst_mode=False):
    vocab = []
    if zst_mode:
        dataset = opt.zstfrom
    else:
        dataset = opt.dataset    
    for cls_name in matcontent['allclasses_names']:
        if dataset == 'CUB':
            vocab.append(cls_name[0][0][4:])
        elif dataset == 'SUN':
            vocab.append(cls_name[0][0])
        elif dataset == 'AWA2':
            vocab.append(cls_name[0][0].replace('+', '_'))
        else:
            raise NotImplementedError
        
    return vocab

def prep_imagenet_vocab(imagenet_vocab):
    pruned_vocab = []
    for label in imagenet_vocab:
        first_label = label.split(",")[0]
        pruned_vocab.append(first_label)
    return pruned_vocab

def get_clip_embeddings(opt, vocab):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN101", device=device)
    
    clip_embeddings = []
    input_text = []
    #prompt = 'an image of a'
    prompt = 'a photo of a'
    templates80 = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
    ]

    for word in vocab:
        word = word.replace('_', ' ')
        word = word.lower()
        if word[0] in ['a', 'e', 'i', 'o', 'A', 'E', 'I', 'O']:
            input_text.append(prompt + 'n ' + word)
        else:
            input_text.append(prompt + ' ' + word)
    text = clip.tokenize(input_text).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text)
    
    embeddings = text_features / torch.norm(text_features.float(), dim=-1).unsqueeze(-1)
    
    return embeddings
