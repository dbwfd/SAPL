import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from .myclip import clip,tokenize
from .myclip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from collections import OrderedDict


_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.CLIP
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class TextEncoder(nn.Module): 
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, is_background=False,ctx_init="a photo of a"):
        super().__init__()
        
        self.is_background = is_background
        n_cls = len(classnames)
        n_ctx = cfg.N_CTX
        

        ctx_between="in the"
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(cfg.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
            if is_background:
                                
                n_ctx_between = len(ctx_between.split(" "))
                prompt = clip.tokenize(ctx_between).to(cfg.device)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors_between = embedding[0, 1 : 1 + n_ctx_between, :]
                prompt_between = ctx_between

        else:
            # random initialization
            # if cfg.TRAINER.COOP.CSC:
            #     print("Initializing class-specific contexts")
            #     ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            # else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        if is_background:
            self.ctx_between = nn.Parameter(ctx_vectors_between)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        if is_background:
            prompts = [prompt_prefix + " " + name + " " + prompt_between + " " + "background." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(cfg.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if not is_background:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        else:
            self.register_buffer("token_inffix",embedding[:,1+n_ctx:1+n_ctx+2,:]) # CLS
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 2 + n_ctx_between :, :])  # background, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        if is_background:
            self.n_ctx_between = n_ctx_between
         
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TOKEN_POSITION
        print(f"Class token position: {self.class_token_position}")

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        if self.is_background:
            ctx_between = self.ctx_between
            if ctx_between.dim() ==2:
                ctx_between = ctx_between.unsqueeze(0).expand(self.n_cls,-1,-1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            if hasattr(self, 'token_inffix'):
                inffix = self.token_inffix
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        inffix,  # (n_cls, 2, dim)
                        ctx_between, # (n_cls, n_ctx_between, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )
            

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts
    
class CustomCLIP(nn.Module):


    def __init__(self, cfg, classnames,sentence, clip_model):
        super().__init__()
        self.is_prompt_learner = cfg.IS_PROMPT_LEARNER
        self.cfg = cfg
       
        if self.is_prompt_learner:
            self.prompt_learner = PromptLearner(cfg, classnames, clip_model, is_background=cfg.IS_BACKGROUND,ctx_init=cfg.CTX_INIT)
            self.tokenized_prompts = self.prompt_learner.tokenized_prompts
            self.text_encoder = TextEncoder(clip_model)
        else:
            self.text_encoder = clip_model.encode_text
            self.sentence = sentence
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self,):
       
        if self.is_prompt_learner:
            prompts = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            text_features = self.text_encoder(prompts, tokenized_prompts)
        else:
            text = tokenize(self.sentence).to(self.cfg.device)
            text_features = self.text_encoder(text)
            text_features = text_features.to(torch.float32)

        


        return text_features.unsqueeze(0)
    
def build_custom_clip(cfg):
    
   
    
    sentences = "A photo of a small target in the background."

    print(f"Loading CLIP (backbone: {cfg.CLIP})")
    device = cfg.device
    
    clip_model = load_clip_to_cpu(cfg).to(device)
    clip_model.float()
    
        
    # if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
    #     # CLIP's default precision is fp16
    #     clip_model.float()

    print("Building custom CLIP")
    if cfg.IS_PROMPT_LEARNER:
        print("Using Prompt Learner")
        model = CustomCLIP(cfg, cfg.CLASS_NAME, sentences, clip_model)
        print("Turning off gradients in both the image and the text encoder")
        for name, param in model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
    else:
        print("Not using Prompt Learner")
      
        model = CustomCLIP(cfg, cfg.CLASS_NAME, sentences, clip_model)
        model.requires_grad_(False)

   

    model.to(device)

   
    # # NOTE: only give prompt_learner to the optimizer
    # self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
    # self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
    # self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

    # self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

    return model