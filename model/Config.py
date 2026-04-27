# -*- coding: utf-8 -*-
# @Author  : Shuai Yuan
# @File    : Config.py
# @Software: PyCharm
# coding=utf-8
import os
import torch
import time
import ml_collections




##########################################################################
# SCTrans configs
##########################################################################
def get_SCTrans_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 480  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.transformer.num_delayers = 1
    config.transformer.IS_MUL=True
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 32  # base channel of U-Net
    config.n_classes = 1
    config.CLIP = "ViT-B/32"
    config.device = "cuda:1" if torch.cuda.is_available() else "cpu"
    config.IS_PROMPT_LEARNER = True
    config.is_spitial_transformer_decoder = True
    config.transformer.SATT=False
    config.transformer.IS_FEEDFORWARD=False
    config.IS_WBFILTER=False
    config.IS_CAFILTER=True
    config.IS_FILTER=True
    config.IS_TEXTINSKIP=False
    config.IS_TEXTINDSKIP=True
    config.IS_TEXTINDBASE=False
    config.text_size = 512
    config.hidden_size = 512
    config.IS_BACKGROUND = False
    config.IS_T2V=True
    config.N_CTX=8
    config.PREC = "fp32" 
    config.TOKEN_POSITION = "middle" # "front" "middle" "end"
    config.Filter_type="product"  # "sig" "mul" "product"
    config.CTX_INIT=None 
    config.CLASS_NAME=["small target"]  # "a photo of a small target in the background"
    config.IS_VISUAL=True
    config.IS_RELU=False
    config.Filter_DROPOUT=0.0
    # ********** unused **********
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    return config
