import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import threading
from dataset import *
import time
from collections import OrderedDict
from model.SCTransNet import SCTransNet as SCTransNet
# from loss import *
import model.Config as config
import numpy as np
import torch
from skimage import measure
from metrics import eval_iou
from model.PromptLearner_V import build_custom_clip
from metrics import eval_iou_pd_fa

parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD train")
parser.add_argument("--model_names", default=['SCTransNet'], type=list, help="'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'RISTDnet'")
parser.add_argument("--model_clip", default='ViT-B/32', type=str, help="'ViT-B/16', 'ViT-B/32', 'ViT-L/14'")
parser.add_argument("--dataset_name", default='IRSTD-1K', type=str,help="'IRSTD-1K',  'NUAA-SIRST',  'NUDT-SIRST',  'SIRST3'")
parser.add_argument("--device", default="cuda:1", type=str)
# SIRST3： NUAA NUDT IRSTD-1K
parser.add_argument("--optimizer_name", default='Adam', type=str, help="optimizer name: AdamW, Adam, Adagrad, SGD")
parser.add_argument("--epochs", default=800, type=int, help="optimizer name: AdamW, Adam, Adagrad, SGD")
parser.add_argument("--begin_test", default=250, type=int)
parser.add_argument("--every_test", default=1, type=int)
parser.add_argument("--every_save_pth", default=1000, type=int)
parser.add_argument("--every_print", default=1, type=int)
parser.add_argument("--dataset_dir", default=r'../dataset')
parser.add_argument("--batchSize", type=int, default=16, help="Training batch sizse,default=16")
parser.add_argument("--patchSize", type=int, default=256, help="Training patch size")
parser.add_argument("--save", default=r'./testlog/sig_all', type=str, help="Save path of checkpoints")
parser.add_argument("--log_dir", type=str, default="./otherlogs/SCTransNet", help='path of log files')
parser.add_argument("--img_norm_cfg", default=None, type=dict)
parser.add_argument("--threads", type=int, default=8, help="Number of threads for data loader to use")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test")
parser.add_argument("--seed", type=int, default=42, help="Threshold for test")
parser.add_argument("--resume", default=False, type=list, help="Resume from exisiting checkpoints (default: None)")

global opt
opt = parser.parse_args()

seed_pytorch(opt.seed)

config_vit = config.get_SCTrans_config()


test_set = TestSetLoader(opt.dataset_dir, opt.dataset_name, opt.dataset_name, img_norm_cfg=opt.img_norm_cfg,model=opt.model_clip)
test_loader = DataLoader(dataset=test_set, num_workers=16, batch_size=
                             2, shuffle=False)





class Net(nn.Module):
    def __init__(self, model_name, mode):
        super(Net, self).__init__()
        self.model_name = model_name
        # ************************************************loss*************************************************#
        self.cal_loss = nn.BCELoss(size_average=True)
        if model_name == 'SCTransNet':
            if mode == 'train':
                self.model = SCTransNet(config_vit, mode='train', deepsuper=True)
            else:
                self.model = SCTransNet(config_vit, mode='test', deepsuper=True)
    def build_clip(self):
        custom_clip=build_custom_clip(config_vit)
        self.model.init_clip(custom_clip)
    def forward(self, img,feature=None,is_train=True):
        return self.model(img, feature=feature, is_train=is_train)

    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)

        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss

    
     

       
    
savepath='sweep_logs/product_POSmiddle_NCTX8_CLASSsmall_target/IRSTD-1K/SCTransNet_359_best.pth.tar'

net = Net(model_name="SCTransNet", mode='train').to(opt.device)
net.build_clip()
ckpt = torch.load(savepath)
net.load_state_dict(ckpt['state_dict'])
net.eval()

with torch.no_grad():
    for i, (img, gt_mask, feature,size, _) in enumerate(test_loader):

        image = img.to(opt.device)
        target = gt_mask.to(opt.device)
        feature = feature.to(opt.device)
        start_time = time.time()
        pred = net(image,feature,is_train=False)
        a,b,c=eval_iou_pd_fa(net, test_loader, opt.device, threshold=opt.threshold)
        print(f"iou: {a}, pd: {b}, fa: {c}")
        


