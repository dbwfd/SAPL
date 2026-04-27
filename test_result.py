import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import threading
import time
from collections import OrderedDict
from model.SCTransNet import SCTransNet as SCTransNet
from model.PromptLearner_V import build_custom_clip
# from loss import *
import model.Config as config
import numpy as np
import torch
from skimage import measure
from metrics import *
from model.SCTransNet import SCTransNet as SCTransNet
from tqdm import tqdm
from model.PromptLearner_V import build_custom_clip
from dataset import TestSetLoader
from metrics import *
from utils import *





os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD test")
parser.add_argument('--ROC_thr', type=int, default=10, help='num')
parser.add_argument("--model_name", default='sctransnet', type=str,
                    help="model_name: 'ACM', 'Ours01', 'DNANet', 'ISNet', 'ACMNet', 'SCTRANSNET', 'ISTDU-Net', 'U-Net', 'RISTDnet'")
parser.add_argument("--pth_dirs", default=['IRSTD-1K/SCTransNet_497_best.pth.tar'], type=list)
parser.add_argument("--dataset_dir", default=r'../dataset', type=str, help="train_dataset_dir")
parser.add_argument("--dataset_names", default='IRSTD-1K', type=str,
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--save_img", default=True, type=bool, help="save image of or not")
parser.add_argument("--save_img_dir", type=str, default=r'./presults/',
                    help="path of saved image")
parser.add_argument("--save_log", type=str, default=r'log/', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--device", type=str, default='cuda:2', help="device")
parser.add_argument("--model_clip", default='ViT-L/14', type=str, help="'ViT-B/16', 'ViT-B/32', 'ViT-L/14'")

global testopt
testopt = parser.parse_args()
config_vit = config.get_SCTrans_config()

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


def test(test_net=None,device=None,dataset_name=None,dataset=None,model_name=None):
    

    test_loader = DataLoader(dataset=test_set, num_workers=8, batch_size=1, shuffle=False)
    
    config_vit = config.get_SCTrans_config()
    if model_name is not None:
        testopt.model_name = model_name
    print("Current model_name:", testopt.model_name)


    # CPU
    # net = SCTransNet(config_vit, mode='test', deepsuper=True)
    # state_dict = torch.load(opt.pth_dir, map_location='cpu')
    # CUDA
    if test_net is None:
        net = SCTransNet(config_vit, mode='test', deepsuper=True).to(testopt.device)
        
        state_dict = torch.load(testopt.pth_dir)

        new_state_dict = OrderedDict()
        #
        for k, v in state_dict['state_dict'].items():
            name = k[6:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
            new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
        net.load_state_dict(new_state_dict)
    else:   
        net = test_net
    net.eval()
    # a,b,c=eval_iou_pd_fa(net, test_loader, testopt.device, threshold=testopt.threshold)
    # print("IoU: %.4f, Pd: %.4f, Fa: %.4f" % (a, b, c))
    test_prompt(net, test_loader, testopt.device, threshold=testopt.threshold, save_dir=testopt.save_img_dir, model_name="SAPL", test_dataset_name=testopt.dataset_names)
    




if __name__ == '__main__':
    

    
    config_vit.device = testopt.device
    config_vit.TOKEN_POSITION = "end"  
    
    config_vit.N_CTX = 6
    
    config_vit.CLIP = testopt.model_clip
    config_vit.text_size = 768 if testopt.model_clip=='ViT-L/14' else 512
    
   
   
    
    config_vit.transformer.IS_FEEDFORWARD=True
    config_vit.Filter_DROPOUT=0.0
    # config_vit.base_channel=32
    # config_vit.KV_size=480
    print("Current config_vit:",config_vit)

    savepath = "sweep_logs_iv14/product_POSend_NCTX6_FFTrue_DP0.0/IRSTD-1K/SCTransNet_218_best.pth.tar"
    test_set = TestSetLoader(testopt.dataset_dir, testopt.dataset_names, testopt.dataset_names, img_norm_cfg=testopt.img_norm_cfg,model=testopt.model_clip,split="sc")

    net = Net(model_name="SCTransNet", mode='train').to(testopt.device)
    net.build_clip()
    ckpt = torch.load(savepath)
    net.load_state_dict(ckpt['state_dict'])

    test(test_net=net,device=testopt.device,dataset_name=testopt.dataset_names,dataset=test_set)
    
    