import argparse
import time
import os
import cv2
import wandb
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"]= "100"
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import *
from metrics import *
from utils import *
import model.Config as config
from torch.utils.tensorboard import SummaryWriter
from model.SCTransNet import SCTransNet as SCTransNet
from tqdm import tqdm
from model.PromptLearner_V import build_custom_clip
from test import test

parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD train")
parser.add_argument("--model_names", default=['SCTransNet'], type=list, help="'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'RISTDnet'")
parser.add_argument("--model_clip", default='ViT-B/32', type=str, help="'ViT-B/16', 'ViT-B/32', 'ViT-L/14'")
parser.add_argument("--dataset_names", default=['NUAA-SIRST'], type=list,help="'IRSTD-1K',  'NUAA-SIRST',  'NUDT-SIRST',  'SIRST3'")
parser.add_argument("--device", default="cuda:1", type=str)
# SIRST3： NUAA NUDT IRSTD-1K
parser.add_argument("--optimizer_name", default='Adam', type=str, help="optimizer name: AdamW, Adam, Adagrad, SGD")
parser.add_argument("--epochs", default=900, type=int, help="optimizer name: AdamW, Adam, Adagrad, SGD")
parser.add_argument("--begin_test", default=150, type=int)
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
parser.add_argument("--seed", type=int, default=42, help="seed")
parser.add_argument("--resume", default=False, type=list, help="Resume from exisiting checkpoints (default: None)")

global opt
opt = parser.parse_args()

seed_pytorch(opt.seed)

config_vit = config.get_SCTrans_config()


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def train(run):
    seed_pytorch(opt.seed)
    
    train_set = TrainSetLoader(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name, patch_size=opt.patchSize,
                               img_norm_cfg=opt.img_norm_cfg,model=opt.model_clip)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    test_set = TestSetLoader(opt.dataset_dir, opt.dataset_name, opt.dataset_name, img_norm_cfg=opt.img_norm_cfg,model=opt.model_clip)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    net = Net(model_name=opt.model_name, mode='train').to(opt.device)
    net.apply(weights_init_kaiming)
    net.build_clip()
    
    net.train()

    epoch_state = 0
    total_loss_list = []
    total_loss_epoch = []

    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)
    writer = SummaryWriter(opt.log_dir)

    if opt.resume:
        # for resume_pth in opt.resume:
        #     if opt.dataset_name in resume_pth and opt.model_name in resume_pth:
        ckpt = torch.load('XX\\UCT04_best.pth.tar')
        net.load_state_dict(ckpt['state_dict'])
        epoch_state = ckpt['epoch']
        total_loss_list = ckpt['total_loss']
        # for i in range(len(opt.scheduler_settings['step'])):
        #     opt.scheduler_settings['step'][i] = opt.scheduler_settings['step'][i] - ckpt['epoch']

    ### Default settings of SCTransNet
    if opt.optimizer_name == 'Adam':
        opt.optimizer_settings = {'lr': 0.001}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs': opt.epochs, 'eta_min': 1e-5, 'last_epoch': -1}

    ### Default settings of DNANet
    if opt.optimizer_name == 'Adagrad':
        opt.optimizer_settings = {'lr': 0.05}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs': opt.epochs, 'min_lr': 1e-5}

    ### Default settings of EGEUNet
    if opt.optimizer_name == 'AdamW':
        opt.optimizer_settings = {'lr': 0.001, 'betas': (0.9, 0.999), "eps": 1e-8, "weight_decay": 1e-2,
                                  "amsgrad": False}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs': opt.epochs, 'T_max': 50, 'eta_min': 1e-5, 'last_epoch': -1}

    opt.nEpochs = opt.scheduler_settings['epochs']

    optimizer, scheduler = get_optimizer(net, opt.optimizer_name, opt.scheduler_name, opt.optimizer_settings,
                                         opt.scheduler_settings)

    best_mIOU = 0
    best_pd = 0
    best_fa = 10000
    miou = 0
    pd=0
    fa=10000
    for idx_epoch in tqdm(range(epoch_state, opt.nEpochs),desc="Epochs"):
        net.train()
        
        
        for idx_iter, (img, gt_mask,feature) in enumerate(tqdm(train_loader)):
            img, gt_mask,feature = Variable(img).to(opt.device), Variable(gt_mask).to(opt.device), Variable(feature).to(opt.device)
           
            preds = net.forward(img, feature, is_train=True)
            loss = net.loss(preds, gt_mask)
            total_loss_epoch.append(loss.detach().cpu())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        if (idx_epoch + 1) % opt.every_print == 0:
            total_loss_list.append(float(np.array(total_loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f, lr---%f,'
                  % (idx_epoch + 1, total_loss_list[-1], scheduler.get_last_lr()[0]))
            opt.f.write(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,\n'
                        % (idx_epoch + 1, total_loss_list[-1]))
            total_loss_epoch = []
            # Log the scalar values
            print(f"loss: {total_loss_list[-1]}, lr: {scheduler.get_last_lr()[0]}")
            writer.add_scalar('loss', total_loss_list[-1], idx_epoch + 1)
            writer.add_scalar('lr', scheduler.get_last_lr()[0], idx_epoch + 1)

        
        if (idx_epoch + 1) >= opt.begin_test and (idx_epoch + 1) % opt.every_test == 0:
            
            miou,pd,fa=eval_iou_pd_fa(model=net, dataloader=test_loader, device=opt.device, threshold=opt.threshold)
                # results2 = eval_PD_FA.get()
            print(f"mIoU: {miou}, PD: {pd}, FA: {fa}")

            if pd > best_pd:
                best_pd = pd
            if fa < best_fa:
                best_fa = fa
            
            if miou > best_mIOU:
                best_mIOU = miou
            
           
                print('------save the best model epoch', opt.model_name,'_%d ------' % (idx_epoch + 1))
                opt.f.write("the best model epoch \t" + str(idx_epoch + 1) + '\n')
                print("mIoU:\t" + str(best_mIOU))
           
                opt.f.write("mIoU:\t" + str(best_mIOU) + '\n')

          
                save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '_' + 'best' + '.pth.tar'
                save_checkpoint({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.state_dict(),
                    'total_loss': total_loss_list,
                }, save_pth)

        run.log({
                    "epoch":idx_epoch+1,
                    "mIoU": miou,
                    "bestmIoU": best_mIOU,
                    "pd": pd,
                    "bestpd": best_pd,
                    "fa": fa,
                    "bestfa": best_fa,}) 


        # last epoch
        if (idx_epoch + 1) == opt.nEpochs :
            return save_pth        
           
            





def save_checkpoint(state, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(state, save_path)
    return save_path


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

sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "bestmIoU"},
    "parameters": {
        # "text_pos": {"values": [0, 1,2]}, # 0:skip 1:up sample 2:both
        "TOKEN_POSITION": {"values": [ "middle", "end"]}, # prompt position
        "filter_type": {"values": ["product"]},
        "N_CTX": {"values": [4,6,8,10]},
        "CTX_INIT": {"values": [None]},
        "CLASS_NAME": {"values": ["small target"]},
        "IS_VISUAL": {"values": [True]},
        "is_relu": {"values": [False]},
        "is_feedforward": {"values": [True]},
        "drop_out": {"values": [0.0]},
        "base_channel": {"values": [24,48]},
          # filter type
    },
}

def main():
    seed_pytorch(opt.seed)
    with wandb.init(project="text") as run:
        run.config
        # if run.config.text_pos == 0:
        #     config_vit.IS_TEXTINSKIP = True
        #     config_vit.IS_CAFILTER=False
        # elif run.config.text_pos == 1:
        #     config_vit.IS_TEXTINSKIP = False
        #     config_vit.IS_CAFILTER=True
        # elif run.config.text_pos == 2:
        #     config_vit.IS_TEXTINSKIP = True 
        #     config_vit.IS_CAFILTER=True 

        config_vit.TOKEN_POSITION = run.config.TOKEN_POSITION   
        config_vit.Filter_type = run.config.filter_type  
        config_vit.N_CTX = run.config.N_CTX
        config_vit.CTX_INIT = run.config.CTX_INIT 
        config_vit.CLIP = opt.model_clip
        config_vit.text_size = 768 if opt.model_clip=='ViT-L/14' else 512
        
        class_list=[]
        class_list.append(run.config.CLASS_NAME)
        config_vit.CLASS_NAME = class_list
        config_vit.IS_VISUAL=run.config.IS_VISUAL
        config_vit.device = opt.device
        config_vit.IS_RELU=run.config.is_relu
        config_vit.transformer.IS_FEEDFORWARD=run.config.is_feedforward
        config_vit.Filter_DROPOUT=run.config.drop_out
        config_vit.base_channel= run.config.base_channel
        if config_vit.base_channel==24:
            config_vit.KV_size=360
        elif config_vit.base_channel==48:
            config_vit.KV_size=720
        print("Current config_vit:",config_vit)
        log_dir = f"./sweep_logs_nuaa/{config_vit.Filter_type}_POS{config_vit.TOKEN_POSITION}_NCTX{config_vit.N_CTX}_CLASS{config_vit.CLASS_NAME[0].replace(' ','_')}"
        opt.save = log_dir
        for dataset_name in opt.dataset_names:
            opt.dataset_name = dataset_name
            for model_name in opt.model_names:
                opt.model_name = model_name

                if not os.path.exists(opt.save):
                    os.makedirs(opt.save)
                opt.f = open(opt.save + '/' + opt.dataset_name + '_' + opt.model_name + '_' + (time.ctime()).replace(' ',
                                                                                                                    '_').replace(
                    ':', '_') + '.txt', 'w')
                print(opt.dataset_name + '\t' + opt.model_name)
                savepath=train(run)
                print('\n')
                # opt.f.close()
                # net = Net(model_name=opt.model_name, mode='train').to(opt.device)
                # net.build_clip()
                # ckpt = torch.load(savepath)
                # net.load_state_dict(ckpt['state_dict'])
                # test_set = TestSetLoader(opt.dataset_dir, opt.dataset_name, opt.dataset_name, img_norm_cfg=opt.img_norm_cfg,model=opt.model_clip)
                # mIOU,nIoU,precision,recall,F1_score=test(test_net=net,device=opt.device,dataset_name=opt.dataset_name,dataset=test_set)
                # print(f"Test mIOU: {mIOU}, nIoU: {nIoU}, Precision: {precision}, Recall: {recall}, F1_score: {F1_score}")
                # run.log({"tbestmIoU": mIOU,"tbestnIoU": nIoU,"tbestPrecision": precision,"tbestRecall": recall,"tbestF1_score": F1_score})

                

sweep_id = wandb.sweep(sweep=sweep_configuration, project="text")

wandb.agent(sweep_id, function=main, count=1000, project="text")
