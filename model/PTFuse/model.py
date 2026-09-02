import torch
from libs.base_model_d import base_model_d
from model.PTFuse.network import PTFuseGenerator, ModalityDiscriminator
from model.PTFuse.loss import Fusionloss, PATS
from tqdm import tqdm
import clip
import pdb
def arguments():
    args = { 
        '--viinput': 1,
        '--irinput': 1,
        '--beta':10,
        '--gamma':1
    }
    return args

class PTFuse(base_model_d):
    def __init__(self, parser):
        super().__init__(parser)
        parser.add_args(arguments())
        self.args = parser.get_args()
        self.batchsize = self.args.batch_size
        self.viinput = self.args.viinput
        self.irinput = self.args.irinput
        self.beta = self.args.beta
        self.gamma = self.args.gamma

        # 初始化模型
        self.model = PTFuseGenerator().to(self.device)
        self.dis_ir = ModalityDiscriminator().to(self.device)
        self.dis_vis = ModalityDiscriminator().to(self.device)
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.img_loss = Fusionloss().to(self.device)
        # self.contrastive_loss = SelfContrastiveLoss().to(self.device)
        self.pats_loss = PATS().to(self.device)
        self.bce_loss = torch.nn.BCEWithLogitsLoss().to(self.device)
        # 优化器
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.dis_opt = torch.optim.AdamW(
            list(self.dis_ir.parameters()) + list(self.dis_vis.parameters()),
            lr=1e-4,
            betas=(0.5, 0.999),
        )

        self.net_dict = {'name': ['PTFuseGenerator'],
                        'network': [self.model]}

        self.lossdic = dict(constrastive_loss=[], dis_loss=[], gen_loss=[])
        self.paramdict = dict(epoch=[])
        self.valuedictlist = [["loss", self.lossdic], ["param", self.paramdict]]

    def train_epoch(self, dataloader,epoch1):
        self.model.train()
        self.clip_model.eval()
        total_img_loss = 0
        total_dis_loss = 0
        total_contrastive_loss = 0
        for data in tqdm(dataloader, total = len(dataloader),leave = False):
            img_ir, img_vi,text = data
            if img_ir.shape[0] != 2:
                raise ValueError(
                    "PTFuse PATS training requires batch_size=2 to construct "
                    "the paired and mismatched samples."
                )
            img_vi = img_vi.to(self.device)
            img_ir = img_ir[:,0:1,:,:].to(self.device)
            img_vi = self.RGB2Y(img_vi).to(self.device)  # 提取可见光Y通道
            infrared_text = clip.tokenize(text[0]).to(self.device)
            visible_text = clip.tokenize(text[1]).to(self.device)
            fusion_text = clip.tokenize(text[2]).to(self.device)
            target_text = clip.tokenize(text[3]).to(self.device)
            false_text = clip.tokenize(text[4]).to(self.device)
            infrared_text = self.clip_model.encode_text(infrared_text).float()
            visible_text = self.clip_model.encode_text(visible_text).float()
            fusion_text = self.clip_model.encode_text(fusion_text).float()
            target_text = self.clip_model.encode_text(target_text).float()
            false_text = self.clip_model.encode_text(false_text).float()
            fusion,score = self.model(img_ir, img_vi,target_text)

            vis_output = self.dis_vis(img_vi, visible_text)
            ir_output = self.dis_ir(img_ir, infrared_text)
            fus_output_ir = self.dis_ir(fusion.detach(), fusion_text)
            fus_output_vis = self.dis_vis(fusion.detach(), fusion_text)
            vis_loss = self.bce_loss(vis_output, torch.Tensor(vis_output.shape).uniform_(0.7, 1.0).to(self.device))
            ir_loss = self.bce_loss(ir_output, torch.Tensor(ir_output.shape).uniform_(0.7, 1.0).to(self.device))
            fusion_loss_ir = self.bce_loss(
                fus_output_ir, torch.Tensor(fus_output_ir.shape).uniform_(0.0, 0.3).to(self.device)
            )
            fusion_loss_vis = self.bce_loss(
                fus_output_vis, torch.Tensor(fus_output_vis.shape).uniform_(0.0, 0.3).to(self.device)
            )
            dis_loss = vis_loss + ir_loss + fusion_loss_ir + fusion_loss_vis
            self.dis_opt.zero_grad()
            dis_loss.backward()
            self.dis_opt.step()
            fus_output_ir = self.dis_ir(fusion, fusion_text)
            fus_output_vis = self.dis_vis(fusion, fusion_text)
            gen_loss = self.bce_loss(
                fus_output_ir, torch.Tensor(fus_output_ir.shape).uniform_(0.7, 1.0).to(self.device)
            ) + self.bce_loss(
                fus_output_vis, torch.Tensor(fus_output_vis.shape).uniform_(0.7, 1.0).to(self.device)
            )
            img_loss = self.img_loss(img_vi, img_ir, fusion)
            # contrastive_loss = self.contrastive_loss(fusion)
            # PATS: mismatched source pairs provide pseudo-fused negative samples.
            fusion_false1, _ = self.model(
                img_ir[0].unsqueeze(0), img_vi[1].unsqueeze(0),
                false_text[0].unsqueeze(0)
            )
            fusion_false2, _ = self.model(
                img_ir[1].unsqueeze(0), img_vi[0].unsqueeze(0),
                false_text[1].unsqueeze(0)
            )
            fusion_all = torch.cat(
                [fusion[0].unsqueeze(0), fusion_false1,
                 fusion[1].unsqueeze(0), fusion_false2], dim=0
            )
            # 计算对比损失
            contrastive_loss = self.pats_loss(fusion_all)

            g_total_loss = gen_loss + self.beta * img_loss + self.gamma * contrastive_loss
            self.opt.zero_grad()
            g_total_loss.backward(retain_graph=False)
            self.opt.step()

            # 记录loss
            total_dis_loss += dis_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            total_img_loss += g_total_loss.item()
        self.lossdic['constrastive_loss'].append(total_contrastive_loss)
        self.lossdic['dis_loss'].append(total_dis_loss)
        self.lossdic['gen_loss'].append(total_img_loss)

        print('Epoch {}/{}, loss = {:.6f}, contrastive_loss = {:.6f}, dis_loss = {:.6f}'.format(epoch1, self.args.epochs,total_img_loss,total_contrastive_loss,total_dis_loss))  

    def train(self,train_loader,test_loader):
        if self.args.load_weights:
            self.load_weights()
        best = float(0)
        for epoch in range(1,self.epochs+1):
            self.train_epoch(train_loader, epoch)
            if (self.train_with_test) and (epoch%self.train_with_test==0 or epoch == self.epochs):
                img_fus = self.inference(test_loader)
                self.save_weights(epoch)
                if self.args.control_save_img:
                    self.save_img(img_fus,epoch)

    def test(self, train_loader, test_loader):
        self.load_weights()
        best = float(0)
        img_fus = self.inference(test_loader)
        assert self.args.phase == 'test', '''Call test function but phase is not testing.'''
        if self.args.control_save_img:
            self.save_img(img_fus, 'test')
 
    def inference(self, dataloader):
        with torch.no_grad():
            img_fus_libs = [] #RoadScene和TNO数据集的尺寸不一致，无法初始化tensor的方式实现，只能append
            self.model.eval() 
            self.clip_model.eval()
            for i, data in enumerate(tqdm(dataloader, total=len(dataloader), leave=False)):
                img_ir, img_vi,text = data
                img_vi = img_vi.to(self.device)
                img_ir = img_ir[:,0:1,:,:].to(self.device)
                img_vi_y = self.RGB2Y(img_vi).to(self.device)  # 提取可见光Y通道
                target_text = clip.tokenize(text[3]).to(self.device)
                target_text = self.clip_model.encode_text(target_text).float()
                
                img_fus,score = self.model(img_ir,img_vi_y,target_text)

                img_fus = (img_fus - torch.min(img_fus)) / (torch.max(img_fus) - torch.min(img_fus)) # 归一化融合结果

                # 保存融合图像
                img_fus = self.Y2RGB(img_fus,img_vi)
                for j in range(img_fus.shape[0]):
                    img_fus_libs.append(img_fus[j]) 
            return img_fus_libs

# Example: python main.py PTFuse --data MSRS --phase train --batch_size 2 --epochs 100 --aug_methods random_crop RVF RHF --convert_mode RGB --prompt discussion
