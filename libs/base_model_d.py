import os
import torch
import copy
import visdom
import warnings
import torch.nn as nn
import imageio
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from ptflops import get_model_complexity_info
from Vis import Netinfo  ###change
from libs.visualizer import visualizer
import numpy as np
import pdb
def arguments():
    args = {'--epochs':20,
    '--batch_size':16,
    '--lr':1e-4,
    '--test_weight_choose':'best',
    '--weight_path':'None',#yuan None
    '--test_interval':1,###节省时间，两个epoch计算一次
    '--time_avg':10}
    return args


class base_model_d(nn.Module):

    def __init__(self, parser):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True if not self.device == "cpu" else False
        parser.add_args(arguments())

        self.args = parser.get_args()

        self.model = self.args.model
        self.dataset = self.args.data
        self.nc = self.args.nc
        self.metric = getattr(self.args, "metric", [])

        self.epochs = self.args.epochs
        self.batchsize = self.args.batch_size
        self.lr = self.args.lr
        self.train_with_test = self.args.test_interval
        self.time_avg = self.args.time_avg

        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.indicator_for_best = None
        self.vis = visualizer(parser)

        ########changed
        # self.vis = Visualizer(parser)
        if self.args.control_monitor:  ###开关控制是否监控
            self.viz = visdom.Visdom(env=self.args.cls_type, port=self.args.visdom_port)  ####visdom监控
            if not self.viz.check_connection():
                warnings.warn("visdom服务器尚未启动,请打开visdom")  # 测试一下链接，链接错误的话会警告
        if self.args.control_save_img_type is not None and "metricepoch" in self.args.control_save_img_type:
            self.metricdic = dict()

        self.oufdir = os.path.join(self.args.outf, os.path.basename(os.getcwd()), self.dataset, self.model)
        self.netinfo = Netinfo(self.model, save_path=self.oufdir)

    def train(self, train_dataloader, test_dataloader):
        raise NotImplementedError

    def test(self, train_dataloader, test_dataloader):
        raise NotImplementedError

    def inference(self, dataloader):
        raise NotImplementedError

    def weights_init(self, mod):
        if isinstance(mod, nn.Conv2d):
            if self.args.conv_init_type == "uniform":
                nn.init.uniform_(mod.weight.data, a=0.0, b=1.0)
            elif self.args.conv_init_type == "normal":
                nn.init.normal_(mod.weight.data, mean=0.0, std=1.0)
            elif self.args.conv_init_type == "constant":
                nn.init.constant_(mod.weight.data, val=1)
            elif self.args.conv_init_type == "ones":
                nn.init.ones_(mod.weight.data)
            elif self.args.conv_init_type == "zeros":
                nn.init.zeros_(mod.weight.data)
            elif self.args.conv_init_type == "eye":
                nn.init.eye_(mod.weight.data)
            elif self.args.conv_init_type == "dirac":
                nn.init.dirac_(mod.weight.data, groups=1)
            elif self.args.conv_init_type == "xavier_uniform":
                nn.init.xavier_uniform_(mod.weight.data, gain=1.0)
            elif self.args.conv_init_type == "xavier_normal":
                nn.init.xavier_normal_(mod.weight.data, gain=1.0)
            elif self.args.conv_init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(mod.weight.data, a=0, mode="fan_in", nonlinearity="leaky_relu")
            elif self.args.conv_init_type == "kaiming_normal":
                nn.init.kaiming_normal_(mod.weight.data, a=0, mode="fan_in", nonlinearity="leaky_relu")
            elif self.args.conv_init_type == "orthogonal":
                nn.init.orthogonal_(mod.weight.data, gain=1)
            elif self.args.conv_init_type == "sparse":
                nn.init.sparse_(mod.weight.data, sparsity=0.1, std=0.01)
        elif isinstance(mod, nn.BatchNorm2d):
            if self.args.bn_init_type == "uniform":
                nn.init.uniform_(mod.weight, a=0.0, b=1.0)
                nn.init.uniform_(mod.bias, a=0.0, b=1.0)
            elif self.args.bn_init_type == "normal":
                nn.init.normal_(mod.weight, mean=0.0, std=1.0)
                nn.init.normal_(mod.bias, mean=0.0, std=1.0)
            elif self.args.bn_init_type == "constant":
                nn.init.constant_(mod.weight, val=1)
                nn.init.constant_(mod.bias, val=0)
            elif self.args.bn_init_type == "ones":
                nn.init.ones_(mod.weight)
                nn.init.ones_(mod.bias)
            elif self.args.bn_init_type == "zeros":
                nn.init.zeros_(mod.weight)
                nn.init.zeros_(mod.bias)
            elif self.args.bn_init_type == "eye":
                nn.init.eye_(mod.weight)
                nn.init.eye_(mod.bias)
            elif self.args.bn_init_type == "dirac":
                nn.init.dirac_(mod.weight, groups=1)
                nn.init.dirac_(mod.bias, groups=1)
            elif self.args.bn_init_type == "xavier_uniform":
                nn.init.xavier_uniform_(mod.weight, gain=1.0)
                nn.init.xavier_uniform_(mod.bias, gain=1.0)
            elif self.args.bn_init_type == "xavier_normal":
                nn.init.xavier_normal_(mod.weight, gain=1.0)
                nn.init.xavier_normal_(mod.bias, gain=1.0)
            elif self.args.bn_init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(mod.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")
                nn.init.kaiming_uniform_(mod.bias, a=0, mode="fan_in", nonlinearity="leaky_relu")
            elif self.args.bn_init_type == "kaiming_normal":
                nn.init.kaiming_normal_(mod.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")
                nn.init.kaiming_normal_(mod.bias, a=0, mode="fan_in", nonlinearity="leaky_relu")
            elif self.args.bn_init_type == "orthogonal":
                nn.init.orthogonal_(mod.weight, gain=1)
                nn.init.orthogonal_(mod.bias, gain=1)
            elif self.args.bn_init_type == "sparse":
                nn.init.sparse_(mod.weight, sparsity=0.1, std=0.01)
                nn.init.sparse_(mod.bias, sparsity=0.1, std=0.01)

    def save_weights(self, epoch, stage=0):
        if self.args.control_save_weights:
            if stage == 1:
                self.best_trigger = True
            self.final_epoch_trigger = True if epoch == self.epochs else False
            # if self.best_trigger or self.final_epoch_trigger:
            weight_dir = os.path.join(self.oufdir, "weight")  ########changed
            if not os.path.exists(weight_dir):
                os.makedirs(weight_dir)
            if stage == 0:
                for i in range(len(self.net_dict['name'])):
                    torch.save({'epoch':epoch,'net_params':self.net_dict['network'][i].state_dict()},'{}/{}_epoch{}.pth'.format(weight_dir,self.net_dict['name'][i],epoch))
                    torch.save({'epoch':epoch,'net_params':self.net_dict['network'][i].state_dict()},'{}/{}_current.pth'.format(weight_dir,self.net_dict['name'][i]))
            else:
                torch.save({'epoch':epoch,'net_params':self.net_dict['network'][stage-1].state_dict()},'{}/{}_epoch{}.pth'.format(weight_dir,self.net_dict['name'][stage-1],epoch))
                torch.save({'epoch':epoch,'net_params':self.net_dict['network'][stage-1].state_dict()},'{}/{}_current.pth'.format(weight_dir,self.net_dict['name'][stage-1]))

            if self.best_trigger:
                if stage == 0:
                    for i in range(len(self.net_dict['name'])):
                        cmd = 'cp {}/{}_current.pth {}/{}_best.pth'.format(weight_dir,self.net_dict['name'][i],weight_dir,self.net_dict['name'][i])
                        os.system(cmd) # 直接调用系统功能
                else:
                    cmd = 'cp {}/{}_current.pth {}/{}_best.pth'.format(weight_dir,self.net_dict['name'][stage-1],weight_dir,self.net_dict['name'][stage-1])
                    os.system(cmd)

            if self.final_epoch_trigger:
                if stage == 0:
                    for i in range(len(self.net_dict['name'])):
                        cmd = 'mv {}/{}_current.pth {}/{}_final.pth'.format(weight_dir,self.net_dict['name'][i],weight_dir,self.net_dict['name'][i])
                        os.system(cmd)
                else:
                    cmd = 'mv {}/{}_current.pth {}/{}_final.pth'.format(weight_dir,self.net_dict['name'][stage-1],weight_dir,self.net_dict['name'][stage-1])
                    os.system(cmd)

    def load_weights(self, stage = 0):
        if stage == 0:
            for i in range(len(self.net_dict['name'])):
                if self.args.weight_path == "None":
                    weight_dir = os.path.join(self.oufdir, 'weight')
                    weight_dir = os.path.join(weight_dir,self.net_dict['name'][i]+'_'+self.args.test_weight_choose+'.pth') if self.args.test_weight_choose == 'best' or self.args.test_weight_choose=='final' else \
                    os.path.join(weight_dir,self.net_dict['name'][i]+'_current.pth')
                else :
                    weight_dir = self.args.weight_path
                self.pretrain_dict = torch.load(weight_dir)
                self.net_dict['network'][i].load_state_dict(self.pretrain_dict['net_params'])
        else:
            if self.args.weight_path == "None":
                weight_dir = os.path.join(self.args.outf, os.path.basename(os.getcwd()), self.args.data,self.args.model, 'weight')
                weight_dir = os.path.join(weight_dir, self.net_dict['name'][stage-1]+'_'+self.args.test_weight_choose+'.pth') if self.args.test_weight_choose == 'best' or self.args.test_weight_choose=='final' else \
                os.path.join(weight_dir, self.net_dict['name'][stage-1]+'_current.pth')
                self.pretrain_dict = torch.load(weight_dir)
                self.net_dict['network'][stage-1].load_state_dict(self.pretrain_dict['net_params'])
            else :
                weight_dir = self.args.weight_path
                self.net_dict['network'][stage-1].load_state_dict(torch.load(weight_dir))

    ########changed
    def save_loggin_print(self, current_epoch, best):
        self.final_epoch_trigger = True if current_epoch == self.epochs else False
        if self.args.control_print or self.args.control_save:
            self.vis.loggin(current_epoch, best, self.indicator_for_best)
        if self.args.control_save:
            self.vis.save()
        if self.args.control_print:
            self.vis.output()
        if self.final_epoch_trigger and self.args.control_save:
            self.vis.plot_menu(best)

    def save_img(self,img_fus,epoch):
        for i in range(len(img_fus)):
            img_fus[i] = (img_fus[i].cpu().numpy()*255.0).astype(np.uint8)
            if img_fus[i].shape[0] == 1:
                self.vis.imsave(img_fus[i].squeeze(0),i,epoch)
            else:
                self.vis.imsave(img_fus[i].transpose(1,2,0),i,epoch)
    def RGB2Y(self, img):
        img_y = 0.299 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.114 * img[:, 2:3, :, :]
        return img_y

    # def Y2RGB(self, img_Y, img_vi):
    #     vi_r, vi_g, vi_b = img_vi[:, 0:1, :, :], img_vi[:, 1:2, :, :], img_vi[:, 2:3, :, :]
    #     img_vi_y = 0.299*vi_r + 0.587*vi_g + 0.114*vi_b
    #     img_vi_cb = (vi_b - img_vi_y)
    #     img_vi_cr = (vi_r - img_vi_y)
    #     R = img_Y + img_vi_cr
    #     G = img_Y - 0.1942*img_vi_cb - 0.5097*img_vi_cr
    #     B = img_Y + img_vi_cb
    #     return torch.clamp(torch.cat([R, G, B], 1), 0, 1)
    def Y2RGB(self, img_Y, img_vi):
        """
        利用融合后的 Y 和原始可见光的 Cb, Cr 重建 RGB
        输入:
            img_Y: (B, 1, H, W) 融合后的亮度通道
            img_vi: (B, 3, H, W) 原始可见光图像 (用于提取色度)
        输出:
            (B, 3, H, W) 重建后的 RGB 图像
        """
        vi_r, vi_g, vi_b = img_vi[:, 0:1, :, :], img_vi[:, 1:2, :, :], img_vi[:, 2:3, :, :]
        vi_y = 0.299 * vi_r + 0.587 * vi_g + 0.114 * vi_b

        vi_cb = (vi_b - vi_y) / 1.772
        vi_cr = (vi_r - vi_y) / 1.402

        R = img_Y + 1.402 * vi_cr
        G = img_Y - 0.344136 * vi_cb - 0.714136 * vi_cr
        B = img_Y + 1.772 * vi_cb

        return torch.clamp(torch.cat([R, G, B], 1), 0, 1)


    def imsave(self, image, i, epoch):#################
        self.file_name = ''
        if self.args.model == 'fusiongan':
            if self.args.no_content:
                self.file_name += '-no_content'
            else:
                self.file_name += '_content'
            if self.args.no_adv:
                self.file_name += '-no_adv'
            else:
                self.file_name += '_adv'
            self.file_name += '-ratio{}'.format(self.args.gen_dis_ratio)
            self.file_name += '_epoch{}'.format(self.args.epochs)
        elif self.args.model == 'pmgi':
            if self.args.ratio2:
                self.file_name += '-ratio2'
            else:
                self.file_name += '-ratio3'
            if self.args.pathwise:
                self.file_name += '-pathwise'
            else:
                self.file_name += '-no_pathwise'
            if self.args.int_only:
                self.file_name += '-int_only'
            elif self.args.grad_only:
                self.file_name += '-grad_only'
            elif self.args.ir_grad_vi_int:
                self.file_name += '-ir_grad_vi_int'
            elif self.args.ir_int_vi_grad:
                self.file_name += '-ir_int_vi_grad'
            elif self.args.ir_vi_normal:
                self.file_name += '-ir_vi_normal'
            self.file_name += '_epoch{}'.format(self.args.epochs)
        elif self.args.model == 'densefuse':
            if self.args.no_ssim:
                self.file_name += '-no_ssim'
            else:
                self.file_name += '-ssim'
            if self.args.no_pixel:
                self.file_name += '-no_pixel'
            else:
                self.file_name += '-pixel'
            if self.args.fusion_type == 'addition_sum':
                self.file_name += '-addition_sum'
            elif self.args.fusion_type == 'addition_mean':
                self.file_name += '-addition_mean'
            elif self.args.fusion_type == 'attention_sum':
                self.file_name += '-attention_sum'
            elif self.args.fusion_type == 'attention_mean':
                self.file_name += '-attention_mean'
            self.file_name += '_epoch{}'.format(self.args.epochs)
        elif self.args.model == 'wang_3':
            if self.args.no_content:
                self.file_name += '-no_content'
            else:
                self.file_name += '_content'
            if self.args.no_adv:
                self.file_name += '-no_adv'
            else:
                self.file_name += '_adv'
            if self.args.single_no_att:
                self.file_name += '-single_no_att'
            else:
                self.file_name += '-double_no_att'
            self.file_name += '_epoch{}'.format(self.args.epochs)
        elif self.args.model == 'wang_4':
            if self.args.no_content:
                self.file_name += '-no_content'
            else:
                self.file_name += '_content'
            if self.args.no_ssim:
                self.file_name += '-no_ssim'
            else:
                self.file_name += '_ssim'
            if self.args.meta:
                self.file_name += '-meta'
            else:
                self.file_name += '-no_meta'
            self.file_name += '_epoch{}'.format(self.args.epochs)
        elif self.args.model == 'xia_3':
            if self.args.no_content:
                self.file_name += '-no_content'
            else:
                self.file_name += '_content'
            if self.args.no_ssim:
                self.file_name += '-no_ssim'
            else:
                self.file_name += '_ssim'
            if self.args.no_adv:
                self.file_name += '-no_adv'
            else:
                self.file_name += '_adv'
            if self.args.single_dis:
                self.file_name += '-single_dis'
            else:
                self.file_name += '-double_dis'
            self.file_name += '_epoch{}'.format(self.args.epochs)
        elif self.args.model == 'xia_4':
            if self.args.no_content:
                self.file_name += '-no_content'
            else:
                self.file_name += '_content'
            if self.args.no_ssim:
                self.file_name += '-no_ssim'
            else:
                self.file_name += '_ssim'
            if self.args.no_content_:
                self.file_name += '-no_content_'
            else:
                self.file_name += '_content_'
            if self.args.path_wise:
                self.file_name += '-path_wise'
            else:
                self.file_name += '-no_path_wise'
            self.file_name += '_epoch{}'.format(self.args.epochs)
        elif self.args.model == 'dual_branch':
            if self.args.no_mse:
                self.file_name += '-no_mse'
            else:
                self.file_name += '_mse'
            if self.args.no_grad:
                self.file_name += '-no_grad'
            else:
                self.file_name += '_grad'
            if self.args.no_std:
                self.file_name += '-no_std'
            else:
                self.file_name += '_std'
            if self.args.no_perceptual:
                self.file_name += '-no_perceptual'
            else:
                self.file_name += '_perceptual'
            if self.args.fusion_type == 'channel':
                self.file_name += '-channel'
            elif self.args.fusion_type == 'addition':
                self.file_name += '-addition'
            else:
                self.file_name += '-l1'
            self.file_name += '_epoch{}'.format(self.args.epochs)
        elif self.args.model == 'ganmcc':
            if self.args.no_adv:
                self.file_name += '-no_adv'
            else:
                self.file_name += '_adv'
            if self.args.no_content:
                self.file_name += '_no_content'
            else:
                self.file_name += '_content'
            self.file_name += '_epoch{}'.format(self.args.epochs)
        elif self.args.model == 'std_fusion':
            if self.args.no_grad:
                self.file_name += '-no_grad'
            else:
                self.file_name += '_grad'

            self.file_name += '_mask{}'.format(self.args.alpha)
            self.file_name += '_epoch{}'.format(self.args.epochs)
        elif self.args.model == 'rfn_nest':
            if self.args.two_stage:
                self.file_name += '_two_stage'
            if self.args.alpha == 0:
                self.file_name += 'alpha_0'
            else:
                self.file_name += 'alpha_7'
            if self.args.no_short_connect:
                self.file_name += '_no_short_connect'
            else:
                self.file_name += '_get_short_connect'
            if self.args.use_strategy:
                self.file_name += '_fusion_type{}'.format(self.args.fusion_type)
            else:
                self.file_name += '_no_fusion_type'
            if self.args.comple_train:
                self.file_name += '_comple_trian'
            else:
                self.file_name += '_nocomple_train'
            if self.args.load_weights:
                self.file_name += '_load_weights'
            else:
                self.file_name += '_no_weights'
            self.file_name += '_epoch{}'.format(self.args.epochs)
        elif self.args.model == 'ren':
            self.file_name += 'ir_int{}_ir_grad{}_vi_int{}_vi_grad{}'.format(self.args.ir_int,self.args.ir_grad,self.args.vi_int,self.args.vi_grad)
            self.file_name += '_lda{}'.format(self.args.lda)
            self.file_name += '_gmma{}'.format(self.args.gmma)

            if self.args.content == 1:
                self.file_name +='_content'
            else:
                self.file_name += '_nocontent'
            if self.args.ms == 1:
                self.file_name +='_ms'
            else:
                self.file_name += '_noms'

            self.file_name += '_epoch{}'.format(self.args.epochs)
            self.file_name += '_times{}'.format(self.args.times)
        elif self.args.model == 'ren_yushiyan':
            self.file_name += '_epoch{}'.format(self.args.epochs)
        elif self.args.model == 'ddcgan':
            self.file_name += '_epoch{}'.format(self.args.epochs)
        elif self.args.model == 'U2Fusion':
            if self.args.act_type == 'prelu':
                self.file_name += '_prelu'.format(self.args.act_type)
            else :
                self.file_name += '_lrelu'.format(self.args.act_type)
            self.file_name += '_epoch{}'.format(self.args.epochs)
        elif self.args.model == 'UNFusion':
            if self.args.fusion_type:
                self.file_name += '_fusion_type{}'.format(self.args.fusion_type)
            self.file_name += '_epoch{}'.format(self.args.epochs)

        save_path = os.path.join(self.args.outf,os.path.basename(os.getcwd()),self.args.data,self.args.model,'imgs',self.file_name,str(epoch))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        imageio.imwrite(os.path.join(save_path,'fus_{}.bmp'.format(i+1)), image)

    ########整个function——changed
    def save_result_img(self, current_epoch, best_trigger, metric_cluster, valuediclist, model_layer_list=None):
        from Vis import Monitor, Embeddingimg, Valueepoch, Attentionmap, Featuremap, Filtermap

        self.final_epoch_trigger = True if current_epoch == self.epochs else False
        if self.args.control_save_img_type is not None:
            #########记录每个epoch的非曲线metric值###########
            def get_metric_log():
                metric_name = list(set(self.metric) - set(metric_cluster.metric_properties.keys()))
                for i in range(len(metric_name)):
                    if current_epoch == 1:
                        self.metricdic[metric_name[i]] = []
                    self.metricdic[metric_name[i]].append(metric_cluster.forward([metric_name[i]])[0].item())

            if current_epoch != "test" and "metricepoch" in self.args.control_save_img_type:
                get_metric_log()
            ############################################
            """
            if best_trigger and "t-SNE" in self.args.control_save_img_type:  # 最优
                eimg = Embeddingimg(embedding_dim=self.args.embedding_dim, tsne_init=self.args.tsne_init, class_dict=self.classes_dict_r, save_path=self.oufdir,loc_legend='lower right')
                eimg.embeddingimg(embedding, real)
            """
            if self.final_epoch_trigger:  # 最后
                if "valueepoch" in self.args.control_save_img_type:
                    vimg = Valueepoch(self.epochs, self.oufdir)
                    vimg.valueepoch(valuediclist)
                if "metricepoch" in self.args.control_save_img_type:
                    metricdiclist = [["metric", self.metricdic]]
                    vimg = Valueepoch(self.epochs, self.oufdir)
                    vimg.valueepoch(metricdiclist)

            if self.final_epoch_trigger or current_epoch == "test":  # 最后或test
                if self.final_epoch_trigger:
                    self.load_weights()
                if "attentionmap" in self.args.control_save_img_type:
                    amap = Attentionmap(self.args.image_size_h, self.args.image_path, image_size2= self.args.image_size_2,save_path=self.oufdir, mean=self.args.mean, std=self.args.std)
                    amap.save_attentionmap(model_layer_list)
                if "featuremap" in self.args.control_save_img_type:
                    fmap = Featuremap(self.args.image_size_h, self.args.image_path, image_size2= self.args.image_size_2,save_path=self.oufdir, mean=self.args.mean, std=self.args.std)
                    fmap.save_featuremap(model_layer_list)
                if "filter" in self.args.control_save_img_type:
                    filter = Filtermap(self.args.image_path, save_path=self.oufdir)
                    filter.save_filter(model_layer_list)

        if self.args.control_monitor and current_epoch != "test":  # 实时,不要更改顺序
            monit = Monitor(self.metric, self.args.control_monitor, self.oufdir)
            monit.monitor(current_epoch, self.viz, valuediclist, metric_cluster)

    def set_input(self, input):
        if isinstance(input[1], tuple):
            import numpy as np

            input[1] = torch.from_numpy(np.array([int(x) for x in np.array(input[1])]))
        return input

    def save_args(self):  ###########保存opt参数
        dir_name = ""
        args_dir = os.path.join(self.args.outf, os.path.basename(os.getcwd()), self.dataset, self.model)
        if not os.path.exists(args_dir):
            os.makedirs(args_dir)
        with open(os.path.join(args_dir, "options.txt"), "w", newline="\n") as file:
            file.seek(0)
            file.truncate()
            for arg, content in self.args.__dict__.items():
                file.write("{}:{},\n".format(arg, content))

    # def compute_macs_params(self, net, net_name, size):
    #     macs, params = get_model_complexity_info(net, (size[0], size[1], size[2]))
    #     self.vis.plot_macs_params(macs, params, net_name)
    #     return macs, params

    def create_empty_loader(self):
        self.imsize2 = self.args.image_size_h if self.args.image_size_2 == -1 else self.args.image_size_2
        sample_tensor = torch.zeros(self.batchsize * self.time_avg, self.nc, self.args.image_size_h, self.imsize2)
        label_tensor = torch.zeros(self.batchsize * self.time_avg)
        empty_dataset = TensorDataset(sample_tensor, label_tensor)
        empty_dataloader = DataLoader(empty_dataset, shuffle=True, batch_size=self.batchsize)
        return empty_dataloader

    def compute_time_fps(self, empty_dataloader):

        if self.inference(empty_dataloader) == None:
            raise Exception("You need define an inference function in your model.")

        else:
            self.starter.record()
            self.inference(empty_dataloader)
            self.ender.record()
            torch.cuda.synchronize()
            self.inference_time = self.starter.elapsed_time(self.ender)
            self.gpu_time = self.inference_time / self.time_avg
            self.fps = 1000 / self.gpu_time

    ###change
    def get_inference_time(self):
        self.compute_time_fps(self.create_empty_loader())
        self.netinfo.plot_inference_time(self.gpu_time)
        return self.gpu_time

    ###change
    def get_fps(self):
        self.compute_time_fps(self.create_empty_loader())
        self.netinfo.plot_fps(self.fps)
        return self.fps