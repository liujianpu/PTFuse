'''
Author: your name
Date: 2021-05-03 19:11:55
LastEditTime: 2022-01-05 18:34:55
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Image_Fusion/libs/visualizer.py
'''
import os
import imageio
import pdb
class visualizer(object):
    """docstring for visualizer"""
    def __init__(self, parser):
        super(visualizer, self).__init__()
        self.args = parser.get_args()
        self.outf_ = os.path.join(self.args.outf, os.path.basename(os.getcwd()), self.args.data)
        dir_name = ''
        if self.args.control_save:
            self.outf = os.path.join(self.outf_, self.args.model)
        if not os.path.exists(self.outf):
            os.makedirs(self.outf)

    def plot_txt(self,file_name, message):
        txt_menu = os.path.join(self.outf, file_name)
        with open(txt_menu, 'a', newline = '\n') as file:
            file.write('%s \n' % message)

    def plot_menu(self, best):
        menu_path = self.outf_ + '/menu.txt'
        menu = os.path.join(menu_path)
        with open(menu, 'a', newline =  '\n') as file:
            datas = [self.args.model, best]
            file.write('%s\n' % datas)

    def loggin(self, metric_cluster, current_epoch, best_value, indicator_for_best):
        self.file_name = ''
        self.message = '''Epoch {}, {}'s '''.format(current_epoch,self.args.model)
        for i in range(len(self.args.metric)):
            self.message += '''{} is {:.4f} '''.format(self.args.metric[i], metric_cluster[i])
            self.message += '''the best {} is {:.4f} '''.format(self.args.metric[i], best_value) if i == indicator_for_best else ''
            self.file_name += self.args.metric[i] if i == indicator_for_best else '_'+ self.args.metric[i]
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
            self.file_name += '_epoch{}.txt'.format(self.args.epochs)
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
            self.file_name += '_epoch{}.txt'.format(self.args.epochs)
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
            self.file_name += '_epoch{}.txt'.format(self.args.epochs)
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
            self.file_name += '_epoch{}.txt'.format(self.args.epochs)
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
            self.file_name += '_epoch{}.txt'.format(self.args.epochs)
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
            self.file_name += '_epoch{}.txt'.format(self.args.epochs)
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
            self.file_name += '_epoch{}.txt'.format(self.args.epochs)
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
            self.file_name += '_epoch{}.txt'.format(self.args.epochs)     
        elif self.args.model == 'ganmcc':
            if self.args.no_adv:
                self.file_name += '-no_adv'
            else:
                self.file_name += '_adv'
            if self.args.no_content:
                self.file_name += '_no_content'
            else:
                self.file_name += '_content' 
            self.file_name += '_epoch{}.txt'.format(self.args.epochs)
        elif self.args.model == 'std_fusion':
            if self.args.no_grad:
                self.file_name += '-no_grad'
            else:
                self.file_name += '_grad'

            self.file_name += '_mask{}'.format(self.args.alpha) 
            self.file_name += '_epoch{}.txt'.format(self.args.epochs)
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
            self.file_name += '_epoch{}.txt'.format(self.args.epochs)
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
            self.file_name += '_times{}'.format(self.args.times)
            self.file_name += '_epoch{}.txt'.format(self.args.epochs)
        elif self.args.model == 'ren_yushiyan':
            self.file_name += '_epoch{}.txt'.format(self.args.epochs)
        elif self.args.model == 'ddcgan':
            self.file_name += '_epoch{}.txt'.format(self.args.epochs)
    def save(self):
        self.plot_txt(self.file_name, self.message)

    def imsave(self, image,i,epoch):#################
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
        save_path = os.path.join(self.args.outf,os.path.basename(os.getcwd()),self.args.data,self.args.model,'imgs',self.file_name,str(epoch))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        imageio.imwrite(os.path.join(save_path,'fus_{}.png'.format(i+1)), image)

    def output(self):
        print(self.message)
        pass

    def plot_inference_time(self, time):
        message = '''{}'s inference_time: {} ms'''.format(self.args.model, time)
        if self.args.control_print:
            print(message)
        if self.args.control_save:
            self.file_name = 'inference_time.txt'
            self.plot_txt(self.file_name, message)

    def plot_fps(self, fps):
        message = '''{}'s fps:{} img/s'''.format(self.args.model, fps)
        if self.args.control_print:
            print(message)
        if self.args.control_save:
            self.file_name = 'fps.txt'
            self.plot_txt(self.file_name, message)

    def plot_macs_params(self, macs, params, net_name):
        message = '''{} :macs:{}, params:{}.'''.format(net_name, macs, params)
        if self.args.control_print:
            print(message)
        if self.args.control_save:
            self.file_name = 'macs_params.txt'
            self.plot_txt(self.file_name, message)
