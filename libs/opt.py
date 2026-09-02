'''
Author: your name
Date: 2021-05-03 19:11:55
LastEditTime: 2021-11-18 21:31:48
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Image_Fusion/libs/opt.py
'''
import argparse
class  options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('model', type = str, default = 'baseline')
        self.parser.add_argument('--data_root', type = str, default = '../data')
        self.parser.add_argument(
            '--data',
            type=str,
            default='MSRS',
            choices=['MSRS', 'FMB', 'M3FD'],
            help='Dataset used by the PTFuse paper protocol.',
        )
        self.parser.add_argument('--nc',type = int, default = 1)
        self.parser.add_argument('--control_print',default = True, action = 'store_true',help = 'print the results on terminal(default False)')
        self.parser.add_argument('--control_save',default = True, action = 'store_true',help = 'save the results to files')      
        self.parser.add_argument('--data_expension',default = False, action = 'store_true',help = 'crop the image to expend the number of data')
        self.parser.add_argument('--control_save_weights',default = True, action = 'store_true',help = 'save model.weights or not')
        self.parser.add_argument('--load_weights',default = False, action = 'store_true')
        self.parser.add_argument('--control_save_img',default = True, action = 'store_true',help = 'imsave the result_img on folder')     
        self.parser.add_argument('--phase', type = str, default = 'train',choices = ['train','test'])
        self.parser.add_argument('--outf', type = str, default = '../output')      
        # version 1.1
        self.parser.add_argument("--control_monitor", type=int, default=0, help="use Visdom to monitor training")
        self.parser.add_argument("--control_save_img_type", nargs="+", default=None, choices=['filter', 'valueepoch'], help="save images(default False)")
        self.parser.add_argument("--visdom_port", type=int, default=8097, help="Enter the port number of the Visdom server")
        self.parser.add_argument("--conv_init_type", type = str, default = "xavier_normal", help="Convolutional layer parameter initialization")
        self.parser.add_argument("--bn_init_type", type = str, default = "constant", help="BN layer parameter initialization")
        self.parser.add_argument("--mean", default=[0.485, 0.456, 0.406], help="Normalize images(ImageNet)")
        self.parser.add_argument("--std", default=[0.229, 0.224, 0.225], help="Normalize images(ImageNet)")
        self.parser.add_argument("--image_path", type=str, default="../output/test", help="a file that including test imgs apply in attentionmap/featuremap/processed.")

        # version 1.3
        self.parser.add_argument("--aug_methods", nargs="+", default=[])
        self.parser.add_argument('--convert_mode', type = str, default = 'RGB', choices = ['None','RGB','L','YCbCr'])
        self.parser.add_argument('--testsize', type = int, default = 1)
        self.parser.add_argument(
            '--prompt',
            type=str,
            default=None,
            choices=['None', 'discussion'],
            help='Text prompt file to load. Use None to disable prompts.',
        )
        self.parser.add_argument('--resize16', action='store_true', default=False, help='use the resize16')
        self.parser.add_argument('--resize32', action='store_true', default=False, help='use the resize32')
        
        self.args = self.parser.parse_known_args()[0]
        self.unknown_args = self.parser.parse_known_args()[1]
        if self.args.prompt == 'None':
            self.args.prompt = None
    def parse(self):
        self.args = self.parser.parse_known_args()[0]
        self.unknown_args = self.parser.parse_known_args()[1]	
        if self.args.prompt == 'None':
            self.args.prompt = None
    def add_args(self, arg_pairs):
        if arg_pairs is not None:
            for name, value in zip(arg_pairs.keys(), arg_pairs.values()):
                self.parser.add_argument(name, type = type(value), default = value)
            self.parse()
    def add_args_(self,args):
        if args is not None:
            for item in args:
                self.parser.add_argument(item,action='store_true',help='without')
            self.parse()
    def get_args(self):
        return self.args 
    

    def change_args(self, name, value):
        exec("self.parser.set_defaults({} = {})".format(name, value))
        self.parse()
