'''
Author: your name
Date: 2021-05-03 19:11:56
LastEditTime: 2021-09-24 20:23:24
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Image_Fusion/main.py
'''
from libs.opt import options
from libs.data import data
opt = options()
exec('from model.{}.model import {}'.format(opt.get_args().model,opt.get_args().model)) 
exec('model = {}(opt)'.format(opt.get_args().model))

data = data(opt)
train_loader, test_loader = data.get_data() 
exec('model.{}(train_loader,test_loader)'.format(opt.get_args().phase))

