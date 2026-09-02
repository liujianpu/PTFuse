import os
from PIL import Image
from torch.utils.data import Dataset
from libs import my_transform
from torch.utils.data import DataLoader as dataloader
import random
import pdb
def arguments():
    args =  {'--input_size':96,
    '--stride':200 }
    return args
def arguments_randomcrop():
    args = {"--random_crop_w": 96, "--random_crop_h": 96}
    return args
def arguments_centercrop():
    args = {"--center_crop": 16}
    return args
def RandomHorizontalFlip():
    args = {"--RHF_p": 0.5}
    return args
def RandomVerticalFlip():
    args = {"--RVF_p": 0.5}
    return args
def arguments_resize():
    args = {"--resize": 128}
    return args

class data():
    """docstring for data"""
    def __init__(self, parser):
        super(data, self).__init__()
        parser.add_args(arguments())
        self.args = parser.get_args()
        self.expension = self.args.data_expension
        self.input_size = self.args.input_size
        self.stride = self.args.stride
        self.resize16 = self.args.resize16
        self.resize32 = self.args.resize32
        self.convert_mode = self.args.convert_mode
        self.path = os.path.join(self.args.data_root,self.args.data)
        self.prompt = self.args.prompt   #是否有提示信息

        exec("""{}""".format(self.transform_all(parser, "train")))
        exec("""{}""".format(self.transform_all(parser, "test")))

        self.train_set = my_dataset_ir_vi(os.path.join(self.path,'train'),self.transform_train,self.convert_mode,self.expension,self.input_size,self.stride,self.prompt)
        self.test_set = my_dataset_ir_vi(os.path.join(self.path,'test'),transform = self.transform_test,convert_mode = self.convert_mode,prompt = self.prompt)

    def transform_all(self, parser, mode_str):
        methods = {"random_crop": "arguments_randomcrop", "center_crop": "arguments_centercrop", "color": "arguments_colorjitter", "RHF":"RandomHorizontalFlip", "RVF":"RandomVerticalFlip", "resize":"arguments_resize"}
        transform_statement = """self.transform_""" + mode_str + """= my_transform.Compose(["""
        if mode_str == "train":
            for tran in self.args.aug_methods:
                exec("parser.add_args({}())".format(methods[tran]))
                self.args = parser.get_args()
            for tran in self.args.aug_methods:
                if tran == "random_crop":
                    transform_statement += """my_transform.RandomCrop({}),""".format(self.args.random_crop_w)
                elif tran == "center_crop":
                    transform_statement += """my_transform.CenterCrop({}),""".format(self.args.center_crop)
                elif tran == "RHF":
                    transform_statement += """my_transform.RandomHorizontalFlip({}),""".format(self.args.RHF_p)
                elif tran == "RVF":
                    transform_statement += """my_transform.RandomVerticalFlip({}),""".format(self.args.RVF_p)
                elif tran == "resize":
                    transform_statement += """my_transform.Resize({}),""".format(self.args.resize)
        if mode_str == "test":    
            if self.resize16 is not False:
                transform_statement += """my_transform.Resize_16(),"""
            if self.resize32 is not False:
                transform_statement += """my_transform.Resize_32(),"""
        transform_statement += """my_transform.ToTensor()])"""
        return transform_statement
    

    def get_data(self):
        self.train_set = dataloader(self.train_set, batch_size = self.args.batch_size, shuffle = True, pin_memory=True,drop_last=True)
        self.test_set = dataloader(self.test_set, batch_size = self.args.testsize, shuffle = False, pin_memory=True)
        return self.train_set,self.test_set

class my_dataset_ir_vi(Dataset):
    """docstring for my_dataset"""
    def __init__(self,root,transform = None,convert_mode = None,data_expension = False,input_size = None, stride = None, prompt = False):
        self.root = root
        self.transform = transform
        self.convert_mode = convert_mode
        self.data_expension = data_expension
        self.input_size = input_size
        self.stride = stride
        self.prompt = prompt
        self.vi_folder = os.path.join(self.root, 'vi')
        self.vi_libs = sorted(os.listdir(self.vi_folder))  # 排序文件名
        self.ir_folder = os.path.join(self.root, 'ir')
        self.ir_libs = sorted(os.listdir(self.ir_folder))  # 排序文件名

        if self.prompt is not None:
            self.file_path = os.path.join(self.root, f"{self.prompt}.txt")
            with open(self.file_path, "r", encoding="utf-8") as file: 
                self.text = file.read().splitlines()  # 读取所有行，并以列表形式返回
        # 存在数据集尺寸不一致问题，导致需要逐个计算窗口数量
        if self.data_expension:
            self.window_counts = []
            for k in range(len(self.ir_libs)):
                img_ir = Image.open(os.path.join(self.ir_folder, self.ir_libs[k]))
                w, h = img_ir.size
                num_w = max((w - self.input_size) // self.stride + 1, 0)
                num_h = max((h - self.input_size) // self.stride + 1, 0)
                total = num_w * num_h
                self.window_counts.append(total)  

    def data_patched(self,current_idx):
        # 定位到对应图像和窗口索引
        idx = 0
        while current_idx >= self.window_counts[idx]:
            current_idx -= self.window_counts[idx]
            idx += 1
        img_vi = Image.open(os.path.join(self.vi_folder, self.vi_libs[idx])).convert(str(self.convert_mode))
        img_ir = Image.open(os.path.join(self.ir_folder, self.ir_libs[idx])).convert(str(self.convert_mode))
        w, h = img_vi.size
        num_w = (w - self.input_size) // self.stride + 1
        # 计算窗口坐标
        row = current_idx // num_w
        col = current_idx % num_w
        left = col * self.stride
        top = row * self.stride
        # 裁剪窗口
        img_vi = img_vi.crop((left, top, left+self.input_size, top+self.input_size))
        img_ir = img_ir.crop((left, top, left+self.input_size, top+self.input_size))
        return img_ir,img_vi

    def __getitem__(self, idx):
        # 读取数据
        if self.data_expension:
            img_ir,img_vi = self.data_patched(idx)
        else:
            img_vi = Image.open(os.path.join(self.vi_folder, self.vi_libs[idx])).convert(str(self.convert_mode))
            img_ir = Image.open(os.path.join(self.ir_folder, self.ir_libs[idx])).convert(str(self.convert_mode))
        img_ir,img_vi = self.transform(img_ir,img_vi)  # 图像增强或resize为了测试模型能跑,所有人都需要totensor，所以不用if
        # 返回有无提示的数据
        if self.prompt is not None:
            # text = random.choice(self.text)  #如果有多行文本信息，则随机选择一行，如果只有一行就选自己
            text = self.text
            return img_ir, img_vi, text    
        else:
            return img_ir,img_vi
    def __len__(self):
        return sum(self.window_counts) if self.data_expension else len(self.vi_libs)

