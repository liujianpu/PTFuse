import os
import cv2
import torch
from PIL import Image

def _tensor2cv2(img_tensor):
        # 到cpu,去掉批次维度
        input_tensor = img_tensor.clone().detach().cpu().squeeze()  
        # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
        input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()  
        return input_tensor

class Processed_image(object):
    def __init__(self, image_path, save_path="./result_img/", format=".jpg"):
        super(Processed_image, self).__init__()
        self.image_path = image_path
        self.save_path = save_path
        self.format = format
        
    def save_processed_image(self,transform,is_train=True):
        """
        输入：
            transform为图像增强方法
        输出：保存增强后的图片至对应位置
        """
        try:
            imglist = os.listdir(self.image_path)
        except FileNotFoundError:
            imglist= []
            print("FileNotFoundError: [Errno 2] No such file or directory: " + self.image_path)

        mode = 'train' if is_train else 'test'
        save_process_path = os.path.join(self.save_path, 'result_img','process_result')
        if not os.path.exists(save_process_path):
            os.makedirs(save_process_path)
        
        
        num_images = len(imglist) 
        for i in range(num_images):
            # Load image
            im_path = os.path.join(self.image_path, imglist[i])
            img = Image.open(im_path).convert("RGB")
            filename = os.path.splitext(imglist[i])[0]
            img_tensor = transform(img)
            img_rgb = _tensor2cv2(img_tensor)
            # RGB转BRG
            image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  
            #save
            filename1 = filename + "_" + mode + "_aug"+ self.format
            image_save_path = os.path.join(save_process_path, filename1)
            cv2.imwrite(image_save_path, image)
