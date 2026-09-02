import os
import cv2
import torch
import warnings
import numpy as np
from PIL import Image
import torchextractor as tx
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

class Attentionmap(object):
    def __init__(self, image_size, image_path, image_size2=-1,mean=None, std=None, target_category=None, aug_smooth=True, eigen_smooth=True, save_path="./result_img/", format='.jpg'):
        super(Attentionmap, self).__init__()
        """
        model_layer_list: list of dict, like [{'model':self.netd,'layerlist':["DB3.dbblock1","DB1.dbblock2"]},{...}]
            The fully qualified names of the modules producing the relevant feature maps.
        image_path: images you want to test,must put in a file,for example(./testimg).
        save_path: saving path.
        image_size：defualt(224).
        
        Choose the target layer you want to compute the visualization for.
        Usually this will be the last convolutional layer in the model.
        Some common choices can be:
        Resnet18 and 50: model.layer4[-1]
        VGG, densenet161: model.features[-1]
        mnasnet1_0: model.layers[-1]
        You can print the model to help chose the layer
    
        """
        self.image_size = image_size
        self.image_size2 = image_size2 if image_size2!=-1 else image_size
        self.save_path = save_path
        self.image_path = image_path
        self.mean = mean
        self.std = std
        self.target_category = target_category
        self.aug_smooth = aug_smooth
        self.eigen_smooth = eigen_smooth
        self.format = format
        self.use_cuda = torch.cuda.is_available()
    
    def _save_image(self, img, save_path, filename):
        save_path = os.path.join(save_path, filename)
        cv2.imwrite(save_path, img)

    def save_attentionmap(self, model_layer_list):
        # load img
        imglist = os.listdir(self.image_path)
        num_images = len(imglist)
        num_peer = len(model_layer_list)
        for i in range(num_peer):
            model = model_layer_list[i]["model"]
            layerlist = model_layer_list[i]["layerlist"]
            if not (layerlist is not None and set(layerlist).issubset(set(tx.list_module_names(model)))):
                warnings.warn(
                    "You should either specify the fully qualifying names")
            layer_dict = tx.find_modules_by_names(model, layerlist)
            for name, layer in layer_dict.items():
                layer_dict = layer._modules
                if len(layer_dict) >1:
                    target_layer = layer._modules[next(reversed(layer._modules))]
                else:
                    target_layer = layer

                cam = GradCAM(model=model, target_layer=target_layer, use_cuda=self.use_cuda)
                # init path
                result_path = os.path.join(
                    self.save_path, "result_img", "attention_result", name)
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                for i in range(num_images):
                    # Load image
                    im_file = os.path.join(self.image_path, imglist[i])
                    # input 灰度
                    rgb_img = Image.open(im_file, "r") 
                    rgb_img = rgb_img.resize((self.image_size, self.image_size2), Image.ANTIALIAS) 
                    rgb_img = np.float32(rgb_img) / 255
                    # trans = transforms.ToTensor()
                    # input_tensor = trans(rgb_img).unsqueeze(0)
                    input_tensor = preprocess_image(rgb_img, mean=self.mean, std=self.std) 
                    # Construct the CAM object once, and then re-use it on many images:
                    cam = GradCAM(model=model, target_layer=target_layer, use_cuda=self.use_cuda)
                    # If target_category is None, the highest scoring category will be used for every image in the batch.
                    grayscale_cam = cam(input_tensor=input_tensor,
                                        target_category=self.target_category, 
                                        aug_smooth=self.aug_smooth, 
                                        eigen_smooth=self.eigen_smooth)
                    # In this example grayscale_cam has only one image in the batch:
                    grayscale_cam = grayscale_cam[0, :]
                    filename = os.path.splitext(imglist[i])[0]
                    # save heatmap
                    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    filename2 = filename + "_heatmap" + self.format
                    self._save_image(heatmap, result_path, filename2)
                    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                    # save the original image
                    ori_image = cv2.cvtColor(rgb_img * 255, cv2.COLOR_RGB2BGR)
                    filename1 = filename + "_original" + self.format
                    self._save_image(ori_image, result_path, filename1)
                    # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
                    filename3 = filename + "_attented" + self.format
                    self._save_image(cam_image, result_path, filename3)
