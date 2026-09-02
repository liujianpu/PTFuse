import os
import torch
import warnings
import matplotlib.pyplot as plt
import torchextractor as tx

class Filtermap(object):
    def __init__(self,image_path,save_path="./result_img/", format='.png'):
        super(Filtermap, self).__init__()
        """
        model_layer_list: list of dict, like [{'model':self.netd,'layerlist':["DB3.dbblock1","DB1.dbblock2"]},{...}]
            The fully qualified names of the modules producing the relevant feature maps.
        image_path: images you want to test,must put in a file,for example(./testimg).
        save_path: saving path.
       
        
        Choose the target layer you want to compute the visualization for.
        Usually this will be the last convolutional layer in the model.
        Some common choices can be:
        Resnet18 and 50: model.layer4[-1]
        VGG, densenet161: model.features[-1]
        mnasnet1_0: model.layers[-1]
        You can print the model to help chose the layer
    
        """
        self.save_path = save_path
        self.image_path = image_path
        self.format = format
        self.use_cuda = torch.cuda.is_available()

    def save_filter(self, model_layer_list):
        """You can get relevant details in a dictionary by calling extractor.info()"""
        num_peer = len(model_layer_list)
        for i in range(num_peer):
            model = model_layer_list[i]["model"]
            layerlist = model_layer_list[i]["layerlist"]
            if not (layerlist is not None and set(layerlist).issubset(set(tx.list_module_names(model)))):
                warnings.warn(
                    "You should either specify the fully qualifying names")
            # layer_dict = tx.find_modules_by_names(model,layerlist)
            parm = {}
            num_layer = len(layerlist)
            for i in range(num_layer):
                for name, parameters in model.named_parameters():
                    if layerlist[i] in name and "weight" in name:
                        parm[name] = parameters
            # Visualising the filters
            for name, parameters in parm.items():
                if parameters.shape[-1] == 1 or len(parameters.shape) == 1:
                    continue
                f_min, f_max = parameters.min(), parameters.max()
                parameters = (parameters - f_min) / (f_max - f_min)
                plt.figure(figsize=(35, 35))
                save_filter_path = os.path.join(
                    self.save_path, "result_img", "filter_result", name)
                if not os.path.exists(save_filter_path):
                    os.makedirs(save_filter_path)
                # plot first few filters
                if parameters.shape[0] > 64:
                    for i in range(64):
                        # specify subplot and turn of axis
                        ax = plt.subplot(8, 8, i + 1)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        # plot filter channel in grayscale
                        plt.imshow(parameters[i, 0, :, :].data.cpu(
                        ).numpy(), cmap="gray")  # coolwarm
                        plt.axis("off")
                    plt.savefig(save_filter_path + "/filtermap_%s"% (name) + self.format)
                    plt.close()
                else:
                    for i in range(parameters.shape[0]):
                        ax = plt.subplot(8, 8, i + 1)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        # plot filter channel in grayscale
                        plt.imshow(parameters[i, 0, :, :].data.cpu(
                        ).numpy(), cmap="gray")  # coolwarm
                        plt.axis("off")
                    plt.savefig(save_filter_path + "/filtermap_%s"% (name) + self.format)
                    plt.close()
