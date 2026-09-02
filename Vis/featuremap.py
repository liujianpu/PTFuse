import os
import cv2
import torch
import warnings
import numpy as np
from PIL import Image
import torchextractor as tx
from pytorch_grad_cam.utils.image import preprocess_image

class Featuremap(object):

    def __init__(self,image_size, image_path,image_size2=-1, save_path="./", mean=None, std=None, module_filter_fn=None, capture_fn=None,format= '.png'):
        super().__init__()
        """
        Capture the intermediate feature maps of of model.
        Parameters
        ----------
        model: nn.Module,
            The model to extract features from.
        layerlist: list of str, default None
            The fully qualified names of the modules producing the relevant feature maps.
        module_filter_fn: callable, default None
            A filtering function. Takes a module and module name as input and returns True for modules
            producing the relevant features. Either `module_names` or `module_filter_fn` should be
            provided but not both at the same time.
            Example::
                def module_filter_fn(module, name):
                    return isinstance(module, torch.nn.Conv2d)
            # Hook everything !
            module_filter_fn = lambda module, name: True
            # Capture of all modules inside first layer
            module_filter_fn = lambda module, name: name.startswith("layer1")
            # Focus on all convolutions
            module_filter_fn = lambda module, name: isinstance(module, torch.nn.Conv2d)
        capture_fn: callable, default None
            Operation to carry at each forward pass. The function should comply to the following interface.
            Example::
                def capture_fn(
                        module: nn.Module,
                        input: Any,
                        output: Any,
                        module_name:str,
                        feature_maps: Dict[str, Any]
                    ):
                    feature_maps[module_name] = output
        """
        self.image_size = image_size
        self.image_size2 = image_size2 if image_size2!=-1 else image_size
        self.save_path = save_path
        self.image_path = image_path
        self.module_filter_fn = module_filter_fn
        self.capture_fn = capture_fn
        self.mean = mean
        self.std = std
        self.format = format
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def save_featuremap(self, model_layer_list):
        # get module info
        # print(tx.list_module_names(model))
        imglist = os.listdir(self.image_path)
        num_peer = len(model_layer_list)
        num_images = len(imglist)
        for i in range(num_peer):
            model = model_layer_list[i]["model"]
            layerlist = model_layer_list[i]["layerlist"]
            if not (layerlist is not None and set(layerlist).issubset(set(tx.list_module_names(model)))):
                warnings.warn(
                    "You should either specify the fully qualifying names")
            model = tx.Extractor(
                model, [layer for layer in layerlist], self.module_filter_fn, self.capture_fn)
            for j in range(num_images):
                im_file = os.path.join(self.image_path, imglist[j])
                filename = os.path.splitext(imglist[j])[0]
                img = Image.open(im_file).convert("RGB")
                img = img.resize((self.image_size, self.image_size2), Image.ANTIALIAS)
                img_tensor = preprocess_image(
                    img, mean=self.mean, std=self.std).to(self.device)  # torch.Size([1, 3, 500, 500])
                _, features = model(img_tensor)
                therd_size = 256
                for name, f in features.items():
                    dst_path = os.path.join(
                        self.save_path, "result_img" , "feature_result", filename, name)
                    if not os.path.exists(dst_path):
                            os.makedirs(dst_path)

                    features = f[0]
                    iter_range = features.shape[0]
                    for q in range(iter_range):
                        if "fc" in name:
                            continue
                        feature = features.data.cpu().numpy()
                        feature_img = feature[q, :, :]
                        feature_img = np.asarray(feature_img * 255, dtype=np.uint8)
                        feature_img = cv2.applyColorMap(
                            feature_img, cv2.COLORMAP_JET)
                        if feature_img.shape[0] < therd_size:
                            tmp_file = os.path.join(dst_path, str(
                                q) + "_" + str(therd_size) + self.format)
                            tmp_img = feature_img.copy()
                            tmp_img = cv2.resize(
                                tmp_img, (therd_size, therd_size), interpolation=cv2.INTER_NEAREST)
                            cv2.imwrite(tmp_file, tmp_img)

                        dst_file = os.path.join(dst_path, str(q) + self.format)
                        cv2.imwrite(dst_file, feature_img)
