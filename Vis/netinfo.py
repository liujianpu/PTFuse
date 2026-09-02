from . import setup_logger
from ptflops import get_model_complexity_info

class Netinfo(object):
    def __init__(self, method_name, save_path = None, control_print =True):
        super(Netinfo).__init__()
        self.method_name = method_name
        self.control_print = control_print
        self.save_path = save_path


    def plot_inference_time(self, time):
        """
        输入：
            time为gpu计算时间
        输出：
            打印出推理时间
        """
        logger =  setup_logger('inference-time', save_dir=self.save_path, control_print=self.control_print, filename="inference_time.txt")
        logger.info('''{}'s inference_time: {} ms'''.format(self.method_name, time))
     
    def plot_fps(self, fps):
        """
        输入：
            fps为每秒处理多少张图片
        输出：
            打印出模型的fps
        """
        logger =  setup_logger('fps', save_dir=self.save_path, control_print=self.control_print, filename="fps.txt")
        logger.info('''{}'s fps: {} img/s'''.format(self.method_name, fps))

    def plot_macs_params(self, macs, params, net_name):
        """
        输入：
            macs为模型计算量
            params为模型参数量
        输出：
            打印出模型的macs和params
        """
        logger =  setup_logger('macs_params', save_dir=self.save_path, control_print=self.control_print, filename="macs_params.txt")
        logger.info('''{} : macs: {}, params: {}.'''.format(net_name, macs, params))

    def compute_macs_params(self, net, net_name, size):
        size[2] = size[1] if size[2]==-1 else size[2]
        macs, params = get_model_complexity_info(
            net, (size[0], size[1], size[2]))
        self.plot_macs_params(macs, params, net_name)
        return macs, params
