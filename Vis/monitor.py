import os
import torch
import warnings
import matplotlib.pyplot as plt


class Monitor(object):
    """docstring for Monitor"""
    def __init__(self, metric, control_monitor, save_path=None,tSNEdir="result_img/embeddingimg.jpg"):
        super(Monitor, self).__init__()
        self.metric = metric
        self.control_monitor=control_monitor
        if save_path != None:
            self.tSNE_pth = os.path.join(save_path,tSNEdir)
        else:
            self.tSNE_pth = None


    def monitor(self, epoch, viz, valuediclist, metric_cluster):  ####在visdom实时监控loss和metric
        """
        visdom 使用
        1.在终端对应环境导入库 pip install visdom
        2.在终端打开visdom服务  python -m visdom.server -p [端口号] （等待几分钟）
        3.在浏览器导航栏输入  http://localhost:8097（部署端口号默认）或http://localhost:端口号
        4.另打开一个终端 运行文件 python main.py

        epoch：当前epoch值
        valuedic：要监控的loss的名称及值,eg.{'loss1':[1,2,3,...],'loss2':[1,2,3,...]}
        metricdic：要监控的metric的名称及值,eg.{'metric1':[1,2,3,...],'metric2':[1,2,3,...]}
        """
        metric_name = list(set(self.metric) - set(metric_cluster.metric_properties.keys()))
        if self.control_monitor:  ###开关控制是否监控
            if epoch == 1:  ####只在第一个epoch的时候新建窗口,保证每次重新运行窗口刷新
                for valuedic in valuediclist:
                    viz.line(X=torch.zeros((1,)).cpu(), Y=torch.zeros((1, len(valuedic[1]))).cpu(), win="win_{}".format(valuedic[0]), opts=dict(xlabel="epoch", ylabel="{}".format(valuedic[0]), title="{}".format(valuedic[0]), legend=list(valuedic[1].keys())))  ##,legend=list(valuedic.keys())
                viz.line(X=torch.zeros((1,)).cpu(), Y=torch.zeros((1, len(metric_name))).cpu(), win="win_metric", opts=dict(xlabel="epoch", ylabel="metric", title="metric", legend=metric_name))  ##,legend=self.metric
            for valuedic in valuediclist:
                if not viz.win_exists("win_{}".format(valuedic[0])):
                    warnings.warn("Created window marked as not existing(win_{})!".format(valuedic[0]))
            for valuedic in valuediclist:
                for i in range(len(valuedic[1])):  ######按序追加数据持续监控loss
                    viz.line(X=[epoch], Y=[list(valuedic[1].values())[i][-1]], name=list(valuedic[1].keys())[i], win="win_{}".format(valuedic[0]), update="append")
            if not viz.win_exists("win_metric"):
                warnings.warn("Created window marked as not existing(win_metric)!")
            for i in range(len(metric_name)):
                viz.line(X=[epoch], Y=[metric_cluster.get_metric([metric_name[i]])[0].item()], name=metric_name[i], win="win_metric", update="append")
            if self.tSNE_pth != None:
                try:
                    image = plt.imread(self.tSNE_pth)
                    viz.image(image.transpose(2, 0, 1), win="win_t-SNE")  #######save_embedding函数保存的位置
                except FileNotFoundError:
                    # print("FileNotFoundError: [Errno 2] No such file or directory: " + self.tSNE_pth)
                    pass
