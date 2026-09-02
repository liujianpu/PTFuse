import os
import matplotlib.pyplot as plt



def _createandsaveimg(cls_epochs, valuedictlist, save_path):
    epochs = cls_epochs
    steps_measure = "epochs"
    for valuedict in valuedictlist:
        measure = valuedict[0]
        plt.figure()
        steps = range(1, epochs + 1)
        ax = plt.gca()
        num = len(valuedict[1])
        for i in range(num):
            color = next(ax._get_lines.prop_cycler)["color"]
            print(valuedict[1].values())
            plt.plot(steps, list(valuedict[1].values())[i], linewidth=1.5, color=color, linestyle="-", label=list(valuedict[1].keys())[i])
        if epochs <= 5:
            plt.xticks(steps)  ####多轮次过于密集
        plt.title("{}-epoch curves".format(valuedict[0]))
        plt.xlabel(steps_measure)
        plt.ylabel(measure)
        plt.legend(loc="best", numpoints=1, fancybox=True)

        imgname="{}-epochimg.jpg".format(valuedict[0])        # save_path = os.path.join(outf, save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, imgname))
        plt.close()



class Valueepoch(object):
    """docstring for Valueepoch"""
    def __init__(self, cls_epochs, save_path="./"):
        super(Valueepoch, self).__init__()
        self.cls_epochs = cls_epochs
        self.save_path = os.path.join(save_path,'result_img/')

    def valueepoch(self, valuedictlist):
        """
        输入：
            cls_epochs
            valuedictlist为输入变量的名称与每个轮次的对应值的字典列表,
            save_path,pic_name:为图片保存的路径和名字
        输出：保存图片至对应位置
        """
        _createandsaveimg(self.cls_epochs, valuedictlist, self.save_path)


