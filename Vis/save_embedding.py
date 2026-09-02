import os
from warnings import simplefilter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches

simplefilter(action = 'ignore', category = FutureWarning)

def _normalization(X):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)  # 标准化
    return X


def _splitdata(y_list, X, y):
    X_y = dict()
    for i in y_list:
        X_y[i] = []
        for j in range(X.shape[0]):
            if y[j] == i:
                X_y[i].append(X[j].tolist())
    return X_y


class Embeddingimg(object):
    """docstring for Embeddingimg"""
    def __init__(self, embedding_dim, tsne_init, class_dict, save_path="./", loc_legend='best'):
        super(Embeddingimg, self).__init__()
        self.embedding_dim = embedding_dim
        self.tsne_init = tsne_init
        self.class_dict = class_dict
        self.save_path = os.path.join(save_path, 'result_img/')
        self.loc_legend = loc_legend

    def embeddingimg(self, X, y, pic_name="embeddingimg.jpg"):
        if isinstance(X, list):
            self.doublembedding(X, y, pic_name)
        else:
            self.singlembedding(X, y, pic_name)

    def singlembedding(self, X, y, pic_name="embeddingimg.jpg"):
        """
        输入：
            X为flatten后的logits(模型的输出）
            y为标签；
            save_path,pic_name:为图片保存的路径和名字
        输出：保存图片至对应位置
        """
        #####对数据进行降维和标准化
        tsne = TSNE(n_components=self.embedding_dim,
                    init=self.tsne_init, learning_rate=200, random_state=0)
        X = tsne.fit_transform(X)[:, 0: self.embedding_dim]
        X = _normalization(X)
        ####依据设定选择坐标系
        fig = plt.figure()
        ax = fig.add_subplot(
            111, projection="rectilinear" if self.embedding_dim == 2 else "3d")
        axc = plt.gca()
        y_list = list(set(y.numpy()))
        types = [[]] * len(y_list)
        #####依据y的取值划分数据
        X_y = _splitdata(y_list, X, y)
        ######开始画图
        for i in range(len(X_y)):
            color = next(axc._get_lines.prop_cycler)["color"]
            X_y[i] = np.asarray(X_y[i])
            if self.embedding_dim == 2:
                types[i] = plt.scatter(
                    X_y[i][:, 0], X_y[i][:, 1], s=10, color=color, cmap="plasma")
            if self.embedding_dim == 3:
                types[i] = plt.scatter(
                    X_y[i][:, 0], X_y[i][:, 1], X_y[i][:, 2], color=color, cmap="plasma")

        # plt.title("t-SNE embedding")
        plt.legend(tuple(types), tuple(self.class_dict.values()), loc=self.loc_legend)  # 图例'loc' = "best"/"lower left"/"right"/"center left"/"lower center"/"center"
        plt.xticks([])
        plt.yticks([])
        ax.spines["right"].set_visible(False)  # 去除右边框
        ax.spines["top"].set_visible(False)  # 去除上边框
        ax.spines["left"].set_visible(False)  # 去除右边框
        ax.spines["bottom"].set_visible(False)  # 去除上边框

        # save_path = os.path.join(outf, save_path)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        plt.savefig(os.path.join(self.save_path, pic_name))  # 图像路径需要接口
        plt.close()

    def doublembedding(self, X, y, pic_name="embeddingimg.jpg"):
        tsne = TSNE(n_components=self.embedding_dim,
                    init=self.tsne_init, perplexity=70, learning_rate=500, n_iter=1000, early_exaggeration=10)  #
        self.fig = plt.figure()

        plt.rcParams['figure.dpi'] = 300  # 图片像素
        plt.rcParams['font.family'] = ["Times New Roman"]
        # 依据设定选择坐标系
        text = ['source', 'target']
        self.color_types = {}
        for i in range(2):
            X[i] = tsne.fit_transform(X[i])[:, 0: self.embedding_dim]
            x_min, x_max = np.min(X[i], 0), np.max(X[i], 0)
            X[i] = (X[i] - x_min) / (x_max - x_min)  # 标准化
            self.y_list = list(set(y[i].numpy().astype(int)))
            # self.types[i] = [[]] * len(self.y_list)
            self.color_types[text[i]] = [[]]*len(self.y_list)

        dfs = pd.DataFrame(X[0], columns=['x1', 'x2'])
        dft = pd.DataFrame(X[1], columns=['x1', 'x2'])
        dfs['label'] = y[0].numpy().astype(int)
        dft['label'] = y[1].numpy().astype(int)
        dfs['label_name'] = 'source'
        dft['label_name'] = 'target'
        dfs['color'] = "goldenrod"
        dft['color'] = "steelblue"
        dfs['marker'] = '^'
        dft['marker'] = '*'
        df = pd.concat([dfs, dft], sort=False)

        g1 = sns.JointGrid(x='x1', y='x2', data=df, ratio=6,
                           xlim=(-0.2, 1.1), ylim=(-0.1, 1.1))

        self._draw_plot_df(g1, df)

        g1.ax_joint.set_xticks([])
        g1.ax_joint.set_yticks([])
        g1.ax_joint.axes.set_ylabel('')
        g1.ax_joint.axes.set_xlabel('')

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        plt.savefig(os.path.join(self.save_path, pic_name),
                    bbox_inches='tight', pad_inches=0)
        plt.close()
    
    def _draw_plot_df(self,g, df):
        mark_patches = []
        for name, df_group in df.groupby('label_name'):
            def colored_scatter(x, y):
                sns.scatterplot(data=df_group, x='x1', y='x2', hue="label", palette=sns.color_palette(
                    n_colors=len(self.y_list)), legend=False, edgecolors='k', s=20, linewidth=0, marker=df_group['marker'][0])
            g.plot_joint(colored_scatter)
            sns.kdeplot(
                data=df_group,
                x='x1',
                ax=g.ax_marg_x,
                color=df_group['color'][0],
                shade=True,
            )
            sns.kdeplot(
                data=df_group,
                y='x2',
                ax=g.ax_marg_y,
                color=df_group['color'][0],
                shade=True,
            )
            mark_patches += [plt.plot([], [], marker=df_group['marker'][0],
                                        label="{:s}".format(name), ms=8, ls="", mec=None, color='r')[0]]

        color = sns.color_palette(n_colors=len(self.y_list))
        for q in range(len(self.y_list)):
            self.color_types[q] = mpatches.Patch(color=color[q], label=self.class_dict[self.y_list[q]])

        color_patches = [self.color_types[i] for i in range(len(self.y_list))]

        l1 = plt.legend(handles=color_patches, loc="upper left", fontsize=10)
        plt.legend(handles=mark_patches, loc="lower left", fontsize=10)
        # #重新显示l1图例
        plt.gca().add_artist(l1)
