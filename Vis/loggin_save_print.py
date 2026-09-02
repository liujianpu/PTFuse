from builtins import print
import numpy as np
from . import setup_logger

class Loggin_save_print(object):
    def __init__(self, method_name, metric, epochs, save_path=None, control_print=True, use_formatter=True):
        super(Loggin_save_print).__init__()
        """
        method_name(str): Your method name.
        metric(list): list of metric, e.g.,['ACC'].
        epochs(int): your total epochs.
        """
        self.method_name = method_name
        self.metric = metric
        self.epochs = epochs
        self.save_path = save_path
        self.use_formatter = use_formatter
        self.control_print = control_print 
        self.save_mode = {'matrix': 'txt',
                          'curve': 'csv'}
        self.logger = {}

    def loggin(self, metric_cluster, current_epoch, best_value, indicator_for_best,valuelist=None):
        ### change
        for i in range(len(self.metric)):
            self.message = """Epoch : {}, {}'s """.format(current_epoch, self.method_name)
            filename = (self.metric[i] if i == indicator_for_best else '_' + self.metric[i]) +'_epoch{}.txt'.format(self.epochs)
            self.logger['metric'] = setup_logger('metric', save_dir=self.save_path, control_print=self.control_print,filename = filename,use_formatter =self.use_formatter)
            self.message += """{} is {:.4f} """.format(self.metric[i], metric_cluster.values[i])  # .item()
            self.message += """ the best {} is {:.4f} """.format(self.metric[i], best_value) if i == indicator_for_best else ""  # .item()
            self.logger['metric'].info(self.message)

        if valuelist != None:
            for i in range(len(valuelist)):
                value_type = valuelist[i][0]
                log = valuelist[i][1]
                filename = value_type +'.txt'
                self.logger[value_type] = setup_logger(value_type, save_dir=self.save_path, filename = filename, control_print=False)
                self.logger[value_type].info(log)
        
        # if current_epoch == self.epochs:
        #     self.logger['best'] = setup_logger(
        #         'best', save_dir=self.save_path, filename='best_menu.txt', control_print=True)
        #     self.logger['best'].info(
        #         '''[{},{:.4f}]'''.format(self.method_name, best_value))


# logger.info(
#         "Total training time: {} ({:.4f} s / it)".format(
#             total_time_str, total_training_time / (max_iter)
#         )
#     )
