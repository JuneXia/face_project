from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import time
from tensorflow.core.protobuf import config_pb2


class ResnetTrainer(BaseTrain):
    def __init__(self, sess, model, config, logger, train_data, val_data=None, eval_data=None):
        super(ResnetTrainer, self).__init__(sess, model, config, logger,
                                            train_data=train_data, val_data=val_data, val_freq=config.val_freq,
                                            eval_data=eval_data, eval_freq=config.eval_freq)

    def train_epoch(self, epoch_size=1000):
        loop = tqdm(range(epoch_size))
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.train_step()
        #     losses.append(loss)
        #     accs.append(acc)
        # loss = np.mean(losses)
        # acc = np.mean(accs)
        #
        # cur_it = self.model.global_step_tensor.eval(self.sess)
        # summaries_dict = {
        #     'loss': loss,
        #     'acc': acc,
        # }
        # self.logger.summarize(cur_it, summaries_dict=summaries_dict)

        self.model.save(self.sess)

    def train_step(self):
        step = self.model.global_step_tensor.eval(self.sess)

        batch_x, batch_y = self.train_data.next_batch()
        feed_dict = {self.model.images: batch_x, self.model.labels: batch_y,
                     self.model.phase_train: True, self.model.keep_rate: 1. - self.config.dropout_rate}

        feed_dict.update(self.model.net.all_drop)
        start = time.time()

        # tensor_list = [self.model.train_op, self.model.total_loss, self.model.cross_entropy, self.model.wd_loss, inc_op, self.model.acc]
        tensor_list = [self.model.train_op, self.model.total_loss, self.model.cross_entropy, self.model.wd_loss, self.model.acc, self.model.net.outputs]
        if step % self.config.summary_interval == 0:
            tensor_list += [self.logger.summary_op]
            _, total_loss, cross_entropy, wd_loss, acc, embs, summ = self.sess.run(tensor_list, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
            self.logger.add_summary(summ, step)
        else:
            _, total_loss, cross_entropy, wd_loss, acc, embs = self.sess.run(tensor_list, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

        end = time.time()
        pre_sec = self.config.batch_size / (end - start)

        print(embs.min(), embs.max(), embs.mean(), embs.var(), embs.std(), embs.sum())

        print('TotalLoss %.2f , CEloss %.2f, WDloss %.2f, TrainAcc %.6f, time %.3f samples/sec' %
              (total_loss, cross_entropy, wd_loss, acc, pre_sec))

        return 0, 0

    def validation(self):
        batch_x, batch_y = self.val_data.next_batch()
        feed_dict = {self.model.images: batch_x, self.model.labels: batch_y,
                     self.model.phase_train: False, self.model.keep_rate: 1.}

        embs = self.sess.run(self.model.net.outputs, feed_dict)
        print(embs.shape)
        print(embs.min(), embs.max(), embs.mean(), embs.var(), embs.std(), embs.sum())

    def evaluation(self):
        print('evaluation')