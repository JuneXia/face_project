from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import time
import os
from tensorflow.core.protobuf import config_pb2
from utils import tools
import tensorflow as tf
from tensorflow.python.client import timeline


class InceptionResnetTrainer(BaseTrain):
    def __init__(self, sess, model, config, logger, train_data, val_data=None, eval_data=None):
        super(InceptionResnetTrainer, self).__init__(sess, model, config, logger,
                                            train_data=train_data, val_data=val_data, val_freq=config.val_freq,
                                            eval_data=eval_data, eval_freq=config.eval_freq)

        # TODO: 这个实际上应该可以放到基类中去。
        self.logger.set_summary_tensor(self.model.summary_scalar, self.model.summary_histogram)

    def train_epoch(self, cur_epoch, epoch_size=1000):
        # loop = tqdm(range(epoch_size))
        # for _ in loop:
        #     self.train_step()

        self.run_metadata = tf.RunMetadata()
        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

        for step in range(epoch_size):
            if self.config.learning_rate >= 0:
                self.learning_rate = self.config.learning_rate
            else:
                learning_rate_file = os.path.join(self.config.project_path, self.config.learning_rate_schedule_file)
                self.learning_rate = tools.get_learning_rate_from_file(learning_rate_file, cur_epoch)

            start_time = time.time()
            run_tensor_list, result = self.train_step()
            spend_time = time.time() - start_time

            text = ''
            for tensor, value in zip(run_tensor_list, result):
                text += '\t{} {:.4f}'.format(tensor.op.name, value)

            print('Epoch [%d]%d/%d,%s\ttime %.4f' % (cur_epoch, step, epoch_size, text, spend_time))

        # self.model.save(self.sess)

    def train_step(self):
        debug_timeline = True
        global_step = self.model.global_step_tensor.eval(self.sess)

        time1 = time.time()
        batch_x, batch_y = self.train_data.next_batch()
        print(batch_y.shape, '数据读取时间： ', time.time() - time1)

        feed_dict = {self.model.images: batch_x, self.model.labels: batch_y,
                     self.model.phase_train: True, self.model.keep_rate: 1. - self.config.dropout_rate,
                     self.model.learning_rate_holder: self.learning_rate}

        time1 = time.time()
        print_tensor_list = self.model.run_tensor_list + [self.model.learning_rate]
        run_tensor_list = [self.model.train_op] + print_tensor_list
        if False and global_step % self.config.summary_interval == 0:
            run_tensor_list += [self.logger.summary_op]
            result = self.sess.run(run_tensor_list, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
            # self.logger.add_summary(result[-1], global_step)
            result = result[1:-1]
        else:
            # result = self.sess.run(self.model.train_op, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
            result = self.sess.run(self.model.train_op, feed_dict=feed_dict)

            # if debug_timeline and (global_step == 50):
            #     result = self.sess.run(self.model.train_op,
            #                            feed_dict=feed_dict,
            #                            options=self.run_options, run_metadata=self.run_metadata
            #                            )
            #     tl = timeline.Timeline(self.run_metadata.step_stats)
            #     ctf = tl.generate_chrome_trace_format()
            #     with open('tl_step50.json', 'w') as wd:
            #         wd.write(ctf)
            #
            # else:
            #     result = self.sess.run(self.model.train_op, feed_dict=feed_dict)

            # result = result[1:]

        print('计算图时间： ', time.time() - time1)
        result = [0.001] * len(print_tensor_list)
        return print_tensor_list, result

    def validation(self):
        batch_x, batch_y = self.val_data.next_batch()
        feed_dict = {self.model.images: batch_x, self.model.labels: batch_y,
                     self.model.phase_train: False, self.model.keep_rate: 1.}

        run_tensor_list = self.model.run_tensor_list
        result = self.sess.run(run_tensor_list, feed_dict=feed_dict,
                               options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

        text = ''
        for tensor, value in zip(run_tensor_list, result):
            text += '\t{} {:.4f}'.format(tensor.op.name, value)

        print('Validation %s' % (text))

        return result

    def evaluation(self):
        print('evaluation')