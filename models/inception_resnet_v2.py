from base.base_model import BaseModel
import tensorflow as tf
import tensorlayer as tl
from nets.L_Resnet_E_IR_fix_issue9 import get_resnet
import tensorflow.contrib.slim as slim

from models import inception_resnet_v2_slim as network
# from models import resnet_v1 as network
from models import facenet
from losses.face_losses import arcface_loss
import numpy as np


class InceptionResnetV2(BaseModel):
    def __init__(self, n_cls, config):
        super(InceptionResnetV2, self).__init__(config)

        self.n_cls = n_cls
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.images = tf.placeholder(name='img_inputs', shape=[None] + self.config.state_size, dtype=tf.float32)
        self.labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
        self.phase_train = tf.placeholder(name='trainable_bn', dtype=tf.bool)
        self.keep_rate = tf.placeholder(name='keep_rate', dtype=tf.float32)
        self.learning_rate_holder = tf.placeholder(tf.float32, name='learning_rate')

        w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)

        network.g_debug = self.config.debug
        prelogits, _ = network.inference(self.images, self.config.dropout_rate,
                                         phase_train=self.phase_train, bottleneck_layer_size=self.config.embedding_size,
                                         weight_decay=self.config.weight_decay)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        logits = slim.fully_connected(prelogits, self.n_cls, activation_fn=None,
                                      weights_initializer=slim.initializers.xavier_initializer(),
                                      weights_regularizer=slim.l2_regularizer(self.config.weight_decay),
                                      scope='Logits', reuse=False)

        eps = 1e-4
        prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits) + eps, ord=self.config.prelogits_norm_p, axis=1), name='prelogits_norm_loss')
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * self.config.prelogits_norm_loss_factor)

        self.learning_rate = tf.train.exponential_decay(self.learning_rate_holder, self.global_step_tensor,
                                                   self.config.learning_rate_decay_epochs * self.config.epoch_size,
                                                   self.config.learning_rate_decay_factor, staircase=True, name='decay_learning_rate')
        # tf.summary.scalar('learning_rate', self.learning_rate)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy)

        correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(self.labels, tf.int64)), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction, name='accuracy')

        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.total_loss = tf.add_n([cross_entropy] + regularization_losses, name='total_loss')

        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.train_op = facenet.train(self.total_loss, self.global_step_tensor, self.config.optimizer, self.learning_rate, self.config.moving_average_decay, trainable_vars, self.config.log_histograms)

        regularization_losses_sum = tf.reduce_sum(regularization_losses, name='regularization_losses_sum')
        self.run_tensor_list = [self.total_loss, cross_entropy, regularization_losses_sum, accuracy]

        # self.net = get_resnet(self.images, self.config.net_depth, type='ir', w_init=w_init_method, trainable=self.phase_train, keep_rate=self.keep_rate)
        # # 3.2 get arcface loss
        # logit = arcface_loss(embedding=self.net.outputs, labels=self.labels, w_init=w_init_method, out_num=self.config.num_classes)
        #
        # self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=self.labels), name='cross_entropy')
        # # cross_entropy_avg = tf.reduce_mean(cross_entropy)
        # # 3.4 define weight deacy losses
        # # for var in tf.trainable_variables():
        # #     print(var.name)
        # # print('##########'*30)
        # wd_loss = 0
        # for weights in tl.layers.get_variables_with_name('W_conv2d', True, True):
        #     wd_loss += tf.contrib.layers.l2_regularizer(self.config.weight_deacy)(weights)
        # for W in tl.layers.get_variables_with_name('resnet_v1_50/E_DenseLayer/W', True, True):
        #     wd_loss += tf.contrib.layers.l2_regularizer(self.config.weight_deacy)(W)
        # for weights in tl.layers.get_variables_with_name('embedding_weights', True, True):
        #     wd_loss += tf.contrib.layers.l2_regularizer(self.config.weight_deacy)(weights)
        # for gamma in tl.layers.get_variables_with_name('gamma', True, True):
        #     wd_loss += tf.contrib.layers.l2_regularizer(self.config.weight_deacy)(gamma)
        # # for beta in tl.layers.get_variables_with_name('beta', True, True):
        # #     wd_loss += tf.contrib.layers.l2_regularizer(self.config.weight_deacy)(beta)
        # for alphas in tl.layers.get_variables_with_name('alphas', True, True):
        #     wd_loss += tf.contrib.layers.l2_regularizer(self.config.weight_deacy)(alphas)
        # # for bias in tl.layers.get_variables_with_name('resnet_v1_50/E_DenseLayer/b', True, True):
        # #     wd_loss += tf.contrib.layers.l2_regularizer(self.config.weight_deacy)(bias)
        # self.wd_loss = wd_loss
        #
        # # 3.5 total losses
        # self.total_loss = self.cross_entropy + self.wd_loss
        #
        # # 3.6 define the learning rate schedule
        # p = int(512.0 / self.config.batch_size)
        # lr_steps = [p * val for val in self.config.lr_steps]
        # print(lr_steps)
        # lr = tf.train.piecewise_constant(self.global_step_tensor, boundaries=lr_steps, values=[0.001, 0.0005, 0.0003, 0.0001],
        #                                  name='lr_schedule')
        # # 3.7 define the optimize method
        # opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=self.config.momentum)
        # # 3.8 get train op
        # grads = opt.compute_gradients(self.total_loss)
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     self.train_op = opt.apply_gradients(grads, global_step=self.global_step_tensor)
        # # train_op = opt.minimize(self.total_loss, global_step=global_step)
        # # 3.9 define the inference accuracy used during validate or test
        # pred = tf.nn.softmax(logit)
        # self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), self.labels), dtype=tf.float32), name='acc')
        # # 3.10 define sess
        # # config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # # config.gpu_options.allow_growth = True
        # # sess = tf.Session(config=config)

        summ_scalar = [cross_entropy, regularization_losses_sum, self.total_loss, accuracy, self.learning_rate]
        self.add_summary_scalar(summ_scalar)
        self.add_summary_histgoram(tf.trainable_variables())

        self.summ_hist = {}
        # for grad, var in grads:
        #     if grad is not None:
        #         self.summ_hist[var.op.name + '/gradients'] = grad

        for var in tf.trainable_variables():
            self.summ_hist[var.op.name] = var

        # Old
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # self.is_training = tf.placeholder(tf.bool)
        #
        # self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        # self.y = tf.placeholder(tf.float32, shape=[None, 10])
        #
        # # network architecture
        # d1 = tf.layers.dense(self.x, 512, activation=tf.nn.relu, name="dense1")
        # d2 = tf.layers.dense(d1, 10, name="dense2")
        #
        # with tf.name_scope("loss"):
        #     self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d2))
        #     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #     with tf.control_dependencies(update_ops):
        #         self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
        #                                                                                  global_step=self.global_step_tensor)
        #     correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(self.y, 1))
        #     self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def get_summary_dict(self):
        return self.summ_scalar, self.summ_hist

