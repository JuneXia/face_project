import tensorflow as tf


class BaseModel:
    def __init__(self, config):
        self.config = config
        self.summary_histogram = {}
        self.summary_scalar = {}
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def add_summary(self, summarys):
        raise Exception('发现有的张量shape是未知的，所以无法通过其shape来判断是scalar还是histgoram。\
                         所以又重写了add_summary_scalar和add_summary_histogram')

        if type(summarys) == dict:
            for k, v in summarys.items():
                if len(v.get_shape().as_list()) == 0:
                    self.summary_scalar[k] = v
                else:
                    self.summary_histogram[k] = v
        elif type(summarys) == list:
            for tensor in summarys:
                if len(tensor.get_shape().as_list()) == 0:
                    self.summary_scalar[tensor.op.name] = tensor
                else:
                    self.summary_histogram[tensor.op.name] = tensor
        else:
            raise Exception('not supported temporary!')

    def add_summary_scalar(self, summarys):
        if type(summarys) == dict:
            for k, v in summarys.items():
                self.summary_scalar[k] = v
        elif type(summarys) == list:
            # raise Exception('not supported temporary!')

            for tensor in summarys:
                self.summary_scalar[tensor.op.name] = tensor
        else:
            raise Exception('not supported temporary!')

    def add_summary_histgoram(self, summarys):
        if type(summarys) == dict:
            for k, v in summarys.items():
                self.summary_histogram[k] = v
        elif type(summarys) == list:
            # raise Exception('not supported temporary!')

            for tensor in summarys:
                self.summary_histogram[tensor.op.name] = tensor
        else:
            raise Exception('not supported temporary!')

    def init_saver(self):
        # just copy the following line in your child class
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError
