import tensorflow as tf


class BaseTrain:
    def __init__(self, sess, model, config, logger, train_data, val_data=None, val_freq=1, eval_data=None, eval_freq=1):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.train_data = train_data
        self.val_data = val_data
        self.val_freq = val_freq
        self.eval_data = eval_data
        self.eval_freq = eval_freq
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self, max_epochs=100, epoch_size=1000):
        # for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.max_epochs + 1, 1):
        for cur_epoch in range(1, max_epochs + 1):
            self.train_epoch(cur_epoch, epoch_size)
            if self.val_data and (cur_epoch % self.val_freq == 0):
                self.validation()
            if self.eval_data and (cur_epoch % self.eval_freq == 0):
                self.evaluation()

            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self, cur_epoch, epoch_size=1000):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError

    def validation(self):
        """
        implement the logic of the validation
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError

    def evaluation(self):
        """
        implement the logic of the evaluation
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
