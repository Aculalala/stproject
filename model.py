from math import isnan

import numpy as np
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

class regession_model():
    def __init__(self, p, K=4, loss_function='DWD', loss_para={}):
        self.p = p
        self.K = K
        self.loss_function = loss_function
        self.loss_para = loss_para
        self.__build_graph()

    def __build_graph(self):
        TF_Var_A_pre = tf.Variable(np.random.randn(self.K - 1), trainable=True, dtype=tf.float32, name='PA')
        TF_Var_B_pre = tf.Variable(np.random.randn(self.p, self.K - 1), trainable=True, dtype=tf.float32, name='PB')

        TF_Var_A_last_element = -tf.reduce_sum(TF_Var_A_pre, keepdims=True)
        TF_Var_B_last_col = -tf.reduce_sum(TF_Var_B_pre, axis=1, keepdims=True)

        self.TF_Var_A = tf.concat((TF_Var_A_pre, TF_Var_A_last_element), axis=0)
        self.TF_Var_B = tf.concat((TF_Var_B_pre, TF_Var_B_last_col), axis=1)

        self.TF_X = tf.placeholder(dtype=tf.float32, shape=(None, None))
        self.TF_Y = tf.placeholder(dtype=tf.int32, shape=(None))

        self.XB = tf.matmul(self.TF_X, self.TF_Var_B)
        self.XBpA = self.XB + self.TF_Var_A
        self.F = tf.gather_nd(self.XBpA, tf.stack((tf.range(tf.shape(self.XBpA)[0], dtype=self.TF_Y.dtype), self.TF_Y),
                                                  axis=1))  # Fuck tensorflow

        if self.loss_function == 'DWD':
            lambda_1 = self.loss_para['l'] * self.loss_para['alpha']
            lambda_2 = self.loss_para['l'] * (1 - self.loss_para['alpha'])

            def TF_phi(x, q):
                Q = q / (q + 1)
                return tf.where(tf.greater(x, Q), (Q / tf.abs(x)) ** (Q),
                                1 - x)  # Abs due to https://stackoverflow.com/questions/50187342/tensorflow-gradient-with-tf-where-returns-nan-when-it-shouldnt

            self.loss_1 = tf.reduce_mean(TF_phi(self.F, self.loss_para['q']))
            self.loss_2 = lambda_1 * tf.reduce_mean(tf.abs(self.TF_Var_B))
            self.loss_3 = lambda_2 * tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(self.TF_Var_B), axis=0)))
            self.loss = self.loss_1 + self.loss_2 + self.loss_3
        elif self.loss_function == 'SVM':
            def TF_phi_infty(x):
                return tf.nn.relu(1 - x)
                # return tf.where(tf.greater(x, 1), 0.0*x, 1 - x)

            self.loss_1 = tf.reduce_mean(TF_phi_infty(self.F))
            self.loss_2 = self.loss_para['l'] * tf.reduce_mean(tf.abs(self.TF_Var_B))
            self.loss = self.loss_1 + self.loss_2
        elif self.loss_function == 'logistic':
            self.loss_1 = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.TF_Y, depth=self.K),
                                                        logits=tf.nn.softmax(self.XBpA)))
            self.loss_2 = self.loss_para['l'] * tf.reduce_mean(tf.abs(self.TF_Var_B))
            self.loss = self.loss_1 + self.loss_2

        self.predictions = tf.cast(tf.argmax(self.XBpA, 1), tf.int32)
        self.ac = tf.reduce_mean(tf.cast(tf.equal(self.TF_Y, self.predictions), tf.float32))
        self.cfm = tf.math.confusion_matrix(self.TF_Y, self.predictions, num_classes=self.K, weights=None)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = self.optimizer.minimize(self.loss)
        self.trainable_variables = tf.trainable_variables()
        # clipped_grads_and_vars = [(tf.clip_by_value(grad, -5, 5), var) for grad, var in self.optimizer.compute_gradients(self.loss, trainable_variables)]
        # self.train_op_safe = self.optimizer.apply_gradients( clipped_grads_and_vars )
        self.gradients = self.optimizer.compute_gradients(self.loss, self.trainable_variables)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    def train(self, X, Y):
        # print(self.sess.run(self.loss , feed_dict={self.TF_X: X, self.TF_Y: Y}).shape)
        return self.sess.run([self.train_op, self.loss, self.ac], feed_dict={self.TF_X: X, self.TF_Y: Y})

    def debug(self, X, Y):
        all_tensors = [tensor for op in tf.get_default_graph().get_operations() for tensor in op.values()]
        for n in all_tensors:
            print(n.name, self.sess.run(n, feed_dict={self.TF_X: X, self.TF_Y: Y}))
        return None

    def full_auto(self, X, Y, TX, TY):
        auto_report = {}
        i = 0
        before = float('inf')
        while True:
            i += 1
            # print(i)
            after = self.train(X, Y)[1]
            # print(after)
            if isnan(after) or (before < after + 1e-5 and i > 1000):
                break
            before = after
        auto_report['Loss_train'], auto_report['Ac_train'], auto_report['Cfm_train'] = self.sess.run(
            [self.loss, self.ac, self.cfm],
            feed_dict={self.TF_X: X,
                       self.TF_Y: Y})
        auto_report['Loss_test'], auto_report['Ac_test'], auto_report['Cfm_test'] = self.sess.run(
            [self.loss, self.ac, self.cfm],
            feed_dict={self.TF_X: TX,
                       self.TF_Y: TY})
        auto_report['A'], auto_report['B'] = self.sess.run([self.TF_Var_A, self.TF_Var_B])
        auto_report['i'] = i
        return auto_report

    def infer(self):
        pass
