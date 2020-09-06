from math import isnan
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()




def relu(x):
    return (x > 0) * x
class regession_model():
    def __init__(self, p, K=4, loss_function='DWD', loss_para={}):
        self.i = 0
        self.p = p
        self.K = K
        self.loss_function = loss_function
        self.loss_para = loss_para
        self.__build_graph()

    def __build_graph(self):
        init_scale = 1 / np.sqrt(self.p)
        # init_scale = 0.0
        self.TF_Var_A = tf.Variable(init_scale * np.random.randn(self.K), trainable=True, dtype=tf.float32, name='PA')
        self.TF_Var_B = tf.Variable(init_scale * np.random.randn(self.p, self.K), trainable=True, dtype=tf.float32,
                                    name='PB')

        self.TF_X = tf.placeholder(dtype=tf.float32, shape=(None, None))
        self.TF_Y = tf.placeholder(dtype=tf.int32, shape=(None))

        self.XB = tf.matmul(self.TF_X, self.TF_Var_B)
        self.XBpA = self.XB + self.TF_Var_A

        if self.loss_function == 'DWD' or self.loss_function == 'DWDnc' or self.loss_function == 'DWDSM':
            self.ss = self.K * self.loss_para['q'] / ((self.loss_para['q'] + 1) ** 2) / self.p
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.ss)
            if self.loss_function == 'DWDSM':
                self.XSM = tf.nn.softmax(self.XBpA)
                self.F = tf.gather_nd(self.XSM,
                                      tf.stack((tf.range(tf.shape(self.XSM)[0], dtype=self.TF_Y.dtype), self.TF_Y),
                                               axis=1))
            else:
                self.F = tf.gather_nd(self.XBpA,
                                      tf.stack((tf.range(tf.shape(self.XBpA)[0], dtype=self.TF_Y.dtype), self.TF_Y),
                                               axis=1))
            self.lambda_1 = self.loss_para['l'] * self.loss_para['alpha']
            self.lambda_2 = self.loss_para['l'] * (1 - self.loss_para['alpha'])


            def TF_phi(x, q):
                Q = q / (q + 1)
                return tf.where(tf.greater(x, Q), (Q / tf.abs(x)) ** (Q),
                                1 - x)  # Abs due to https://stackoverflow.com/questions/50187342/tensorflow-gradient-with-tf-where-returns-nan-when-it-shouldnt

            self.loss_2 = self.lambda_1 * tf.reduce_mean(tf.abs(self.TF_Var_B))
            self.loss_3 = self.lambda_2 * tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(self.TF_Var_B), axis=0)))
            self.loss_1 = tf.reduce_mean(TF_phi(self.F, self.loss_para['q']))
            self.train_op = self.optimizer.minimize(self.loss_1)
            self.loss = self.loss_1 + self.loss_2 + self.loss_3

            self.TF_parameter_override = tf.placeholder(dtype=tf.float32)
            self.set_parameter_A = tf.assign(self.TF_Var_A, self.TF_parameter_override)
            self.set_parameter_B = tf.assign(self.TF_Var_B, self.TF_parameter_override)


        elif self.loss_function == 'logistic':
            ss = 0.01
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=ss)
            self.loss_1 = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.TF_Y, depth=self.K),
                                                        logits=self.XBpA))
            self.loss_2 = self.loss_para['l'] * tf.reduce_mean(tf.abs(self.TF_Var_B))
            self.loss = self.loss_1 + self.loss_2
            self.shrink = tf.assign(self.TF_Var_B,
                                    tf.math.sign(self.TF_Var_B) * tf.nn.relu(tf.math.abs(self.TF_Var_B) - ss * self.loss_para['l']))
            self.train_op = [self.optimizer.minimize(self.loss_1), self.shrink]

        self.predictions = tf.cast(tf.argmax(self.XBpA, 1), tf.int32)
        self.ac = tf.reduce_mean(tf.cast(tf.equal(self.TF_Y, self.predictions), tf.float32))
        self.cfm = tf.math.confusion_matrix(self.TF_Y, self.predictions, num_classes=self.K, weights=None)

        self.initializer = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.reset()

    def train(self, X, Y):
        self.sess.run([self.train_op], feed_dict={self.TF_X: X, self.TF_Y: Y})
        if self.loss_function == 'DWD' or self.loss_function == 'DWDnc' or self.loss_function == 'DWDSM':
            # To be changed into tensor so the parameters don't need to leave GRAM
            A, B = self.sess.run([self.TF_Var_A, self.TF_Var_B])
            A = A - np.mean(A)
            self.sess.run([self.set_parameter_A], feed_dict={self.TF_parameter_override: A})
            B = np.sign(B) * relu(np.abs(B) - self.lambda_1 * self.ss)
            row_eff = relu(1 - self.ss * self.lambda_2 / (np.linalg.norm(B, ord=2, axis=1) + 1e-8))
            B = B * row_eff[:, np.newaxis]
            if self.loss_function == 'DWD':
                B[self.i % self.K, ::] -= np.sum(B, axis=0)
            self.sess.run([self.set_parameter_B], feed_dict={self.TF_parameter_override: B})
        return self.sess.run([self.loss, self.ac], feed_dict={self.TF_X: X, self.TF_Y: Y})

    def debug(self, X, Y):
        all_tensors = [tensor for op in tf.get_default_graph().get_operations() for tensor in op.values()]
        for n in all_tensors:
            print(n.name, self.sess.run(n, feed_dict={self.TF_X: X, self.TF_Y: Y}))
        return None

    def full_auto(self, X, Y, TX, TY):
        retry = True
        auto_report = {}
        before = float('inf')
        while True:
            self.i += 1
            # print(i)
            after = self.train(X, Y)[0]
            # print(after)
            if (before < after + 1e-6 and self.i > 1000) or self.i > 50000:
                break
            if isnan(after):
                if retry:
                    self.reset()
                    before = float('inf')
                    retry = False
                    continue
                else:
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
        auto_report['large_parameter'] = np.count_nonzero(np.abs(auto_report['B']) > 1e-4)
        auto_report['i'] = self.i
        return auto_report

    def reset(self):
        self.sess.run(self.initializer)
        self.i =0
