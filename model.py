import tensorflow as tf

from data_gen import *


class regession_model():
    def __init__(self, X, Y, K=4, q=1, lambda_1=1e-3, lambda_2=1e-3):
        self.X = X
        self.Y = Y
        self.p = X.shape[1]
        self.K = K
        self.q = q
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.__build_graph()

    def __build_graph(self):
        TF_Var_A_pre = tf.Variable(np.ones(shape=(self.K - 1)), trainable=True, dtype=tf.float32)
        TF_Var_B_pre = tf.Variable(np.ones(shape=(self.p, self.K - 1)), trainable=True, dtype=tf.float32)

        TF_Var_A_last_element = -tf.reduce_sum(TF_Var_A_pre, keepdims=True)
        TF_Var_B_last_col = -tf.reduce_sum(TF_Var_B_pre, axis=1, keepdims=True)

        self.TF_Var_A = tf.concat((TF_Var_A_pre, TF_Var_A_last_element), axis=0)
        self.TF_Var_B = tf.concat((TF_Var_B_pre, TF_Var_B_last_col), axis=1)

        def TF_phi(x, q):
            return tf.where(tf.greater(x, q / (q + 1)), (q / x / (q + 1)) ** q / (q + 1), 1 - x)

        self.TF_X = tf.convert_to_tensor(X, dtype=tf.float32)
        self.TF_Y = tf.convert_to_tensor(Y)

        XB = tf.matmul(self.TF_X, self.TF_Var_B)
        print(XB)
        F = tf.nn.embedding_lookup(self.TF_Var_A, Y) + tf.reduce_sum(tf.one_hot(Y, self.K) * XB, axis=1)
        self.loss_1 = tf.reduce_mean(TF_phi(F, self.q))
        self.loss_2 = self.lambda_1 * tf.reduce_sum(tf.abs(self.TF_Var_B))
        self.loss_3 = self.lambda_2 * tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(self.TF_Var_B), axis=0)))
        self.loss = self.loss_1 + self.loss_2 + self.loss_3

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)
        self.train_op = self.optimizer.minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    def train(self):
        return self.sess.run([self.train_op, self.loss])[1]

    def report(self):
        def cut_off(x):
            return np.where(np.abs(x) > 1e-3, x, 0),

        return cut_off(self.sess.run(self.TF_Var_B))


X, Y = data_gen()
Model = regession_model(X, Y)
for i in range(10):
    for j in range(1000):
        print(Model.train())
Final = np.array(Model.report()).T
print(Final)
