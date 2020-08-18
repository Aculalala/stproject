from data_gen import data_gen
from model import regession_model

X, Y = data_gen(Nk=100, K=2, p=2, d=3)
TX, TY = data_gen(Nk=100, K=2, p=2, d=3)
Model = regession_model(p=2, K=2, loss_function='DWD', loss_para={'l': 1e-3, 'alpha': 0.5, 'q': 1})
while True:
    print(Model.sess.run([Model.loss_1, Model.loss_2, Model.loss_3], feed_dict={Model.TF_X: X, Model.TF_Y: Y}))
    print(Model.debug(X, Y))
    raise
