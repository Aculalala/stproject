import multiprocessing
import os
from itertools import product
from shutil import rmtree

from numpy import savetxt

from data_gen import data_gen
from model import regession_model


def work(Coordinator, report_lock, loss_function, loss_para, data, base_path, r):
    X, Y = data_gen(Nk=data['Nk'], K=data['K'], p=data['p'], d=3)
    TX, TY = data_gen(Nk=data['Nk'], K=data['K'], p=data['p'], d=3)
    Model = regession_model(p=data['p'], K=data['K'], loss_function=loss_function, loss_para=loss_para)
    res = Model.full_auto(X, Y, TX, TY)
    local_path = os.path.join(base_path, str(r))
    os.makedirs(local_path)
    matrix_to_save = {'Cfm_train', 'Cfm_test', 'A', 'B'}
    for m in matrix_to_save:
        savetxt(os.path.join(local_path, str(m) + ".csv"), res[m], delimiter=",")
    if loss_function == 'DWD':
        loss_para_pad_to_3 = (loss_para['l'], loss_para['alpha'], loss_para['q'])
    elif loss_function == 'SVM':
        loss_para_pad_to_3 = (loss_para['l'], "-", "-")
    elif loss_function == 'logistic':
        loss_para_pad_to_3 = (loss_para['l'], "-", "-")
    else:
        raise

    settings = (data['K'], data['p'], data['Nk'], loss_function)
    rps = (str(r), res['Loss_train'], res['Loss_test'], res['Ac_train'], res['Ac_test'])
    to_report = list(map(lambda x: str(x), settings + loss_para_pad_to_3 + rps))
    report_lock.acquire()
    print(",".join(to_report), file=open("./results/Detail_summary.csv", 'a'))
    report_lock.release()
    Coordinator.release()
    return res


Env = 'Sim'
Repeat = 10
if __name__ == '__main__':
    if Env == 'Sim':
        def path_gen(Env, K, p, Nk, loss_function, l, al, q):
            return os.path.join(os.getcwd(), "results", "Sim", "env=" + str(Env), "K=" + str(K), "p=" + str(p),
                                "Nk=" + str(Nk), "loss_function=" + str(loss_function), "Lambda=" + str(l),
                                "Alpha=" + str(al), "q=" + str(q))


        Coordinator = multiprocessing.Semaphore(16)
        rmtree("./results", ignore_errors=True)
        os.makedirs(os.path.join(os.getcwd(), "results"))
        Names = (
        "K", "p", "Nk", "Loss_f", "Lambda", "Alpha", "q", "id", "Loss_train", "Loss_test", "Ac_train", "Ac_test")
        print(",".join(Names), file=open("./results/Detail_summary.csv", 'w+'), flush=True)
        report_lock = multiprocessing.Lock()

        Ks = {2, 3, 7}
        ps = {10, 100, 500}
        Nks = {10, 100, 200}
        for K, p, Nk in product(Ks, ps, Nks):
            # SVM
            ls = {0, 1e-3, 1e-5, 1e-7}
            for l in ls:
                base_path = path_gen(Env, K, p, Nk, "SVM", l, "NA", "NA")
                os.makedirs(base_path)
                for r in range(Repeat):
                    Coordinator.acquire()
                    print("working on: " + base_path + ":repeat" + str(r))
                    multiprocessing.Process(target=work, args=(
                    Coordinator, report_lock, 'SVM', {'l': l}, {'K': K, 'p': p, 'Nk': Nk}, base_path, r)).start()

            # SVM
            ls = {0, 1e-3, 1e-5, 1e-7}
            for l in ls:
                base_path = path_gen(Env, K, p, Nk, "logistic", l, "NA", "NA")
                os.makedirs(base_path)
                for r in range(Repeat):
                    Coordinator.acquire()
                    print("working on: " + base_path + ":repeat" + str(r))
                    multiprocessing.Process(target=work, args=(
                    Coordinator, report_lock, 'logistic', {'l': l}, {'K': K, 'p': p, 'Nk': Nk}, base_path, r)).start()

            # DWD
            ls = {0, 1e-3, 1e-5, 1e-7}
            als = {0.1, 0.5, 0.9}
            qs = {1, 3, 0.1}
            for l, al, q in product(ls, als, qs):
                base_path = path_gen(Env, K, p, Nk, "DWD", l, al, q)
                os.makedirs(base_path)
                for r in range(Repeat):
                    Coordinator.acquire()
                    print("working on: " + base_path + ":repeat" + str(r))
                    # report = work (Coordinator,report_lock, 'DWD', {'l':l,'alpha':al,'q':q}, {'K':K,'p':p,'Nk':Nk},base_path,r)
                    multiprocessing.Process(target=work, args=(
                    Coordinator, report_lock, 'DWD', {'l': l, 'alpha': al, 'q': q}, {'K': K, 'p': p, 'Nk': Nk},
                    base_path, r)).start()
