import multiprocessing
import os
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from itertools import product
from shutil import rmtree
from psutil import cpu_count

from numpy import savetxt

from data_gen import data_gen, data_gen_MNIST, data_gen_REAL

from model import regession_model


def work(Coordinator, report_lock, loss_function, loss_para, data, base_path, repeat=1):
    os.makedirs(base_path)
    Model = regession_model(p=data['p'], K=data['K'], loss_function=loss_function, loss_para=loss_para)
    for r in range(repeat):
        if data['Env'] == 'Sim':
            X, Y = data_gen(Nk=data['Nk'], K=data['K'], p=data['p'], seed=hash("Train" + str(r)))
            TX, TY = data_gen(Nk=data['Nk'], K=data['K'], p=data['p'], seed=hash("Test" + str(r)))
        elif data['Env'] == 'MNIST':
            X, Y = data_gen_MNIST(data['Nk'] * data['K'], False, seed=hash("Train" + str(r)))
            TX, TY = data_gen_MNIST(data['Nk'] * data['K'], True, seed=hash("Test" + str(r)))
        elif data['Env'] == 'REAL':
            X, Y, TX, TY = data_gen_REAL(r)
        res = Model.full_auto(X, Y, TX, TY)
        local_path = os.path.join(base_path, str(r))
        os.makedirs(local_path)
        matrix_to_save = {'Cfm_train', 'Cfm_test', 'A', 'B'}
        for m in matrix_to_save:
            savetxt(os.path.join(local_path, str(m) + ".csv"), res[m], delimiter=",")
        if loss_function == 'DWD' or loss_function == 'DWDSM' or loss_function == 'DWDnc':
            loss_para_pad_to_3 = (loss_para['l'], loss_para['alpha'], loss_para['q'])
        elif loss_function == 'logistic':
            loss_para_pad_to_3 = (loss_para['l'], "-", "-")
        else:
            raise

        settings = (data['K'], data['p'], data['Nk'], loss_function)
        rps = (
            str(r), res['Loss_train'], res['Loss_test'], res['Ac_train'], res['Ac_test'], res['large_parameter'],
            res['i'])
        to_report = list(map(lambda x: str(x), settings + loss_para_pad_to_3 + rps))
        report_lock.acquire()
        print(",".join(to_report),
              file=open(os.path.join(os.getcwd(), "results", "env=" + data['Env'], "sum.csv"), 'a'))
        report_lock.release()
        Model.reset()
    Coordinator.release()
    return None


def path_gen(Env, K, p, Nk, loss_function, l, al, q):
    return os.path.join(os.getcwd(), "results", "env=" + str(Env), "K=" + str(K), "p=" + str(p),
                        "Nk=" + str(Nk), "loss_function=" + str(loss_function), "Lambda=" + str(l),
                        "Alpha=" + str(al), "q=" + str(q))


if __name__ == '__main__':
    Env = 'Sim'
    Repeat = 100
    Coordinator = multiprocessing.Semaphore(cpu_count(logical = True)+4)
    rmtree("./results", ignore_errors=True)
    os.makedirs(os.path.join(os.getcwd(), "results", "env=" + str(Env)))
    Names = (
        "K", "p", "Nk", "Loss_f", "Lambda", "Alpha", "q", "id", "Loss_train", "Loss_test", "Ac_train", "Ac_test",
        "non_zero", "i")
    print(",".join(Names), file=open(os.path.join(os.getcwd(), "results", "env=" + str(Env), "sum.csv"), 'w+'),
          flush=True)
    report_lock = multiprocessing.Lock()
    if Env == 'Sim':
        Env_combination = ((3, 150, 50), (3, 150, 100), (3, 300, 50), (3, 300, 100),
                           (5, 150, 50), (5, 150, 100), (5, 300, 50), (5, 300, 100))
        for K, p, Nk in reversed(Env_combination):
            # Logistic
            ls = {0.3, 0.1, 0.06, 0.03, 0.01, 0.006, 0.003, 0.001}
            for l in ls:
                base_path = path_gen(Env, K, p, Nk, "logistic", l, "NA", "NA")
                Coordinator.acquire()
                print("working on: " + base_path)
                multiprocessing.Process(target=work, args=(
                    Coordinator, report_lock, 'logistic', {'l': l}, {'Env': 'Sim', 'K': K, 'p': p, 'Nk': Nk}, base_path,
                    Repeat)).start()
            # DWD
            ls = {0.06, 0.03, 0.01, 0.006, 0.005, 0.004, 0.003, 0.001, 0.0006}
            als = {0.1, 0.5, 0.9}
            qs = {0.5, 1, 20}
            for dwdv in ["DWDnc"]:  # ,"DWDnc","DWDSM"
                for l, al, q in product(ls, als, qs):
                    base_path = path_gen(Env, K, p, Nk, dwdv, l, al, q)
                    Coordinator.acquire()
                    print("working on: " + base_path)
                    # report = work (Coordinator,report_lock, 'DWD', {'l':l,'alpha':al,'q':q}, {'K':K,'p':p,'Nk':Nk},base_path,Repeat )
                    multiprocessing.Process(target=work, args=(
                        Coordinator, report_lock, dwdv, {'l': l, 'alpha': al, 'q': q},
                        {'Env': Env, 'K': K, 'p': p, 'Nk': Nk},
                        base_path, Repeat)).start()

    elif Env == 'MNIST':
        K = 10
        p = 28 * 28
        for Nk in (50, 100, 200, 700):
            # DWD
            ls = {0.06, 0.03, 0.01, 0.006, 0.005, 0.004, 0.003, 0.001, 0.0006}
            als = {0.1, 0.5, 0.9}
            qs = {0.5, 1, 20}
            for dwdv in ["DWD"]:  # ,"DWDnc","DWDSM"
                for l, al, q in product(ls, als, qs):
                    base_path = path_gen(Env, K, p, Nk, dwdv, l, al, q)
                    Coordinator.acquire()
                    print("working on: " + base_path)
                    multiprocessing.Process(target=work, args=(
                        Coordinator, report_lock, dwdv, {'l': l, 'alpha': al, 'q': q},
                        {'Env': Env, 'K': K, 'p': p, 'Nk': Nk},
                        base_path, Repeat)).start()
            # Logistic
            for l in ls:
                base_path = path_gen(Env, K, p, Nk, "logistic", l, "NA", "NA")
                Coordinator.acquire()
                print("working on: " + base_path)
                multiprocessing.Process(target=work, args=(
                    Coordinator, report_lock, 'logistic', {'l': l}, {Env: Env, 'K': K, 'p': p, 'Nk': Nk},
                    base_path,
                    Repeat)).start()
    elif Env == 'REAL':
        K = 4
        p = 1714
        Nk = "NA"
        #ls = {3,1,0.6,0.3,0.1,0.06, 0.03, 0.01, 0.006, 0.005, 0.004, 0.003, 0.001, 0.0006,0.0003,0.0001}
        ls = list(map(lambda x: (x)/100,list(range(41))))
        als = {0.1, 0.5, 0.9}
        qs = {0.5, 1, 20}
        for dwdv in ["DWD", "DWDnc", "DWDSM"]:  # ,"DWDnc","DWDSM"
            for l, al, q in product(ls, als, qs):
                base_path = path_gen(Env, K, p, Nk, dwdv, l, al, q)
                Coordinator.acquire()
                print("working on: " + base_path)
                multiprocessing.Process(target=work, args=(
                    Coordinator, report_lock, dwdv, {'l': l, 'alpha': al, 'q': q},
                    {'Env': Env, 'K': K, 'p': p, 'Nk': Nk},
                    base_path, Repeat)).start()
            # Logistic
        for l in ls:
            base_path = path_gen(Env, K, p, Nk, "logistic", l, "NA", "NA")
            Coordinator.acquire()
            print("working on: " + base_path)
            multiprocessing.Process(target=work, args=(
            Coordinator, report_lock, 'logistic', {'l': l}, {'Env': Env, 'K': K, 'p': p, 'Nk': Nk}, base_path,
            Repeat)).start()
