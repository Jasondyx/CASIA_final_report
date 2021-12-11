
import numpy as np
import pandas as pd
import torch
import os
from torch.autograd import Variable
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from multiprocessing.pool import ThreadPool


n_sim = 100
score_table_all = pd.DataFrame(index=['score_MLE', 'score_MLE_L2', 'score_dropout_data', 'score_dropout_penalty'])
score_table_active = pd.DataFrame(index=['score_MLE', 'score_MLE_L2', 'score_dropout_data', 'score_dropout_penalty'])


def simulation(i_sim):

    print(f'----------i_sim={i_sim}-----------')
    if os.path.exists(f'./result/{i_sim}th-beta.npz'):
        r = np.load(f'./result/{i_sim}th-beta.npz')
        beta_MLE = r['beta_MLE']
        beta_MLE_L2 = r['beta_MLE_L2']
        beta_dropout_data = r['beta_dropout_data']
        beta_dropout_penalty = r['beta_dropout_penalty']
    else:
        # 1. beta estimation
        # data generation
        np.random.seed(i_sim)
        p = 1050
        p1, p2 = 50, 1000
        n, g = 75, 100
        C = 1/2 * np.log(3)

        X1 = np.zeros((n, p1))
        ind = np.random.choice(range(n), int(n/5), replace=False).reshape((int(n/25),-1))
        for i in range(ind.shape[1]):
            X1[np.ix_(ind[:, i], np.arange(10) + 10 * i)] = np.random.choice([-1, 1], (ind.shape[0], 1)) * np.random.exponential(1, (ind.shape[0], 10))

        beta1 = 0.2792 * np.ones(p1)
        y_ori = np.random.binomial(1, 1/(1 + np.exp(-X1.dot(beta1))))
        X_ori = np.hstack((X1, np.random.standard_normal((n, p2))))

        y = torch.from_numpy(y_ori).double()
        X = torch.from_numpy(X_ori).double()


        # MLE of LR
        beta_h = Variable(torch.ones(p,1).double(), requires_grad=True)
        Loss_old = 0
        while 1:
            Loss = torch.sum(- y.unsqueeze(1) * torch.mm(X, beta_h) + torch.log(1 + torch.exp(torch.mm(X, beta_h))))

            g_beta_h = torch.autograd.grad(Loss, beta_h, create_graph=True)
            beta_h = Variable(beta_h - 0.001 * g_beta_h[0].data, requires_grad=True)
            # print(abs(Loss - Loss_old))
            if abs(Loss - Loss_old) < 1e-5:
                print('Converged. (MLE of LR)')
                break
            Loss_old = Loss.clone().detach()
        beta_MLE = beta_h.clone().detach().numpy()


        # MLE with L2-regularization
        beta_h = Variable(torch.ones(p,1).double(), requires_grad=True)
        lamb = 32
        Loss_old = 0
        while 1:
            Loss = torch.sum(- y.unsqueeze(1) * torch.mm(X, beta_h) + torch.log(1 + torch.exp(torch.mm(X, beta_h)))) + lamb/2 * torch.mm(beta_h.T, beta_h)

            g_beta_h = torch.autograd.grad(Loss, beta_h, create_graph=True)
            beta_h = Variable(beta_h - 0.001 * g_beta_h[0].data, requires_grad=True)
            # print(abs(Loss - Loss_old))
            if abs(Loss - Loss_old) < 1e-5:
                print('Converged. (MLE with L2-regularization)')
                break
            Loss_old = Loss.clone().detach()
        beta_MLE_L2 = beta_h.clone().detach().numpy()


        # dropout in dataset
        delta = 0.9
        n_copies = 100
        X_dropout = np.tile(X_ori, (n_copies,1)) * np.random.binomial(1, 1 - delta, (n * n_copies, p)) * 1/(1-delta)
        y_dropout = np.tile(y_ori, n_copies)
        X_dr = torch.from_numpy(X_dropout).double()
        y_dr = torch.from_numpy(y_dropout).double()

        beta_h = Variable(1 * torch.ones(p,1).double(), requires_grad=True)
        optim = Adam([beta_h])
        Loss_old = 0
        while 1:
            Loss = torch.sum(- y_dr.unsqueeze(1) * torch.mm(X_dr, beta_h) + torch.log(1 + torch.exp(torch.mm(X_dr, beta_h))))

            optim.zero_grad()
            Loss.backward()
            optim.step()

            # g_beta_h = torch.autograd.grad(Loss, beta_h, create_graph=True)
            # beta_h = Variable(beta_h - 0.0001 * g_beta_h[0].data, requires_grad=True)
            if torch.isnan(Loss):
                print(Loss)
                break
            # print(abs(Loss - Loss_old))
            if abs(Loss - Loss_old) < 1e-5:
                print('Converged. (dropout in dataset)')
                break
            Loss_old = Loss.clone().detach()
        beta_dropout_data = beta_h.clone().detach().numpy()


        # dropout penalty
        beta_h = Variable(torch.ones(p,1).double(), requires_grad=True)
        lamb = 32
        delta = 0.9
        Loss_old = 0

        A_old = torch.zeros((n,))
        while 1:
            A = (torch.exp(torch.mm(X,beta_h)) / (1 + torch.exp(torch.mm(X,beta_h)))**2).clone().detach()
            # optimize beta for fixed A
            while 1:
                Loss = torch.sum(- y.unsqueeze(1) * torch.mm(X, beta_h) + torch.log(1 + torch.exp(torch.mm(X, beta_h)))) + lamb/2 * delta/(1 - delta) * torch.sum(A.squeeze(1) * torch.sum(X**2 * (beta_h**2).T, axis=1))

                g_beta_h = torch.autograd.grad(Loss, beta_h, create_graph=True)
                beta_h = Variable(beta_h - 0.0001 * g_beta_h[0].data, requires_grad=True)
                if torch.isnan(Loss):
                    print(Loss)
                # print(abs(Loss - Loss_old))
                if abs(Loss - Loss_old) < 1e-5:
                    print('beta converged.')
                    break
                Loss_old = Loss.clone().detach()
            if torch.norm(A - A_old) < 1e-5:
                print('Converged. (dropout penalty)')
                break
            A_old = A.clone()
        beta_dropout_penalty = beta_h.clone().detach().numpy()


    # 2. Accuracy Score
    # test data
    np.random.seed(i_sim+1)
    p = 1050
    p1, p2 = 50, 1000
    n, g = 75, 100
    C = 1/2 * np.log(3)

    X1 = np.zeros((n, p1))
    active_ind = np.random.choice(range(n), int(n/5), replace=False)
    ind = active_ind.reshape((int(n/25),-1))
    for i in range(ind.shape[1]):
        X1[np.ix_(ind[:,i],np.arange(10)+10*i)] = np.random.choice([-1, 1], (ind.shape[0], 1)) * np.random.exponential(1, (ind.shape[0],10))

    beta1 = 0.2792 * np.ones(p1)
    y_test = np.random.binomial(1, 1/(1 + np.exp(-X1.dot(beta1))))
    X_test = np.hstack((X1, np.random.standard_normal((n, p2))))
    y_test_active = y_test[active_ind]
    X_test_active = X_test[active_ind, :]

    # prediction
    threshold = 0.5
    # all data
    y_pre_MLE = (1/(1 + np.exp(-X_test.dot(beta_MLE))) > threshold).astype(int).squeeze(1)
    y_pre_MLE_L2 = (1/(1 + np.exp(-X_test.dot(beta_MLE_L2))) > threshold).astype(int).squeeze(1)
    y_pre_dropout_data = (1/(1 + np.exp(-X_test.dot(beta_dropout_data))) > threshold).astype(int).squeeze(1)
    y_pre_dropout_penalty = (1/(1 + np.exp(-X_test.dot(beta_dropout_penalty))) > threshold).astype(int).squeeze(1)

    score_MLE = accuracy_score(y_test, y_pre_MLE)
    score_MLE_L2 = accuracy_score(y_test, y_pre_MLE_L2)
    score_dropout_data = accuracy_score(y_test, y_pre_dropout_data)
    score_dropout_penalty = accuracy_score(y_test, y_pre_dropout_penalty)
    print('acc_score for all data:', score_MLE, score_MLE_L2, score_dropout_data, score_dropout_penalty)
    score_table_all[f'{i_sim}-th_sim'] = [score_MLE, score_MLE_L2, score_dropout_data, score_dropout_penalty]

    # active data
    y_pre_MLE = (1/(1 + np.exp(-X_test_active.dot(beta_MLE))) > threshold).astype(int).squeeze(1)
    y_pre_MLE_L2 = (1/(1 + np.exp(-X_test_active.dot(beta_MLE_L2))) > threshold).astype(int).squeeze(1)
    y_pre_dropout_data = (1/(1 + np.exp(-X_test_active.dot(beta_dropout_data))) > threshold).astype(int).squeeze(1)
    y_pre_dropout_penalty = (1/(1 + np.exp(-X_test_active.dot(beta_dropout_penalty))) > threshold).astype(int).squeeze(1)

    score_MLE = accuracy_score(y_test_active, y_pre_MLE)
    score_MLE_L2 = accuracy_score(y_test_active, y_pre_MLE_L2)
    score_dropout_data = accuracy_score(y_test_active, y_pre_dropout_data)
    score_dropout_penalty = accuracy_score(y_test_active, y_pre_dropout_penalty)
    print('acc_score for active data', score_MLE, score_MLE_L2, score_dropout_data, score_dropout_penalty)
    score_table_active[f'{i_sim}-th_sim'] = [score_MLE, score_MLE_L2, score_dropout_data, score_dropout_penalty]

    np.savez(f'./result/{i_sim}th-beta.npz', beta_MLE=beta_MLE, beta_MLE_L2=beta_MLE_L2, beta_dropout_data=beta_dropout_data,
             beta_dropout_penalty=beta_dropout_penalty)
    print(f'{i_sim}-th simulation ends.')

pool = ThreadPool(1)
pool.map(simulation, range(n_sim))
pool.close()
pool.join()

print('acc_score for all data: \n', score_table_all.mean(axis=1))
print('acc_score for active data: \n', score_table_active.mean(axis=1))

# save
score_table_all.to_pickle('./result/score_table_all.pkl')
score_table_active.to_pickle('./result/score_table_active.pkl')

print(0)

