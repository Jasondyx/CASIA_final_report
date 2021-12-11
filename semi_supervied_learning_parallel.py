
import numpy as np
import pandas as pd
import os
import torch
from torch.autograd import Variable
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from multiprocessing.pool import ThreadPool
import matplotlib.pyplot as plt


n_sim = 10
unlabel_size_list = np.concatenate((([25, 50, 75, 150] ,75 * np.arange(3, 15, 2))))
score_table = pd.DataFrame(index=['score_dropout', 'score_dropout_semi'])

def semi_supervised_simulation(unlabel_size):

    print(f'----------unlabel_size={unlabel_size}-----------')

    if os.path.exists(f'./result/semi_supervised/unlabel_size_{unlabel_size}.pkl'):
        score_table_t = pd.read_pickle(f'./result/semi_supervised/unlabel_size_{unlabel_size}.pkl')
    else:
        score_table_t = pd.DataFrame(index=['score_dropout', 'score_dropout_semi'])
    for i_sim in range(n_sim):
        if f'{i_sim}-th_sim' in score_table_t.columns.to_list():
            print(f'{i_sim}-th_sim of size:{unlabel_size} existed.')
            continue
        np.random.seed(i_sim)
        p = 1050
        p1, p2 = 50, 1000
        n_label, n_unlabel = 75, unlabel_size
        n = n_label + n_unlabel

        X1 = np.zeros((n, p1))
        ind = np.random.choice(range(n), int(n/5), replace=False).reshape((int(n/25),-1))
        for i in range(ind.shape[1]):
            X1[np.ix_(ind[:,i],np.arange(10)+10*i)] = np.random.choice([-1, 1], (ind.shape[0], 1)) * np.random.exponential(1, (ind.shape[0],10))

        beta1 = 0.2792 * np.ones(p1)
        y_ori = np.random.binomial(1, 1/(1 + np.exp(-X1.dot(beta1))))
        X_ori = np.hstack((X1, np.random.standard_normal((n, p2))))

        y_label = y_ori[:n_label]
        X_label = X_ori[:n_label,:]
        X_unlabel = X_ori[n_label:,:]

        y = torch.from_numpy(y_label).double()
        X = torch.from_numpy(X_label).double()
        X_unlab = torch.from_numpy(X_unlabel).double()

        # dropout penalty with only labeled data
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
                print('Converged. (dropout)')
                break
            A_old = A.clone()
        beta_dropout = beta_h.clone().detach().numpy()

        # semi-supervised
        beta_h = Variable(torch.ones(p,1).double(), requires_grad=True)
        lamb = 32
        delta = 0.9
        alpha = 0.4
        Loss_old = 0

        A_old = torch.zeros((n,))
        A_unlab_old = torch.zeros((n,))
        while 1:
            A = (torch.exp(torch.mm(X, beta_h)) / (1 + torch.exp(torch.mm(X, beta_h)))**2).clone().detach()
            A_unlab = (torch.exp(torch.mm(X_unlab, beta_h)) / (1 + torch.exp(torch.mm(X_unlab, beta_h))) ** 2).clone().detach()
            # optimize beta for fixed A
            while 1:
                Loss = torch.sum(- y.unsqueeze(1) * torch.mm(X, beta_h) + torch.log(1 + torch.exp(torch.mm(X, beta_h)))) + \
                       n_label/(n_label + alpha*n_unlabel) * (
                        lamb/2 * delta/(1 - delta) * torch.sum(A.squeeze(1) * torch.sum(X**2 * (beta_h**2).T, axis=1)) +
                        alpha * lamb/2 * delta/(1 - delta) * torch.sum(A_unlab.squeeze(1) * torch.sum(X_unlab**2 * (beta_h**2).T, axis=1)))

                g_beta_h = torch.autograd.grad(Loss, beta_h, create_graph=True)
                beta_h = Variable(beta_h - 0.0001 * g_beta_h[0].data, requires_grad=True)
                if torch.isnan(Loss):
                    print(Loss)
                # print(abs(Loss - Loss_old))
                if abs(Loss - Loss_old) < 1e-5:
                    print('beta converged.')
                    break
                Loss_old = Loss.clone().detach()
            if torch.norm(A - A_old) < 1e-5 and torch.norm(A_unlab - A_unlab_old) < 1e-5:
                print('Converged. (dropout semi-supervised)')
                break
            A_old = A.clone()
            A_unlab_old = A_unlab.clone()
        beta_dropout_semi = beta_h.clone().detach().numpy()

        # 2. Accuracy Score
        # test data
        np.random.seed(i_sim+1)
        p = 1050
        p1, p2 = 50, 1000
        n = 75

        X1 = np.zeros((n, p1))
        active_ind = np.random.choice(range(n), int(n/5), replace=False)
        ind = active_ind.reshape((int(n/25),-1))
        for i in range(ind.shape[1]):
            X1[np.ix_(ind[:,i],np.arange(10)+10*i)] = np.random.choice([-1, 1], (ind.shape[0], 1)) * np.random.exponential(1, (ind.shape[0],10))

        beta1 = 0.2792 * np.ones(p1)
        y_test = np.random.binomial(1, 1/(1 + np.exp(-X1.dot(beta1))))
        X_test = np.hstack((X1, np.random.standard_normal((n, p2))))

        # prediction
        threshold = 0.5
        # all data
        y_pre_dropout = (1/(1 + np.exp(-X_test.dot(beta_dropout))) > threshold).astype(int).squeeze(1)
        y_pre_dropout_semi = (1/(1 + np.exp(-X_test.dot(beta_dropout_semi))) > threshold).astype(int).squeeze(1)

        score_dropout = accuracy_score(y_test, y_pre_dropout)
        score_dropout_semi = accuracy_score(y_test, y_pre_dropout_semi)
        print('acc_score for all data:', score_dropout, score_dropout_semi)
        score_table_t[f'{i_sim}-th_sim'] = [score_dropout, score_dropout_semi]

        score_table_t.to_pickle(f'./result/semi_supervised/unlabel_size_{unlabel_size}.pkl')

    score_table[f'unlabel_size={unlabel_size}'] = score_table_t.mean(axis=1)

pool = ThreadPool(4)
pool.map(semi_supervised_simulation, unlabel_size_list)
pool.close()
pool.join()

name_list = [f'unlabel_size={unlabel_size}' for unlabel_size in unlabel_size_list]
score_table = score_table[name_list]
score_table.to_pickle('./result/semi_supervised/score_table_over_unlabel_size.pkl')

# plot
plt.plot(unlabel_size_list, score_table.T)
plt.legend(score_table.index.to_list())
plt.xlabel('Size of Unlabeled Data')
plt.ylabel('Accuracy Score')
plt.savefig('./result/semi_supervised/Semi_supervised_learning.png')
plt.show()

print(0)