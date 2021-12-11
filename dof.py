import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt


np.random.seed(0)
nrep = 200
delta_list = np.arange(1, 10) * 0.1
dof_lr_list, dof_ls_list, dof_dropout_list = np.zeros(nrep), np.zeros(nrep), np.zeros((nrep, len(delta_list)))
for i in range(nrep):
    # data generation
    n = 1000
    p = 50
    SNR = 0.99

    X_ori = np.random.standard_normal((n, p))
    beta = np.random.uniform(1, 10, p) * (-1)**np.random.binomial(1, 0.5, p)
    beta[np.random.choice(np.arange(p), int(p*1/5), replace=False)] = 0
    y_ = X_ori.dot(beta)
    sigma_e = np.std(y_) * np.sqrt(1 - SNR) / np.sqrt(SNR)
    e = np.random.standard_normal((n,)) * sigma_e
    y = y_ + e

    LR = LinearRegression().fit(X_ori, y)
    y_hat = LR.predict(X_ori)
    dof_lr = 1/sigma_e**2 * np.sum(e * (y_hat - y_))
    dof_lr_list[i] = dof_lr

    LS = Lasso().fit(X_ori, y)
    y_hat_ls = LS.predict(X_ori)
    dof_ls = 1/sigma_e**2 * np.sum(e * (y_hat_ls - y_))
    dof_ls_list[i] = dof_ls

    # dropout
    # delta = 0.2
    for j, delta in enumerate(delta_list):
        X_dropout = X_ori * np.random.binomial(1, 1 - delta, (n, p)) * 1/(1-delta)
        LR_dropout = LinearRegression().fit(X_dropout, y)
        y_hat_dropout = LR_dropout.predict(X_ori)
        dof_dropout = 1/sigma_e**2 * np.sum(e * (y_hat_dropout - y_))
        dof_dropout_list[i, j] = dof_dropout
dof_lr, dof_ls = dof_lr_list.mean(), dof_ls_list.mean()
dof_dropout = dof_dropout_list.mean(axis=0)
plt.plot(delta_list, dof_dropout, label='dof: Linear Regression w/ dropout')
plt.plot(delta_list[[0, 8]], [dof_lr]*2, ':', label='dof: Linear Regression')
plt.plot(delta_list[[0, 8]], [dof_ls]*2, ':', label='dof: Lasso')
plt.xlabel('$\delta$ of dropout')
plt.ylabel('degree of freedom')
plt.legend()
plt.savefig('./result/dof.png')
plt.show()
print(dof_lr, dof_ls, dof_dropout)



