from scipy.sparse.linalg import svds
import numpy as np
import extract_input
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import interactive


def final_rmse(dictx, result, n_minkb):
    # for i in dictx.shape
    sum = float(0)
    count = float(0)
    n_mis = 0
    for row in dictx.index.values:
        for col in dictx.columns.values:
            if (abs(dictx.at[row, col] - abs(n_minkb)) > 0.00001):
                sum += math.pow((dictx.at[row, col] - result.at[row, col]), 2)
                count += 1
                pass
            else:
                n_mis += 1
                pass
            pass
    res = sum / count
    res = np.sqrt(res)
    return res, n_mis


class Recommendation:
    def __init__(self, n_features):
        (self.dict_kb, self.dict_pa, self.dict_rmse, self.n_minkb, self.panda_dataFeatures) = extract_input.get_data(
            n_features)
        pass

    def gen_kb_svd(self, K=10, step=20, n_features=20):
        R = (self.dict_kb + self.n_minkb).as_matrix()
        user_ratings_mean = np.mean(R, axis=1)
        R_demeaned = R - user_ratings_mean.reshape(-1, 1)
        if (K > min(R_demeaned.shape)):
            K = min(R_demeaned.shape) - 1
        U, sigma, Vt = svds(R_demeaned, K)
        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        result = pd.DataFrame(all_user_predicted_ratings, columns=self.dict_kb.columns)
        pd_rate = pd.DataFrame()
        cols = self.dict_kb.columns.values
        rows = self.dict_kb.index.values
        machines = extract_input.get_machineNames()
        for col in cols:
            ratings = []
            res = result.sort_values(col, ascending=False)
            for item in res[cols[0]][:5].index.values:
                ratings.append((item, machines[item]))
            pass
            pd_rate[col] = ratings
            pass
        pd_rate = pd_rate.T

        fn = r'data/result_svd_kb.xlsx'
        writer = pd.ExcelWriter(fn)
        final_rmse1, n_mis = final_rmse(self.dict_kb, result - self.n_minkb, self.n_minkb)
        print(final_rmse1, n_mis)
        pd_rms1 = pd.DataFrame({"rmse:": ["RMSE", final_rmse1]}, index=[0, 1])  # {"RMSE":final_rmse1})
        pd_rms1.to_excel(writer, "Sheet1", startcol=1, startrow=len(rows) + 1, header=False, index=False)
        result.to_excel(writer)
        pd_rate.to_excel("data/result_svd_kb_recommend.xlsx")
        writer.save()
        return (final_rmse1, result)
        # print(R_demeaned)
        pass

    def gen_pa_svd(self, K=10, step=20, n_features=20):
        R = (self.dict_pa).as_matrix()
        user_ratings_mean = np.mean(R, axis=1)
        R_demeaned = R - user_ratings_mean.reshape(-1, 1)
        if (K > min(R_demeaned.shape)):
            K = min(R_demeaned.shape) - 1
        U, sigma, Vt = svds(R_demeaned, K)
        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        result = pd.DataFrame(all_user_predicted_ratings, columns=self.dict_pa.columns)
        pd_rate = pd.DataFrame()
        cols = self.dict_pa.columns.values
        rows = self.dict_pa.index.values
        machines = extract_input.get_machineNames()
        for col in cols:
            ratings = []
            res = result.sort_values(col, ascending=False)
            for item in res[cols[0]][:5].index.values:
                ratings.append((item, machines[item]))
            pass
            pd_rate[col] = ratings
            pass
        pd_rate = pd_rate.T

        fn = r'data/result_svd_pa.xlsx'
        writer = pd.ExcelWriter(fn)
        final_rmse1, n_mis = final_rmse(self.dict_pa, result, 0)
        pd_rms1 = pd.DataFrame({"rmse:": ["RMSE", final_rmse1]}, index=[0, 1])  # {"RMSE":final_rmse1})
        pd_rms1.to_excel(writer, "Sheet1", startcol=1, startrow=len(rows) + 1, header=False, index=False)
        result.to_excel(writer)
        pd_rate.to_excel("data/result_svd_pa_recommend.xlsx")
        writer.save()
        return (final_rmse1, result)
        pass

    def gen_rmse_svd(self, K=10, step=20, n_features=20):
        R = (self.dict_rmse).as_matrix()
        user_ratings_mean = np.mean(R, axis=1)
        R_demeaned = R - user_ratings_mean.reshape(-1, 1)
        if (K > min(R_demeaned.shape)):
            K = min(R_demeaned.shape) - 1
        U, sigma, Vt = svds(R_demeaned, K)
        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        result = pd.DataFrame(all_user_predicted_ratings, columns=self.dict_rmse.columns)
        pd_rate = pd.DataFrame()
        cols = self.dict_rmse.columns.values
        rows = self.dict_rmse.index.values
        machines = extract_input.get_machineNames()
        for col in cols:
            ratings = []
            res = result.sort_values(col, ascending=False)
            for item in res[cols[0]][:5].index.values:
                ratings.append((item, machines[item]))
            pass
            pd_rate[col] = ratings
            pass
        pd_rate = pd_rate.T

        fn = r'data/result_svd_rmse.xlsx'
        writer = pd.ExcelWriter(fn)
        final_rmse1, n_mis = final_rmse(self.dict_rmse, result, 0)
        pd_rms1 = pd.DataFrame({"rmse:": ["RMSE", final_rmse1]}, index=[0, 1])  # {"RMSE":final_rmse1})
        pd_rms1.to_excel(writer, "Sheet1", startcol=1, startrow=len(rows) + 1, header=False, index=False)
        result.to_excel(writer)
        pd_rate.to_excel("data/result_svd_rmse_recommend.xlsx")
        writer.save()
        return (final_rmse1, result)
        pass


test = Recommendation(20)

(final_rmseKb, result_kb) = test.gen_kb_svd(K=10, step=20, n_features=20)
fig1, ax1 = plt.subplots()
ax1 = sns.heatmap(result_kb, ax=ax1, linewidth=0.5)
fig1.suptitle('This is a Kb HeatMap', fontsize=16)

(final_rmsePa, result_pa) = test.gen_pa_svd(K=10, step=20, n_features=20)
fig2, ax2 = plt.subplots()
ax2 = sns.heatmap(result_pa, ax=ax2, linewidth=0.5)
fig2.suptitle('This is a Pa HeatMap', fontsize=16)

(final_rmseRme, result_rmse) = test.gen_rmse_svd(K=10, step=20, n_features=20)
fig3, ax3 = plt.subplots()
ax3 = sns.heatmap(result_rmse, ax=ax3, linewidth=0.5)
fig3.suptitle('This is a Rmse HeatMap', fontsize=16)
plt.show()
