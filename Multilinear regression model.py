# -*- coding: utf-8 -*- 

import pandas as pd
import numpy as np
from scipy.special import comb
from itertools import combinations
from sklearn.metrics import r2_score
from sklearn import linear_model
import os
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import preprocessing
import time
import warnings
warnings.filterwarnings("ignore")
#Convert DataFrame to data with lag and lead format
def buildLagLeadFeatures(s, lag, lead, dropna=True):
    # Builds a new DataFrame to facilitate regressing over all possible lagged features
    if type(s) is pd.DataFrame:
        new_dict = {}
        transfer_dict = {}
        for col_name in s:
            new_dict[col_name] = s[col_name]
            # create lagged Series
            if (lag > 0):
                for l in range(1, lag + 1):
                    new_dict['%s_lag%d' % (col_name, l)] = s[col_name].shift(l)
                    transfer_dict['%s_lag%d' % (col_name, l)] = col_name
            if (lead > 0):
                for j in range(1, lead + 1):
                    new_dict['%s_lead%d' % (col_name, j)] = s[col_name].shift(-j)
                    transfer_dict['%s_lead%d' % (col_name, j)] = col_name
        res = pd.DataFrame(new_dict, index=s.index)
    else:
        print('Only works for DataFrame')
        return None
    if dropna:
        return res.dropna(), transfer_dict
    else:
        return res, transfer_dict
# Perform Transformation, Lag, and Lead on the original data
def buildFeatures(macro_df, lag, lead, dropna=True):
    macro_temp = macro_df
    macro_temp1 = macro_temp
    ind_pd_raw, varsign_dict = buildLagLeadFeatures(macro_temp1, lag, lead, dropna)
    return (ind_pd_raw, varsign_dict)
# Univariate linear regression, return coef and R-square for further selection 
def univariateLRTest(dep_t, ind_raw):
    ind_t = np.array(ind_raw).reshape(-1, 1)
    LR = linear_model.LinearRegression()
    LR.fit(ind_t, dep_t)
    LR_coef = LR.coef_[0]
    LR_pred = LR.predict(ind_t)
    LR_R2 = r2_score(dep_t, LR_pred)
    return (LR_coef, LR_R2)
def univariateTest(model, dep_t, ind_raw):
    if model == 'LR':
        return (univariateLRTest(dep_t, ind_raw))
    else:
        return ('Only LR supported for now')
# Iterate through univariate variables, output coef and r2, and associate with Varsign based on variable name
def BruteForceUniCal(model_name, dep_var_value, ind_candidate_df, varsign_dict):
    candidate_list = ind_candidate_df.columns
    ind_var_name_list = []
    coef_list_total = []
    r2_list_total = []
    Ori_Code = []
    for i in candidate_list:
        if i in varsign_dict.keys():
            Ori_Code.append(varsign_dict[i])
        else:
            Ori_Code.append(i)
        ind_var_name_list.append(i)
        LR_coef_temp, LR_R2 = univariateTest(model_name, dep_var_value, ind_candidate_df[i])
        coef_list_total.append(LR_coef_temp)
        r2_list_total.append(LR_R2)
    result_df = pd.DataFrame(
        {'Var_Code': ind_var_name_list, 'Ori_Code': Ori_Code, 'coef': coef_list_total, 'r2': r2_list_total})
    return (result_df)
def vifTest(ind_Candidate_df):
    X = sm.add_constant(ind_Candidate_df)
    viftest_list = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return (viftest_list)
# Multivariate linear regression
def multivariateLRTest(dep_t, ind_raw):
    vif = vifTest(ind_raw) # VIF Test
    X = sm.add_constant(ind_raw) # Linear Regression
    model = sm.OLS(dep_t, X)
    LR = model.fit()
    # params
    LR_coef = list(LR.params)
    LR_bse = list(LR.bse)
    LR_tvalues = list(LR.tvalues)
    LR_R2 = LR.rsquared
    LR_R2_adj = LR.rsquared_adj
    LR_PValue = list(LR.pvalues)
    LR_AIC = LR.aic
    LR_BIC = LR.bic
    LR_f_pvalue = LR.f_pvalue
    return (vif, LR_coef, LR_bse, LR_tvalues, LR_R2, LR_R2_adj, LR_PValue, LR_AIC, LR_BIC, LR_f_pvalue)
def multivariateTest(model, dep_t, ind_raw):
    if model == 'LR':
        return (multivariateLRTest(dep_t, ind_raw))
    else:
        return ('Only LR supported for now')
def BruteForceMultiCal(model_name,
                       dep_var_value,
                       ind_candidate_df,
                       comb_num,
                       total_uniTest_result_10):
    candidate_list = ind_candidate_df.columns
    num_candidate = len(candidate_list)
    possible_comb_num = comb(num_candidate, comb_num)
    possible_comb = list(combinations(ind_candidate_df.columns, comb_num))     # get all combinations
    ind_name_total = []
    vif_total = []
    coef_total = []
    bse_total = []
    tvalues_total = []
    r2_total = []
    r2adj_total = []
    p_total = []
    aic_total = []
    bic_total = []
    f_pvalue_total = []
    sign_total = []
    type_total = []
    model_total = []
    model_no = 1
    start_date = "Modeling Sample Start Time："+str(dep_var_value.index[0])[0:10]
    for j in tqdm(possible_comb, ascii=True,desc=start_date):
        i = list(j)
        ind_selected = ind_candidate_df[i]
        ind_name_temp = ['Intercept'] + i
        ind_name_temp_df = pd.DataFrame(ind_name_temp, columns=['Var_Code'])
        ind_name_temp_df = pd.merge(ind_name_temp_df, total_uniTest_result_10, how='left', on='Var_Code')
        sign_df = ind_name_temp_df[['Var_Code', 'sign', 'Type']]
        type_select = sign_df['Type'].duplicated().any()
        if type_select == 0:
            sign_df = sign_df.fillna(0)
            sign_temp = sign_df['sign'].tolist()
            type_temp = sign_df['Type'].tolist()
            vif_temp, coef_temp, bse_temp, tvalues_temp, r2_temp, r2adj_temp, p_temp, aic_temp, bic_temp, f_pvalue_temp = \
                multivariateTest(model_name, dep_var_value,ind_selected)
            sign_df['coef'] = coef_temp
            sign_df['coef_sign'] = sign_df['coef'] * sign_df['sign']
            sign_selection = (sign_df['coef_sign'] >= 0).all()
            vif_selection = (np.array(vif_temp[1:]) <= 5).all()
            p_selection = (np.array(p_temp[1:]) <= 0.1).all()
            if VIF_filter in ["no", "n"]:
                selection_filter = r2_temp >= 0.5 and sign_selection == 1 and p_selection == 1
            else:
                selection_filter = r2_temp >= 0.5 and vif_selection == 1 and sign_selection == 1 and p_selection == 1
            if selection_filter == 1:
                ind_name_total.extend(ind_name_temp)
                sign_total.extend(sign_temp)
                type_total.extend(type_temp)
                vif_total.extend(vif_temp)
                coef_total.extend(coef_temp)
                bse_total.extend(bse_temp)
                tvalues_total.extend(tvalues_temp)
                p_total.extend(p_temp)
                r2_total.extend([r2_temp] * (comb_num + 1))
                r2adj_total.extend([r2adj_temp] * (comb_num + 1))
                aic_total.extend([aic_temp] * (comb_num + 1))
                bic_total.extend([bic_temp] * (comb_num + 1))
                f_pvalue_total.extend([f_pvalue_temp] * (comb_num + 1))
                model_total.extend([model_no] * (comb_num + 1))
                model_no = model_no + 1
            else:
                pass
        else:
            pass
    result_df = pd.DataFrame({'Model_no': model_total,
                              'Variable': ind_name_total,
                              'coef': coef_total,
                              'Standard_error': bse_total,
                              't': tvalues_total,
                              'P>|t|': p_total,
                              'VIF': vif_total,
                              'Prob_(F-statistic)': f_pvalue_total,
                              'R-Squared': r2_total,
                              'Adjusted_R-squared': r2adj_total,
                              'AIC': aic_total,
                              'BIC': bic_total,
                              'sign': sign_total,
                              'Type': type_total})

    return (result_df)

def main_modelling(regressor_table):
    ind_candidate_total = regressor_table[ind_pd_raw.columns.tolist()]
    ind_candidate_total = ind_candidate_total.fillna(method='ffill')
    if factor_transform in ["A", "a"]:
        scaler = preprocessing.StandardScaler().fit(np.array(ind_candidate_total))
        X_scaled = scaler.transform(np.array(ind_candidate_total))
        ind_candidate_total = pd.DataFrame(X_scaled, columns=ind_candidate_total.columns, index=ind_candidate_total.index)
    elif factor_transform in ["B", "b"]:
        ind_candidate_total = ind_candidate_total.diff(1)
    else:
        pass
    # choose dependent variable
    dep = regressor_table['Zscore']
    uniTest_10 = BruteForceUniCal('LR', dep, ind_candidate_total, varsign_dict=varsign_dict)
    total_uniTest_result_10 = pd.merge(uniTest_10, transformation, left_on='Ori_Code', right_on='Code')

    if Algorithm_type in ["B", "b", "2"]:
        filter_uniTest_r2_10 = total_uniTest_result_10[total_uniTest_result_10['r2'] > 0.05]
        filter_uniTest_10 = filter_uniTest_r2_10[filter_uniTest_r2_10['coef'] * filter_uniTest_r2_10['sign'] > 0]
        uniTest_selection_10 = filter_uniTest_10.groupby('Type').apply(lambda t: t[t.r2 == t.r2.max()])
        uniTest_candidate_var_10 = list(uniTest_selection_10['Var_Code'])
        # Get the dataframe of independent variables for multivariate regression
        multi_candidate_10 = ind_candidate_total[uniTest_candidate_var_10]
        multiTest_10 = BruteForceMultiCal(model_name='LR',
                                          dep_var_value=dep,
                                          ind_candidate_df=multi_candidate_10,
                                          comb_num=factor_number,
                                          total_uniTest_result_10=total_uniTest_result_10)
    else:
        multiTest_10 = BruteForceMultiCal(model_name='LR',
                                          dep_var_value=dep,
                                          ind_candidate_df=ind_candidate_total,
                                          comb_num=factor_number,
                                          total_uniTest_result_10=total_uniTest_result_10)

    return multiTest_10

transformation = pd.read_excel("C:/Users/Yinong Dong/Desktop/input data.xlsx", sheet_name="Code")
macro_List = pd.read_excel("C:/Users/Yinong Dong/Desktop/input data.xlsx", sheet_name="History_Data", index_col=0)
dep_z = pd.read_excel("C:/Users/Yinong Dong/Desktop/input data.xlsx", sheet_name="NonRisk_Z3", index_col=0)
print("Multivariate Linear Modeling System")
lag = int(input("\nEnter lag value: "))
lead = int(input("\nEnter lead value: "))
factor_number = int(input("\nEnter the number of factors: "))
factor_transform = input("\nSelect macro factor processing category:  \n A：Perform standardization. \n B：Perform first-order differencing. \n C：No processing. Default is no processing：").lower()
Algorithm_type = input("\nSelect model algorithm type: \n A：Iterate through all multivariate models (takes longer time, better performance)\n B: Perform univariate variable selection first, then iterate through multivariate models (takes shorter time) \n Default is A :")
sampling_type = input("\nSelect sampling type，\n A：Full sample. \n B：Incremental sampling starting from a specific date (maintain at least 8 data points, longer time)，\n Default is A：").lower()
VIF_filter = input("\nApply VIF filter to select variables with low multicollinearity:\nFilter criterion: VIF less than 5，Y/N，Default is filter：").lower()
model_type = input("\nEnter model type，\n A：ASRF Model, CreditMetrics. \n B：Logistic Model，\n Default is A：").lower()

start = time.process_time()

if model_type in ["merton", "a"]:
    pass
elif model_type in ["logistic", "b"]:
    transformation['sign'] = -transformation['sign']

factor_no = str(factor_number)+"factor model results"

if Algorithm_type in ["B", "b", "2"]:
    Algorithm_str = "Univariate variable selection and iteration"
else:
    Algorithm_str = "Iterate through variables"

if sampling_type in ["B", "b", "2"]:
    sampling_str = "Incremental sampling based on date"
else:
    sampling_str = "Full sample"

file_name = sampling_str+"-"+Algorithm_str+"-"+factor_no

ind_pd_raw, varsign_dict = buildFeatures(macro_List, lag=lag, lead=lead, dropna=True)

# Filter the independent variable data based on the availability of data for the dependent variable, and obtain the overall table
regressor_table = dep_z.join(ind_pd_raw, how='inner')

if sampling_type in ["B", "b", "2"]:
    nrow = np.shape(regressor_table)[0]
    writer = pd.ExcelWriter('C:/Users/Yinong Dong/Desktop/output.xlsx')
    for n in range(nrow-7):
        regressor_table_sub = regressor_table.iloc[n:, :]
        sampling_start = str(regressor_table_sub.index[0])[0:10]
        outcome = main_modelling(regressor_table_sub)
        outcome.to_excel(excel_writer=writer, sheet_name=sampling_start, index=False)
    writer.save()
    writer.close()

else:
    outcome = main_modelling(regressor_table)
    outcome.to_excel('C:/Users/Yinong Dong/Desktop/3output.xlsx', index=False)


