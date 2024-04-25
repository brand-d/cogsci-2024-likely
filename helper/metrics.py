import numpy as np
import pandas as pd
import ccobra

response_list = [x.lower() for x in ccobra.syllogistic.RESPONSES]

def mfa_congruence(A, B):
    result = []
    for syl in ccobra.syllogistic.SYLLOGISMS:
        patterns_a = [x[0] for x in A[syl]]
        patterns_b = [x[0] for x in B[syl]]
        
        pat_a_res = []
        for pat_a in patterns_a:
            pat_a_mat = np.zeros((9))
            pat_a_idx = [response_list.index(x) for x in pat_a]
            for a in pat_a_idx:
                pat_a_mat[a] = 1
            
            pat_b_res = []
            for pat_b in patterns_b:
                pat_b_mat = np.zeros((9))
                pat_b_idx = [response_list.index(x) for x in pat_b]
                for b in pat_b_idx:
                    pat_b_mat[b] = 1
                
                acc = np.mean(pat_a_mat == pat_b_mat)
                pat_b_res.append(acc)
            pat_a_res.append(np.mean(pat_b_res))
        result.append(np.mean(pat_a_res))
    return np.mean(result)

def rmse(A, B):
    return np.sqrt(mse(A, B))

def mse(A, B):
    mse = (np.square(A - B)).mean()
    return mse

def get_accuracy(elem):
    corr = elem["tp"] + elem["tn"]
    incorr = elem["fp"] + elem["fn"]
    return corr / (corr + incorr)

def get_precision(elem):
    pos = elem["tp"] + elem["fp"]
    if pos == 0:
        # https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
        if elem["fn"] == 0:
            return 1
        else:
            return 0
    return elem["tp"] / pos

def get_recall(elem):
    pos_inst = elem["tp"] + elem["fn"]
    if pos_inst == 0:
        if elem["fp"] == 0:
            return 1
        else:
            return 0
    return elem["tp"] / pos_inst

def get_specificity(elem):
    neg_inst = elem["tn"] + elem["fp"]
    if neg_inst == 0:
        if elem["fn"] == 0:
            return 1
        else:
            return 0
    return elem["tn"] / neg_inst

def check_implicature(data, precondition, implication, threshold=0.5):
    if isinstance(data, pd.DataFrame):
        return _check_implicature_df(data, precondition, implication)
    else:
        return _check_implicature_mat(
            data, precondition, implication, threshold=threshold
        )

def _check_implicature_df(df, precondition, implication):
    occurances = 0
    total = 0
    for _, row in df.iterrows():
        responses = row["enc_responses"]
        responses = eval(responses)
        responses_quant = set([x[0] for x in responses if x != "nvc"])

        if precondition in responses_quant:
            total += 1
            if implication in responses_quant:
                occurances += 1
    return occurances/total

def _check_implicature_mat(mat, precondition, implication, threshold=0.5):
    quants_pre = [x for x in ccobra.syllogistic.RESPONSES 
                    if x[0].lower() == precondition.lower()]
    quant_idc_pre = [ccobra.syllogistic.RESPONSES.index(x) for x in quants_pre]

    quants_imp = [x for x in ccobra.syllogistic.RESPONSES 
                    if x[0].lower() == implication.lower()]
    quant_idc_imp = [ccobra.syllogistic.RESPONSES.index(x) for x in quants_imp]

    total = 0
    occurances = 0
    for p_idx in range(mat.shape[0]):
        p_mat = mat[p_idx]
        
        for syl_idx in range(64):
            resps = p_mat[syl_idx]
            
            pre_in_resp = False
            for pre_idx in quant_idc_pre:
                if resps[pre_idx] > threshold:
                    pre_in_resp = True
            if pre_in_resp:
                total += 1
                imp_in_resp = False
                for imp_idx in quant_idc_imp:
                    if resps[imp_idx] > threshold:
                        imp_in_resp = True
                if imp_in_resp:
                    occurances += 1
    return occurances/total