import marimo

__generated_with = "0.4.4"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    mo.md(f"""
    #Necessity, Possibility and Likelihood in Syllogistic Reasoning
    This notebook contains the analysis for the the CogSci 2024 paper 'Necessity, Possibility and Likelihood in Syllogistic Reasoning'.  
    Most functionality is directly in the notebook, however, some additional helper files are used for the sake of clarity. They can be found in the *helper*-subfolder.
    """)
    return


@app.cell
def __(mo):
    import pandas as pd
    import numpy as np
    import ccobra

    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.patches import Ellipse, Circle
    import seaborn as sns

    from scipy.stats import mannwhitneyu
    from helper import syl_solver as syls
    from helper import metrics as metrics
    mo.md(f"""First, load the required dependencies.""")
    return (
        Circle,
        Ellipse,
        Rectangle,
        ccobra,
        mannwhitneyu,
        metrics,
        np,
        pd,
        plt,
        sns,
        syls,
    )


@app.cell
def __(mo, pd):
    df = pd.read_csv("data/brand2024_complete.csv")

    possible = df[df["type"] == "possible"]
    likely = df[df["type"] == "likely"]
    necessary = df[df["type"] == "necessary"]

    evans_necessary_df = pd.read_csv("data/evans_necessary.csv", sep=";")
    evans_possible_df = pd.read_csv("data/evans_possible.csv", sep=";")

    mo.md(f"""All datasets required for the analysis are loaded and the respective response types are extracted.""")
    return (
        df,
        evans_necessary_df,
        evans_possible_df,
        likely,
        necessary,
        possible,
    )


@app.cell
def __(likely, mo, necessary, np, possible):
    mo.md(f"""
    ## Number of participants
    Necessary: {len(np.unique(necessary["id"]))}  
    Likely: {len(np.unique(likely["id"]))}  
    Possible: {len(np.unique(possible["id"]))}
    """)
    return


@app.cell
def __(likely, mo, np, possible):
    percentage_some_entails_all_likely = np.round(np.mean(likely.groupby("id").agg("first")["some_entails_all"].values) * 100, 2)
    percentage_some_entails_all_possible = np.round(np.mean(possible.groupby("id").agg("first")["some_entails_all"].values) * 100, 2)

    mo.md(f"""
    ## Does *Some* also include *All*?
    For *Likely* and *Possible*, participants were asked if *Some* can also mean *All*:  
    For *Likely*, this was stated by {percentage_some_entails_all_likely}%.  
    For *Possible*, this was stated by {percentage_some_entails_all_possible}%.
    """)
    return (
        percentage_some_entails_all_likely,
        percentage_some_entails_all_possible,
    )


@app.cell
def __(likely, mo, necessary, plot_comparison, possible):
    triple_comparison_ax = plot_comparison(necessary, likely, possible)

    mo.md(f"""
    ## Comparison between necessary, likely and possible (Figure 1)

    {mo.as_html(triple_comparison_ax)}  

    Response distributions for concluding necessary, likely and possible for all 64 syllogisms and 9 response options
    in our data. Darker shades of blue denote a higher proportion of the respective response option. Red circles denote the most
    frequently selected combination of response options (column-wise), yellow circles are used in case of a tie for the alternative.
    """)
    return triple_comparison_ax,


@app.cell
def __(
    ccobra,
    evans_necessary_df,
    evans_possible_df,
    mo,
    np,
    plot_matrices_comparison,
):
    def get_evans_mat(df):
        ordering = ["Aac", "Iac", "Eac", "Oac", "Aca", "Ica", "Eca", "Oca"]
        result = np.zeros((64,9))
        for syl_idx, syl in enumerate(ccobra.syllogistic.SYLLOGISMS):
            col = df[syl].values
            for idx, resp in enumerate(ordering):
                val = col[idx]
                resp_idx = ccobra.syllogistic.RESPONSES.index(resp)
                result[syl_idx, resp_idx] = val / 100
        return result

    # Create matrices from evans data
    evans_necessary_mat = get_evans_mat(evans_necessary_df)
    evans_possible_mat = get_evans_mat(evans_possible_df)

    evans_pattern_ax = plot_matrices_comparison(
        evans_necessary_mat, evans_possible_mat,
        "Necessary", "Possible")

    mo.md(f"""
    ## Patterns of the Evans dataset (Evans et al., 1999)

    {mo.as_html(evans_pattern_ax)}  

    Since the data was only available in an aggregated form, it was not possible to derive any NVC responses: NVC can only be derived if a participant does not select any conclusion, which is an information not present in the data due to the aggregation (and thereby the loss of individual information).

    """)
    return (
        evans_necessary_mat,
        evans_pattern_ax,
        evans_possible_mat,
        get_evans_mat,
    )


@app.cell
def __(likely, mo, plot_dual_comparison, possible):
    crt_likely_comparison_ax = plot_dual_comparison(
        likely[likely["crt"] >= likely["crt"].median()],
        likely[likely["crt"] < likely["crt"].median()],
        "High CRT",
        "Low CRT")
    crt_possible_comparison_ax = plot_dual_comparison(
        possible[possible["crt"] >= possible["crt"].median()],
        possible[possible["crt"] < possible["crt"].median()],
        "High CRT",
        "Low CRT")

    mo.md(f"""
    ## Comparison between high and low CRT scores
    Comparison of the patterns for participants with an higher-than-median score in the cognitive reflection task compared to the participants with a lower-than-median score.  

    ###Likely
    {mo.as_html(crt_likely_comparison_ax)}

    ###Possible
    {mo.as_html(crt_possible_comparison_ax)}
    """)
    return crt_likely_comparison_ax, crt_possible_comparison_ax


@app.cell
def __(
    ccobra,
    metrics,
    mo,
    necessary,
    np,
    plot_matrices_comparison,
    possible,
    response_list,
):
    # Generates the pattern matrix for a single participant
    def get_single_person_pattern(df, weighted=True):
        mat = np.zeros((64,9))

        for _, row in df.iterrows():
            responses = row["enc_responses"]
            responses = eval(responses)

            syl = row["enc_task"]
            syl_idx = ccobra.syllogistic.SYLLOGISMS.index(syl)

            for resp in responses:
                resp_idx = response_list.index(resp)
                score = 1
                if weighted:
                    score = 1 / len(responses)
                mat[syl_idx, resp_idx] += score
        return mat

    # Selects the pattern from the mReasoner cache file that resemble the participants the most (in terms of RMSE)
    def fit_to_participants(cache, df):
        by_person = df.groupby("id")
        num_persons = len(np.unique(df["id"]))
        result = np.zeros((64, 9))
        full_fits = np.zeros((num_persons, 64, 9))
        cur_p = 0
        rmses = []
        for p_id, person_df in by_person:
            person_mat = get_single_person_pattern(person_df, weighted=False)

            best_rmse = 1000
            best_mat = None

            for idx_epsilon in range(cache.shape[0]):
                for idx_lambda in range(cache.shape[0]):
                    for idx_omega in range(cache.shape[0]):
                        for idx_sigma in range(cache.shape[0]):
                            mat = cache[idx_epsilon, idx_lambda, 
                                        idx_omega, idx_sigma]
                            rmse = metrics.rmse(person_mat, mat)
                            if rmse < best_rmse:
                                best_rmse = rmse
                                best_mat = mat
            result += best_mat
            full_fits[cur_p] += best_mat
            cur_p += 1
            rmses.append(best_rmse)
        result /= num_persons
        return result, full_fits, rmses

    # Calculates the probability of NVC for a single task column
    def add_nvc(mat):
        for i in range(mat.shape[0]):
            prob = 1
            for j in range(mat.shape[1]):
                val = mat[i, j]
                prob *= (1- val)
            mat[i, 8] = prob

    # Extends the cached mreasoner files by adding NVC
    def add_nvc_to_cache(cache_mat):
        for idx_epsilon in range(cache_mat.shape[0]):
            for idx_lambda in range(cache_mat.shape[1]):
                for idx_omega in range(cache_mat.shape[2]):
                    for idx_sigma in range(cache_mat.shape[3]):
                        mat = cache_mat[idx_epsilon, idx_lambda, 
                                        idx_omega, idx_sigma]
                        add_nvc(mat)

    cache_necessary = np.load('mreasoner/necessary_full.npy')
    cache_possible = np.load('mreasoner/possible_full.npy')

    add_nvc_to_cache(cache_necessary)
    add_nvc_to_cache(cache_possible)

    necessary_pattern_mr, all_nec_patterns_mr, rmses_nec = fit_to_participants(cache_necessary, necessary)
    possible_pattern_mr, all_pos_patterns_mr, rmses_pos = fit_to_participants(cache_possible, possible)

    mReaonser_plot_ax = plot_matrices_comparison(
        necessary_pattern_mr, possible_pattern_mr, 
        "Necessary", "Possible", 
        offset1=0.23, offset2=0.28)

    mo.md(f"""
    ## mReasoner Patterns (Figure 2)
    Comparison of the patterns that mReasoner generates for *Necessary* and *Possible* when aggregating over the best fits to each participant. Thereby, the resulting artificial dataset is the dataset best resembling each individual participant (with respect to RMSE). NVC is retrospectively added by interpreting the selected conclusions as probabilities (NVC is then the probability that none of the others was selected).

    {mo.as_html(mReaonser_plot_ax)}  

    Thereby, the RMSEs of the fitting process were as follows:  
    *Necessary* patterns: {np.round(np.mean(rmses_nec), 2)}  
    *Possible* patterns: {np.round(np.mean(rmses_pos), 2)}\n
    Note that mReasoner's responses are retrieved from a numpy array (cache). The file contains the results of a grid-search querying mReaonser for each individual conclusion. To obtain the file, the [implementation](https://github.com/nriesterer/cogsci-individualization) provided with the 2020 paper "Do Models Capture Individuals? Evaluating Parameterized Models for Syllogistic Reasoning" by Riesterer, Brand & Ragni was used.
    """)
    return (
        add_nvc,
        add_nvc_to_cache,
        all_nec_patterns_mr,
        all_pos_patterns_mr,
        cache_necessary,
        cache_possible,
        fit_to_participants,
        get_single_person_pattern,
        mReaonser_plot_ax,
        necessary_pattern_mr,
        possible_pattern_mr,
        rmses_nec,
        rmses_pos,
    )


@app.cell
def __(ccobra, metrics, np):
    # Tries to find a combination of n simulated participant patterns (from mReasoner) to approximate a given target using random sampling
    def random_approximation(cache, target, its=1000, samples=30, ignore_nvc=True):
        best_rmse = 10000
        best_mat = None
        for it in range(its):
            # Select samples at random
            mat = np.zeros((samples, 64, 9))
            for sample in range(samples):
                p1 = np.random.choice(np.arange(cache.shape[0]))
                p2 = np.random.choice(np.arange(cache.shape[1]))
                p3 = np.random.choice(np.arange(cache.shape[2]))
                p4 = np.random.choice(np.arange(cache.shape[3]))
                mat[sample] = cache[p1, p2, p3, p4]
            mat = np.mean(mat, axis=0)
            # Compare success
            if ignore_nvc:
                rmse = metrics.rmse(target[:, :8], mat[:, :8])
            else:
                rmse = metrics.rmse(target, mat)
            if rmse < best_rmse:
                best_mat = mat
                best_rmse = rmse
        return best_mat

    # Creates an MFA dictionary for mReasoners fits
    def get_mfa_mreasoner(fits):
        mfp = {}

        for p_id in range(fits.shape[0]):
            person_fit = fits[p_id]
            for syl_idx in range(person_fit.shape[0]):
                resp_vec = person_fit[syl_idx]
                responses = [ccobra.syllogistic.RESPONSES[x].lower() 
                        for x in range(9) if resp_vec[x] >= 0.5]
                syl = ccobra.syllogistic.SYLLOGISMS[syl_idx]

                if syl not in mfp:
                    mfp[syl] = []

                if not responses:
                    responses = ['nvc']
                mfp[syl].append(sorted(responses))

        mfp_res = {}
        for key, value in mfp.items():
            value_tuples = [tuple(x) for x in value]
            val_array = np.empty(len(value_tuples), dtype=tuple)
            val_array[:] = value_tuples
            numbers = np.unique(val_array, return_counts=True)
            numbers = dict(zip(*np.unique(val_array, return_counts=True)))
            numbers = sorted(numbers.items(), key=lambda x: x[1], reverse=True)

            mfp_res[key] = [numbers[0]]
            if len(numbers) > 1 and numbers[0][1] == numbers[1][1]:
                mfp_res[key].append(numbers[1])

        return mfp_res
    return get_mfa_mreasoner, random_approximation


@app.cell
def __(
    all_nec_patterns_mr,
    all_pos_patterns_mr,
    cache_necessary,
    cache_possible,
    evans_necessary_mat,
    evans_possible_mat,
    get_mat,
    get_mfa_mreasoner,
    likely,
    metrics,
    mo,
    necessary,
    necessary_pattern_mr,
    np,
    pd,
    possible,
    possible_pattern_mr,
    random_approximation,
):
    # Obtain matrices and MFA dictionaries for necessary, possible and likely
    necessary_pattern, mfa_pattern_nec = get_mat(necessary)
    possible_pattern, mfa_pattern_pos = get_mat(possible)
    likely_pattern, mfa_pattern_lik = get_mat(likely)

    # Obtain MFA dictionaries for mReasoner fit data
    mr_mfa_necessary = get_mfa_mreasoner(all_nec_patterns_mr)
    mr_mfa_possible = get_mfa_mreasoner(all_pos_patterns_mr)

    # Obtain approximation for Evans data
    evans_mr_nec = random_approximation(cache_necessary, evans_necessary_mat)
    evans_mr_pos = random_approximation(cache_possible, evans_possible_mat)

    # Result table
    rmse_mfa_table = [
        {"Dataset 1": "Necessary", 
         "Dataset 2": "Likely", 
         "RMSE": np.round(
             metrics.rmse(necessary_pattern, likely_pattern), 3), 
         "MFA": np.round(
             metrics.mfa_congruence(mfa_pattern_nec, mfa_pattern_lik), 3)
        },
        {"Dataset 1": "Necessary", 
         "Dataset 2": "Possible", 
         "RMSE": np.round(
             metrics.rmse(necessary_pattern, possible_pattern), 3), 
         "MFA": np.round(
             metrics.mfa_congruence(mfa_pattern_nec, mfa_pattern_pos), 3)
        },
        {"Dataset 1": "Possible", 
         "Dataset 2": "Likely", 
         "RMSE": np.round(
             metrics.rmse(possible_pattern, likely_pattern), 3), 
         "MFA": np.round(
             metrics.mfa_congruence(mfa_pattern_pos, mfa_pattern_lik), 3)
        },
        {"Dataset 1": "Necessary (Evans)",
         "Dataset 2": "Possible (Evans)", 
         "RMSE": np.round(metrics.rmse(
             evans_necessary_mat[:,:8], evans_possible_mat[:,:8]), 3),
         "MFA": np.nan
        },
        {"Dataset 1": "Necessary (Evans)",
         "Dataset 2": "Necessary", 
         "RMSE": np.round(metrics.rmse(
             necessary_pattern[:,:8], evans_necessary_mat[:,:8]), 3),
         "MFA": np.nan},
        {"Dataset 1": "Possible (Evans)",
         "Dataset 2": "Possible", 
         "RMSE": np.round(metrics.rmse(
            possible_pattern[:,:8], evans_possible_mat[:,:8]), 3), 
         "MFA": np.nan
        },
        {"Dataset 1": "Necessary (mR)",
         "Dataset 2": "Possible (mR)", 
         "RMSE": np.round(
             metrics.rmse(necessary_pattern_mr, possible_pattern_mr), 3),
         "MFA": np.round(
             metrics.mfa_congruence(mfa_pattern_nec, mfa_pattern_pos), 3)
        },
        {"Dataset 1": "Necessary (mR)",
         "Dataset 2": "Necessary", 
         "RMSE": np.round(
             metrics.rmse(necessary_pattern_mr, necessary_pattern), 3),
         "MFA": np.round(
             metrics.mfa_congruence(mfa_pattern_nec, mr_mfa_necessary), 3)
        },
        {"Dataset 1": "Possible (mR)",
         "Dataset 2": "Possible", 
         "RMSE": np.round(
             metrics.rmse(possible_pattern_mr, possible_pattern), 3),
         "MFA": np.round(
             metrics.mfa_congruence(mfa_pattern_pos, mr_mfa_possible), 3)
        },
        {"Dataset 1": "Necessary (mR)",
         "Dataset 2": "Necessary (Evans)", 
         "RMSE": np.round(metrics.rmse(
             evans_mr_nec[:,:8], evans_necessary_mat[:,:8]), 3),
         "MFA": np.nan
        },
        {"Dataset 1": "Possible (mR)",
         "Dataset 2": "Possible (Evans)", 
         "RMSE": np.round(metrics.rmse(
             evans_mr_pos[:,:8], evans_possible_mat[:,:8]), 3),
         "MFA": np.nan
        },
    ]
    rmse_mfa_table = pd.DataFrame(rmse_mfa_table).round(3).to_dict('records')

    mo.md(f"""
    ## Quantitative Comparison of Patterns (Table 1)
    Patterns for *Likely*, *Possible*, *Necessary*, as well as the respective patterns (possible and necessary) obtained from the dataset by Evans et al. (1999) and mReasoner are compared with respect to RMSE and MFA congruency (congruency between the most frequent answers of the patterns; not available for the aggregated data by Evans et al.). Furthermore, for every comparison involving the data by evans, NVC is ignored (as it was not available in the data).

    {mo.ui.table(rmse_mfa_table, selection=None, pagination=False)}

    Note that the fitting process for the Evans dataset involves random sampling, so that the results can vary slightly. While we used 10000 iterations in our analysis, the iterations are now set to 1000 only to ensure a faster initialization of the notebook. The number of samples (n=30) is used to mimick the 30 participants for each condition in the original dataset.

    """)
    return (
        evans_mr_nec,
        evans_mr_pos,
        likely_pattern,
        mfa_pattern_lik,
        mfa_pattern_nec,
        mfa_pattern_pos,
        mr_mfa_necessary,
        mr_mfa_possible,
        necessary_pattern,
        possible_pattern,
        rmse_mfa_table,
    )


@app.cell
def __(
    all_nec_patterns_mr,
    all_pos_patterns_mr,
    ccobra,
    metrics,
    mo,
    necessary,
    np,
    pd,
    possible,
    syls,
):
    # Obtain logically correct responses
    logically_correct_responses = {}

    for syl in ccobra.syllogistic.SYLLOGISMS:
        logically_correct_responses[syl] = {
                "possible": syls.get_possible_responses(syl),
                "necessary": syls.get_valid_responses(syl),
        }

    # Returns the confusion data for a single task
    def confusion_per_task(task, resp, question="necessary"):
        p_tp = 0
        p_tn = 0
        p_fp = 0
        p_fn = 0
        cor_sol = logically_correct_responses[task][question]
        cor_sol = set([x.lower() for x in cor_sol])
        if "nvc" in cor_sol:
            cor_sol = set()
        if "nvc" in resp:
            resp = set()

        if not resp and not cor_sol:
            # NVC correctly predicted
            p_tp += 1
            p_tn += 7
            p_fp += 0
            p_fn += 0
        else:
            tp = len(cor_sol.intersection(resp))
            fp = len(resp - cor_sol)
            fn = len(cor_sol - resp)
            p_tp += tp
            p_fp += fp
            p_fn += fn
            p_tn += 8 - tp - fp - fn
        return p_tp, p_fp, p_fn, p_tn

    # Function to calculate the true/false positives/negatives per person and task
    def get_confusions(df, question="necessary"):
        df_grp = df.groupby("id")
        results = []
        for p_id, person_df in df_grp:
            for _, row in person_df.iterrows():
                p_tp, p_fp, p_fn, p_tn = confusion_per_task(
                    row["enc_task"], 
                    set(eval(row["enc_responses"])),
                    question=question)

                results.append({
                    "id": p_id,
                    "task": row["enc_task"],
                    "tp": p_tp,
                    "tn": p_tn,
                    "fp": p_fp,
                    "fn": p_fn,
                })
        result_df = pd.DataFrame(results)
        return result_df

    # Get confusion matrix data for mReasoner fitting results
    def get_mreasoner_confusions(fits, question="necessary"):
        results = []
        for p_id in range(fits.shape[0]):
            person_fit = fits[p_id]
            for syl_idx in range(person_fit.shape[0]):
                resp_vec = person_fit[syl_idx]
                resp = [ccobra.syllogistic.RESPONSES[x].lower() 
                        for x in range(9) if resp_vec[x] >= 0.5]
                task = ccobra.syllogistic.SYLLOGISMS[syl_idx]
                p_tp, p_fp, p_fn, p_tn = confusion_per_task(
                    task, 
                    set(resp),
                    question=question)

                results.append({
                    "id": p_id,
                    "task": task,
                    "tp": p_tp,
                    "tn": p_tn,
                    "fp": p_fp,
                    "fn": p_fn,
                })
        result_df = pd.DataFrame(results)
        return result_df

    # Enriches a table containing tp/tn/fp/fn with the accuracy, precision, recall and specificity
    def insert_correctness_metrics(df):
        df["accuracy"] = df[["tp", "tn", "fp", "fn"]].apply(
            metrics.get_accuracy, axis=1)
        df["precision"] = df[["tp", "tn", "fp", "fn"]].apply(
            metrics.get_precision, axis=1)
        df["recall"] = df[["tp", "tn", "fp", "fn"]].apply(
            metrics.get_recall, axis=1)
        df["specificity"] = df[["tp", "tn", "fp", "fn"]].apply(
            metrics.get_specificity, axis=1)
        return df

    # participants confusion data
    total_confusions_per_pers_necessary = get_confusions(necessary, "necessary").groupby("id")[["tp", "tn", "fp", "fn"]].agg("sum").reset_index()
    total_confusions_per_pers_possible = get_confusions(possible, "possible").groupby("id")[["tp", "tn", "fp", "fn"]].agg("sum").reset_index()

    # mReasoner confusion data
    mR_confusions_necessary = get_mreasoner_confusions(all_nec_patterns_mr, "necessary").groupby("id")[["tp", "tn", "fp", "fn"]].agg("sum").reset_index()
    mR_confusions_possible = get_mreasoner_confusions(all_pos_patterns_mr, "possible").groupby("id")[["tp", "tn", "fp", "fn"]].agg("sum").reset_index()

    # Get accuracy, precision, recall and specificity
    correctness_necessary = insert_correctness_metrics(
        total_confusions_per_pers_necessary)
    correctness_possible = insert_correctness_metrics(
        total_confusions_per_pers_possible)
    correctness_necessary_mR = insert_correctness_metrics(
        mR_confusions_necessary)
    correctness_possible_mR = insert_correctness_metrics(
        mR_confusions_possible)

    # Create combined dataframe for presentation
    necessary_mean_correctness = correctness_necessary[["tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "specificity"]].mean().to_dict()
    necessary_mean_correctness["type"] = "necessary"

    possible_mean_correctness = correctness_possible[["tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "specificity"]].mean().to_dict()
    possible_mean_correctness["type"] = "possible"

    mr_necessary_mean_correctness = correctness_necessary_mR[["tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "specificity"]].mean().to_dict()
    mr_necessary_mean_correctness["type"] = "necessary (mR)"

    mr_possible_mean_correctness = correctness_possible_mR[["tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "specificity"]].mean().to_dict()
    mr_possible_mean_correctness["type"] = "possible (mR)"

    total_mean_correctness = pd.DataFrame(
        [necessary_mean_correctness, possible_mean_correctness,
        mr_necessary_mean_correctness, mr_possible_mean_correctness]
    ).round(3)[["type", "tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "specificity"]].to_dict('records')

    # Number of logically correct conclusions for possible and necessary
    num_possible_responses = [len(logically_correct_responses[syl]["possible"]) 
                              for syl in ccobra.syllogistic.SYLLOGISMS]
    num_necessary_responses = [len(logically_correct_responses[syl]["necessary"]) 
                              for syl in ccobra.syllogistic.SYLLOGISMS]

    mo.md(f"""
    ## Correctness of Responses (Table 2)
    For possible and necessary, the logical correctness of given responses can be determined. For this work, a solver for syllogisms was used to obtain the logically correct conclusions (included in the helper subfolder).
    ###Number of correct conclusions
    Average number of correct conclusions for *Necessary*: {np.round(np.mean(num_necessary_responses), 2)}  
    Average number of correct conclusions for *Possible*: {np.round(np.mean(num_possible_responses), 2)}  
    ###Correctness of participants' responses
    Correctness is determined in terms of Accuracy, Precision, Recall and Specificity in order to provide a better insight into the response behavior.

    {mo.ui.table(total_mean_correctness, selection=None)}

    """)
    return (
        confusion_per_task,
        correctness_necessary,
        correctness_necessary_mR,
        correctness_possible,
        correctness_possible_mR,
        get_confusions,
        get_mreasoner_confusions,
        insert_correctness_metrics,
        logically_correct_responses,
        mR_confusions_necessary,
        mR_confusions_possible,
        mr_necessary_mean_correctness,
        mr_possible_mean_correctness,
        necessary_mean_correctness,
        num_necessary_responses,
        num_possible_responses,
        possible_mean_correctness,
        syl,
        total_confusions_per_pers_necessary,
        total_confusions_per_pers_possible,
        total_mean_correctness,
    )


@app.cell
def __(
    all_nec_patterns_mr,
    all_pos_patterns_mr,
    likely,
    metrics,
    mo,
    necessary,
    np,
    possible,
):
    table_occurance_quantifier = [
        {"Q1" : "A", "Q2": "I"},
        {"Q1" : "I", "Q2": "A"},
        {"Q1" : "E", "Q2": "O"},
        {"Q1" : "O", "Q2": "E"},
        {"Q1" : "I", "Q2": "O"},
        {"Q1" : "O", "Q2": "I"},
    ]

    table_occurance_quantifier_threshold = []

    for implicature in table_occurance_quantifier:
        q1 = implicature["Q1"].lower()
        q2 = implicature["Q2"].lower()
        implicature["Necessary"] = np.round(
            metrics.check_implicature(necessary, q1, q2),
            3
        )
        implicature["Likely"] = np.round(
            metrics.check_implicature(likely, q1, q2),
            3
        )
        implicature["Possible"] = np.round(
            metrics.check_implicature(possible, q1, q2),
            3
        )
        implicature_threshold = implicature.copy()
        
        implicature["Necessary (mR)"] = np.round(
            metrics.check_implicature(all_nec_patterns_mr, q1, q2, threshold=0),
            3
        )
        implicature["Possible (mR)"] = np.round(
            metrics.check_implicature(all_pos_patterns_mr, q1, q2, threshold=0),
            3
        )
        implicature_threshold["Necessary (mR)"] = np.round(
            metrics.check_implicature(all_nec_patterns_mr, q1, q2, threshold=0.5),
            3
        )
        implicature_threshold["Possible (mR)"] = np.round(
            metrics.check_implicature(all_pos_patterns_mr, q1, q2, threshold=0.5),
            3
        )
        table_occurance_quantifier_threshold.append(implicature_threshold)


    mo.md(f"""
    ## Common Occurances of Quantifiers (Table 3)
    Common occurrences of quantifiers for *Necessary*, *Likely* and *Possible* as well as for the predictions by mReasoner (mR). Values reflect the percentage of cases in which the second quantifier (Q2) is selected when the first quantifier (Q1) was selected as a response. Thereby, the information can be used to estimate which quantifiers are interpreted in a way that they imply another.  
    Please note that for mReasoner, it could be argued that a conclusion should only count as selected if it has a probability of >.5. However, in this analysis we were interested in the possibility of other quantifier co-occuring, thereby using a threshold of 0.  

    {mo.ui.table(table_occurance_quantifier, selection=None)}

    For completeness, the results with a threshold of 0.5 are shown in the table below:

    {mo.ui.table(table_occurance_quantifier_threshold, selection=None)}

    """)
    return (
        implicature,
        implicature_threshold,
        q1,
        q2,
        table_occurance_quantifier,
        table_occurance_quantifier_threshold,
    )


@app.cell
def __(
    ccobra,
    evans_necessary_mat,
    evans_possible_mat,
    likely_pattern,
    mannwhitneyu,
    mo,
    necessary_pattern,
    np,
    possible_pattern,
):
    # calculates the difference between direction occurance, the number of cases where the effect holds and the number of cases where it is violated
    def figure_diff(mat):
        enum_syls = list(enumerate(ccobra.syllogistic.SYLLOGISMS))
        enum_resps = list(enumerate(ccobra.syllogistic.RESPONSES))
        figure1_syllogs = [(x, y) for x, y in enum_syls if y[2] == "1"]
        figure1_responses = [(x, y) for x, y in enum_resps if y[1:] == "ac"]
        
        figure2_syllogs = [(x, y) for x, y in enum_syls if y[2] == "2"]
        figure2_responses = [(x, y) for x, y in enum_resps if y[1:] == "ca"]
        
        effect_votes = []
        contra_votes = []
        figure1_diffs = []
        for syl_idx, syllog in figure1_syllogs:
            fig1_votes = np.sum([mat[syl_idx, x] for x, y in figure1_responses])
            fig2_votes = np.sum([mat[syl_idx, x] for x, y in figure2_responses])
            figure1_diffs.append(fig1_votes - fig2_votes)
            effect_votes.append(fig1_votes)
            contra_votes.append(fig2_votes)

        figure2_diffs = []
        for syl_idx, syllog in figure2_syllogs:
            fig1_votes = np.sum([mat[syl_idx, x] for x, y in figure1_responses])
            fig2_votes = np.sum([mat[syl_idx, x] for x, y in figure2_responses])
            figure2_diffs.append(fig2_votes - fig1_votes)
            effect_votes.append(fig2_votes)
            contra_votes.append(fig1_votes)
            
        return figure1_diffs, figure2_diffs, effect_votes, contra_votes

    # Returns an analysis of the figure effect for a given dataset
    def analyze_figure_effect(mat, name):
        figure1_diffs, figure2_diffs, effect_votes, contra_votes = figure_diff(mat)
        total_diff = figure1_diffs + figure2_diffs
        U, p = mannwhitneyu(effect_votes, contra_votes, method="exact")
        result = {
            "Dataset": name,
            "Fig": np.round(np.mean(effect_votes), 3),
            "Not Fig": np.round(np.mean(contra_votes), 3),
            "Total Diff.": np.round(np.mean(total_diff), 3),
            "Fig1 Diff.": np.round(np.mean(figure1_diffs), 3),
            "Fig2 Diff.": np.round(np.mean(figure2_diffs), 3),
            "U": U,
            "p": np.round(p, 3)
        }
        return result

    figural_table = [
        analyze_figure_effect(necessary_pattern, "Necessary"),
        analyze_figure_effect(likely_pattern, "Likely"),
        analyze_figure_effect(possible_pattern, "Possible"),
        analyze_figure_effect(evans_necessary_mat, "Necessary (Evans)"),
        analyze_figure_effect(evans_possible_mat, "Possible (Evans)"),
    ]

    mo.md(f"""
    ## Figural Effect (Table 4)
    The Figural Effect refers to an effect with respect to the direction of the conclusion. Figure 1 syllogisms show an increased number of conclusions in the *ac* direction, while Figure 2 is more likely to have more *ca* conclusions.
    To quantify the effect, we compared the number of responses in line with the figural effect with the number of responses contradicting the figural effect for all syllogisms with figure 1 or figure 2 (None-responses are ignored).

    {mo.ui.table(figural_table, selection=None)}

    """)
    return analyze_figure_effect, figural_table, figure_diff


@app.cell
def __(Ellipse, ccobra, np, plt, sns):
    # Helper definitions and functions for plotting
    response_list = [x.lower() for x in ccobra.syllogistic.RESPONSES]

    def get_mat(df, weighted=False):
        mat = np.zeros((64,9))
        num_persons = len(np.unique(df["id"]))

        mfp = {}

        num_responses_without_nvc = []
        num_responses = []

        for _, row in df.iterrows():
            responses = row["enc_responses"]

            responses = eval(responses)
            num_responses.append(len(responses))
            if "nvc" not in responses:
                num_responses_without_nvc.append(len(responses))

            syl = row["enc_task"]
            syl_idx = ccobra.syllogistic.SYLLOGISMS.index(syl)

            if syl not in mfp:
                mfp[syl] = []
            mfp[syl].append(sorted(responses))

            for resp in responses:
                resp_idx = response_list.index(resp)
                score = 1
                if weighted:
                    score = 1 / len(responses)
                mat[syl_idx, resp_idx] += score / num_persons

        mfp_res = {}
        for key, value in mfp.items():
            value_tuples = [tuple(x) for x in value]
            val_array = np.empty(len(value_tuples), dtype=tuple)
            val_array[:] = value_tuples
            numbers = np.unique(val_array, return_counts=True)
            numbers = dict(zip(*np.unique(val_array, return_counts=True)))
            numbers = sorted(numbers.items(), key=lambda x: x[1], reverse=True)

            mfp_res[key] = [numbers[0]]
            if len(numbers) > 1 and numbers[0][1] == numbers[1][1]:
                mfp_res[key].append(numbers[1])

        return mat, mfp_res

    def plot_pattern(ax, df, mat=None, weighted=False, show_most=True):
        mat, mfp_dict = get_mat(df, weighted=weighted)
        mat = mat.T

        sns.heatmap(mat, ax=ax, cmap="Blues", cbar=False, 
                    vmin=0, linewidths=0.5, linecolor='#00000022')
        renamed_responses = ccobra.syllogistic.RESPONSES.copy()
        renamed_responses[-1] = "None"
        ax.set_yticks(np.arange(len(renamed_responses)) + 0.5)
        ax.set_yticklabels(renamed_responses, rotation=0)
        ax.set_xticks(np.arange(len(ccobra.syllogistic.SYLLOGISMS), step=4) + 0.7)
        ax.set_xticklabels(ccobra.syllogistic.SYLLOGISMS[::4], rotation=90)
        ax.tick_params(axis='x', pad=-4)

        if show_most:
            colors = ["red", "yellow"]
            dot_sizes = [[0.45, 0.36], [0.33, 0.25]]
            for syl_idx, syl in enumerate(ccobra.syllogistic.SYLLOGISMS):
                mfps = mfp_dict[syl]
                for color_idx, mfp in enumerate(mfps):
                    responses = mfp[0]
                    if type(responses) is str:
                        responses = [responses]
                    for response in responses:
                        resp_idx = response_list.index(response)
                        ax.add_patch(Ellipse((syl_idx + 0.5, resp_idx + 0.5),
                                             dot_sizes[color_idx][0],
                                             dot_sizes[color_idx][1], 
                                             fill=True, 
                                             facecolor=colors[color_idx],
                                             edgecolor=colors[color_idx], lw=0.3))

    def plot_comparison(necessary_df, likely_df, possible_df, show_most=True):
        sns.set(style='whitegrid', palette='colorblind')
        fig, axs = plt.subplots(3, 1, figsize=(11, 5.5))

        plot_pattern(axs[0], necessary_df, weighted=False, show_most=show_most)
        plot_pattern(axs[1], likely_df, weighted=False, show_most=show_most)
        plot_pattern(axs[2], possible_df, weighted=False, show_most=show_most)
        axs[0].set_xticklabels([])
        axs[1].set_xticklabels([])

        axs[0].set_title("Necessary", rotation=90, x=-0.056, y=0.25)
        axs[1].set_title("Likely", rotation=90, x=-0.056, y=0.35)
        axs[2].set_title("Possible", rotation=90, x=-0.056, y=0.3)

        plt.subplots_adjust(hspace=0)
        plt.tight_layout(pad=1)
        return plt.gca()

    def plot_dual_comparison(pattern1, pattern2, p1_name, p2_name, show_most=True, y=0.2):
        sns.set(style='whitegrid', palette='colorblind')
        fig, axs = plt.subplots(2, 1, figsize=(11, 4), sharex=True)

        plot_pattern(axs[0], pattern1, weighted=False, show_most=show_most)
        plot_pattern(axs[1], pattern2, weighted=False, show_most=show_most)

        axs[0].set_title(p1_name, rotation=90, x=-0.056, y=y)
        axs[1].set_title(p2_name, rotation=90, x=-0.056, y=y)

        plt.subplots_adjust(hspace=0)
        plt.tight_layout()
        return plt.gca()

    def plot_matrix(ax, mat):
        mat = mat.T
        sns.heatmap(mat, ax=ax, cmap="Blues", cbar=False, 
                    vmin=0, linewidths=0.5, linecolor='#00000022')
        responses = ccobra.syllogistic.RESPONSES.copy()
        responses[-1] = "None"
        ax.set_yticks(np.arange(len(responses)) + 0.5)
        ax.set_yticklabels(responses, rotation=0)
        ax.set_xticks(np.arange(len(ccobra.syllogistic.SYLLOGISMS), step=4) + 0.7)
        ax.set_xticklabels(ccobra.syllogistic.SYLLOGISMS[::4], rotation=90)
        ax.tick_params(axis='x', pad=-4)

    def plot_matrices_comparison(mat1, mat2, text1, text2, 
                                 offset1=0.15, offset2=0.15):
        sns.set(style='whitegrid', palette='colorblind')
        fig, axs = plt.subplots(2, 1, figsize=(11, 3.5), sharex=True)

        plot_matrix(axs[0], mat1)
        plot_matrix(axs[1], mat2)

        axs[0].set_title(text1, rotation=90, x=-0.055, y=offset1)
        axs[1].set_title(text2, rotation=90, x=-0.055, y=offset2)

        plt.subplots_adjust(hspace=0)

        plt.tight_layout()
        return plt.gca()
    return (
        get_mat,
        plot_comparison,
        plot_dual_comparison,
        plot_matrices_comparison,
        plot_matrix,
        plot_pattern,
        response_list,
    )


@app.cell
def __(mo):
    mo.md(f"""
    ## References for Ressources
    Brand, D., & Ragni, M. (2023). Effect of response format on syllogistic reasoning. In M. Goldwater, F. K. Anggoro, B. K. Hayes, & D. C. Ong (Eds.), *Proceedings of the 45th Annual Conference of the Cognitive Science Society*  
    Evans, J., Handley, S., Harper, C., & Johnson-Laird, P. (1999). Reasoning about necessity and possibility: A test of the mental model theory of deduction. *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 25,
    1495-1513.  
    Riesterer, N., Brand, D., & Ragni, M. (2020). Do models capture individuals? Evaluating parameterized models for syllogistic reasoning. In S. Denison, M. Mack, Y. Xu, & B. C. Armstrong (Eds.), *Proceedings of the 42nd Annual
    Conference of the Cognitive Science Society* (pp. 3377–3383). Toronto, ON: Cognitive Science Society
    """)
    return


if __name__ == "__main__":
    app.run()
