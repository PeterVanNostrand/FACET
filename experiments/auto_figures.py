'''
THIS FILE CONTAINS THE .PY EXPORTED CONTENT OF THE JUPYTER NOTEBOOK .IPYNB FILES IN THE FIGURES/ DIRECTORY
IF YOU'RE A HUMAN, USE THOSE FILES INSTEAD OF THIS
'''

def nb_compare_methods_gbc() -> None:
    '''
    This contains code for generating the table "Comparison to state-of-the art counterfactual example generation techniques in terms of explanation time t, sparsity s, L1-Norm 𝛿1, L2-Norm 𝛿2, and validity %. (*) denotes datasets where RFOCSE necessitated uncapped explanation time."

    A CSV and TEX version will be generated, adjustments to the LaTeX table fontsize and table width may be neccessary

    Experiment results files needed: CompareMethods

    Results used in the paper are provided in "results/final" if generating new results run each experiment and update the results paths below
    '''
    # path to each result file
    results_path = "results/compare_methods_tab4.csv"

    # path to output the figure
    export_figures = True
    output_dir = "figures/reproducibility/"
    table_save_name = "compare_methods_gbc_tab4"

    import os
    import re
    import sys

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    sys.path.append(os.path.abspath("../"))
    # from dataset import DS_NAMES
    # from utilities.figure_tools import (get_latest_results_directory, load_results,
    # make_fig_directory)

    cmap = plt.get_cmap("Set1")
    colors = cmap.colors
    if export_figures and not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    all_results = pd.read_csv(results_path)
    all_results = all_results[all_results["n_trees"] == 100]

    ########### rename the FACET varitions to match the paper ###########
    # FACET on random forest --> FCT-RF
    all_results.loc[all_results["model_type"] == "RandomForest", "explainer"] = "FCT-RF"
    # FACET on gradient boosting ensemble with complete interesection --> FCT-GB1
    gb1_rows = (all_results["model_type"] == "GradientBoostingClassifier") & (
        all_results["gbc_intersection"] == "CompleteEnsemble")
    all_results.loc[gb1_rows, "explainer"] = "FCT-GB1"
    # FACET on gradient boosting ensemble with complete interesection --> FCT-GB2
    gb2_rows = (all_results["model_type"] == "GradientBoostingClassifier") & (
        all_results["gbc_intersection"] == "MinimalWorstGuess")
    all_results.loc[gb2_rows, "explainer"] = "FCT-GB2"
    all_results.head(20)

    # group by the new names
    all_results = all_results.groupby(["dataset", "explainer"]).mean().reset_index()

    found_expl = list(all_results["explainer"].unique())
    found_ds = list(all_results["dataset"].unique())
    print("explainers:", found_expl)
    print("datasets", found_ds)
    expl_order = ["FCT-RF", "FCT-GB1", "FCT-GB2"]
    all_results.head()

    opt_vals = {}
    bold_opt = False
    domin = {
        "sample_time": True,
        "avg_length": True,
        "avg_dist": True,
        "avg_manhattan": True,
        "per_valid": False,
    }
    metrics = ["sample_time", "avg_length", "avg_dist", "avg_manhattan", "per_valid"]

    for ds in all_results["dataset"].unique():
        idx = (all_results["dataset"] == ds)
        opt_vals[ds] = {}
        for m in metrics:
            if domin[m]:
                opt_vals[ds][m] = all_results[idx][m].min()
            else:
                opt_vals[ds][m] = all_results[idx][m].max()

    metric_latex = {
        "sample_time": "$t\downarrow$",
        "avg_length": "$\delta_0\downarrow$",
        "avg_manhattan": "$\delta_1\downarrow$",
        "avg_dist": "$\delta_2\downarrow$",
        "per_valid": "$\%\\uparrow$",
    }
    pretty_names = {
        "FCT-RF": "FCT-RF",
        "FCT-GB1": "FCT-GB1",
        "FCT-GB2": "FCT-GB2",
    }
    all_metrics = ["sample_time", "avg_length", "avg_manhattan", "avg_dist", "per_valid"]

    def df_to_csv_latex(df_source, expls, metrics, fname, include_row_label=True):
        # df_ds = df_source["dataset"].unique()
        df_ds = ["compas", "glass", "vertebral"]
        df = df_source.copy()
        df = df.set_index(["dataset", "explainer"])
        with open(output_dir + fname + ".csv", "w") as csv:
            with open(output_dir + fname + ".tex", "w") as tab:
                # csv header row start
                if include_row_label:
                    csv.write("dataset,")
                # tab header row start
                tab.write("\\begin{table*}[t]\n\small\n\centering\n\\begin{tabularx}{0.95\\textwidth}{")
                if include_row_label:
                    tab.write("|X")
                tab.write("|")
                for expl in expls:
                    for m in metrics:
                        tab.write("c")
                    tab.write("|")
                tab.write("}\n\hline")
                if include_row_label:
                    tab.write("\\textbf{Dataset}")
                # csv and tab header row
                for expl in expls:
                    tab.write(" & \multicolumn{5}{c|}{\\textbf{" + pretty_names[expl] + "}}")
                    for m in metrics:
                        csv.write(pretty_names[expl] + ",")
                    csv.write(",")
                csv.write("\n")
                tab.write(" " + re.escape("\\") + "\n")
                # header row two
                csv.write(",")
                for expl in expls:
                    for m in metrics:
                        csv.write(m + ",")
                        tab.write("& " + metric_latex[m] + " ")
                    csv.write(",")
                csv.write("\n")
                tab.write(re.escape("\\") + "\n\hline\n")
                # csv and tab body row
                for ds in df_ds:
                    if include_row_label:
                        csv.write(ds + ",")
                        tab.write(ds)
                    for expl in expls:
                        for m in metrics:
                            if m == "per_valid":
                                val = df.loc[ds, expl][m] * 100
                                val_str = "{:0.1f}".format(val)
                                csv.write(val_str + ",")
                                if bold_opt and df.loc[ds, expl][m] == opt_vals[ds][m]:
                                    tab.write(" & \\textbf{" + val_str + "}")
                                else:
                                    tab.write(" & " + val_str)
                            elif m == "avg_length":
                                val = df.loc[ds, expl][m]
                                val_str = "{:0.2f}".format(val)
                                csv.write(val_str + ",")
                                if bold_opt and df.loc[ds, expl][m] == opt_vals[ds][m]:
                                    tab.write(" & \\textbf{" + val_str + "}")
                                else:
                                    tab.write(" & " + val_str)
                            elif m == "avg_dist":
                                val = df.loc[ds, expl][m] / df.loc[ds, "FCT-RF"][m]
                                val_str = "{:0.2f}".format(val)
                                csv.write(val_str + ",")
                                if bold_opt and df.loc[ds, expl][m] == opt_vals[ds][m]:
                                    tab.write(" & \\textbf{" + val_str + "}")
                                else:
                                    tab.write(" & " + val_str)
                            elif m == "avg_manhattan":
                                val = df.loc[ds, expl][m] / df.loc[ds, "FCT-RF"][m]
                                val_str = "{:0.2f}".format(val)
                                csv.write(val_str + ",")
                                if bold_opt and df.loc[ds, expl][m] == opt_vals[ds][m]:
                                    tab.write(" & \\textbf{" + val_str + "}")
                                else:
                                    tab.write(" & " + val_str)
                            elif m == "sample_time":
                                val = df.loc[ds, expl][m]
                                val_str = "{:0.4f}".format(val)
                                csv.write(val_str + ",")
                                if bold_opt and df.loc[ds, expl][m] == opt_vals[ds][m]:
                                    tab.write(" & \\textbf{" + val_str + "}")
                                else:
                                    tab.write(" & " + val_str)
                            else:
                                val_str = str(df.loc[ds, expl][m])
                                csv.write(val_str + ",")
                                tab.write(" & " + val_str)
                        csv.write(",")
                    csv.write("\n")
                    tab.write(" " + re.escape("\\") + "\n")
                # tab latex close
                tab.write("\hline\n\end{tabularx}\n")
                tab.write("\caption{Comparison to state-of-the art counterfactual example generation techniques in terms of explanation time $t$, explanation distance $\delta$, and percent of instances successfully explained. ($\\ast$) denotes cases which necessitated uncapped explanation time.}\n")
                tab.write("\label{tab.compare_methods}\n\\vspace{-7mm}\n\end{table*}\n")

    def df_to_csv_latex_transponse(df_source, expls, ds_names, metrics, fname, include_row_label=True):
        valid_expls = []
        for e in expls:
            if e in found_expl:
                valid_expls.append(e)
        valid_expls
        print(expls)

        valid_ds = []
        for ds in ds_names:
            if ds in found_ds:
                valid_ds.append(ds)
        ds_names = valid_ds
        print(ds_names)

        df_ds = df_source["dataset"].unique()
        df = df_source.copy()
        df = df.set_index(["dataset", "explainer"])

        with open(output_dir + fname + ".csv", "w") as csv:
            with open(output_dir + fname + ".tex", "w") as tab:
                # csv header row start
                if include_row_label:
                    csv.write("dataset,")
                # tab header row start
                tab.write("\\begin{table*}[t]\n\small\n\centering\n\\begin{tabularx}{0.95\\textwidth}{")
                if include_row_label:
                    tab.write("|X")
                tab.write("|")
                for ds in valid_ds:
                    for m in metrics:
                        tab.write("c")
                    tab.write("|")
                tab.write("}\n\hline")
                if include_row_label:
                    tab.write("\\textbf{Dataset}")
                # csv and tab header row
                for ds in valid_ds:
                    tab.write(" & \multicolumn{5}{c|}{\\textbf{" + ds.upper() + "}}")
                    for m in metrics:
                        csv.write(ds.upper() + ",")
                    csv.write(",")
                csv.write("\n")
                tab.write(" " + re.escape("\\") + "\n")
                # header row two
                csv.write(",")
                for ds in valid_ds:
                    for m in metrics:
                        csv.write(m + ",")
                        tab.write("& " + metric_latex[m] + " ")
                    csv.write(",")
                csv.write("\n")
                tab.write(re.escape("\\") + "\n\hline\n")
                # csv and tab body row
                for expl in valid_expls:
                    if include_row_label:
                        csv.write(pretty_names[expl] + ",")
                        tab.write(pretty_names[expl])
                    for ds in valid_ds:
                        for m in metrics:
                            if m == "per_valid":
                                val = df.loc[ds, expl][m] * 100
                                val_str = "{:0.1f}".format(val)
                                csv.write(val_str + ",")
                                if bold_opt and df.loc[ds, expl][m] == opt_vals[ds][m]:
                                    tab.write(" & \\textbf{" + val_str + "}")
                                else:
                                    tab.write(" & " + val_str)
                            elif m == "avg_length":
                                val = df.loc[ds, expl][m]
                                val_str = "{:0.2f}".format(val)
                                csv.write(val_str + ",")
                                if bold_opt and df.loc[ds, expl][m] == opt_vals[ds][m]:
                                    tab.write(" & \\textbf{" + val_str + "}")
                                else:
                                    tab.write(" & " + val_str)
                            elif m == "avg_dist":
                                val = df.loc[ds, expl][m] / df.loc[ds, "FCT-RF"][m]
                                val_str = "{:0.2f}".format(val)
                                csv.write(val_str + ",")
                                if bold_opt and df.loc[ds, expl][m] == opt_vals[ds][m]:
                                    tab.write(" & \\textbf{" + val_str + "}")
                                else:
                                    tab.write(" & " + val_str)
                            elif m == "avg_manhattan":
                                val = df.loc[ds, expl][m] / df.loc[ds, "FCT-RF"][m]
                                val_str = "{:0.2f}".format(val)
                                csv.write(val_str + ",")
                                if bold_opt and df.loc[ds, expl][m] == opt_vals[ds][m]:
                                    tab.write(" & \\textbf{" + val_str + "}")
                                else:
                                    tab.write(" & " + val_str)
                            elif m == "sample_time":
                                val = df.loc[ds, expl][m]
                                val_str = "{:0.4f}".format(val)
                                csv.write(val_str + ",")
                                if bold_opt and df.loc[ds, expl][m] == opt_vals[ds][m]:
                                    tab.write(" & \\textbf{" + val_str + "}")
                                else:
                                    tab.write(" & " + val_str)
                            else:
                                val_str = str(df.loc[ds, expl][m])
                                csv.write(val_str + ",")
                                tab.write(" & " + val_str)
                        csv.write(",")
                    csv.write("\n")
                    tab.write(" " + re.escape("\\") + "\n")
                # tab latex close
                tab.write("\hline\n\end{tabularx}\n")
                tab.write("\caption{Comparison to state-of-the art counterfactual example generation techniques in terms of explanation time $t$, explanation distance $\delta$, and percent of instances successfully explained. ($\\ast$) denotes cases which necessitated uncapped explanation time.}\n")
                tab.write("\label{tab.compare_methods}\n\\vspace{-7mm}\n\end{table*}\n")

    all_results

    found_ds

    # create the table from the main paper
    main_paper_ds = ["adult", "cancer", "credit", "magic", "spambase"]
    df_to_csv_latex_transponse(all_results, expl_order, main_paper_ds, all_metrics, table_save_name, True)

    # create the table from the appendix
    apdx_ds = ["compas", "glass", "vertebral"]
    df_to_csv_latex_transponse(all_results, expl_order, apdx_ds, all_metrics, table_save_name + "_apdx", True)

    all_results[["dataset", "explainer", "avg_dist", "per_valid", "prep_time", "sample_time"]].pivot(
        index=["dataset"], columns=["explainer"], values=["per_valid"])

    all_results[["dataset", "explainer", "avg_dist", "per_valid", "prep_time", "sample_time"]].pivot(
        index=["dataset"], columns=["explainer"], values=["avg_dist"])

    all_results[["dataset", "explainer", "avg_dist", "per_valid", "prep_time", "sample_time"]].pivot(
        index=["dataset"], columns=["explainer"], values=["sample_time"])


def nb_compare_methods() -> None:
    '''
    This contains code for generating the table "Comparison to state-of-the art counterfactual example generation techniques in terms of explanation time t, sparsity s, L1-Norm 𝛿1, L2-Norm 𝛿2, and validity %. (*) denotes datasets where RFOCSE necessitated uncapped explanation time."

    A CSV and TEX version will be generated, adjustments to the LaTeX table fontsize and table width may be neccessary

    Experiment results files needed: CompareMethods

    Results used in the paper are provided in "results/final" if generating new results run each experiment and update the results paths below
    '''
    # path to each result file
    results_path = "results/compare_methods_tab3.csv"

    # path to output the figure
    export_figures = True
    output_dir = "figures/reproducibility/"
    table_save_name = "compare_methods_tab3"

    import os
    import re
    import sys

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    sys.path.append(os.path.abspath("../"))
    #from dataset import DS_NAMES
    #from utilities.figure_tools import (get_latest_results_directory, load_results,
                                        #make_fig_directory)

    cmap = plt.get_cmap("Set1")
    colors = cmap.colors
    if export_figures and not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    all_results = pd.read_csv(results_path)
    all_results = all_results[all_results["n_trees"] == 10]
    all_results = all_results.groupby(["dataset", "explainer"]).mean().reset_index()

    found_expl = list(all_results["explainer"].unique())
    found_ds = list(all_results["dataset"].unique())
    print("explainers:", found_expl)
    print("datasets", found_ds)
    all_results.head()
    ds_order = ["compas", "glass", "vertebral"]
    expl_order = ["FACET", "MACE", "OCEAN", "RFOCSE", "AFT"]

    opt_vals = {}
    bold_opt = False
    domin = {
        "sample_time": True,
        "avg_length": True,
        "avg_dist": True,
        "avg_manhattan": True,
        "per_valid": False,
    }
    metrics = ["sample_time", "avg_length", "avg_dist", "avg_manhattan", "per_valid"]


    for ds in all_results["dataset"].unique():
        idx = (all_results["dataset"] == ds)
        opt_vals[ds] = {}
        for m in metrics:
            if domin[m]:
                opt_vals[ds][m] = all_results[idx][m].min()
            else:
                opt_vals[ds][m] = all_results[idx][m].max()

    metric_latex = {
        "sample_time": "$t\downarrow$",
        "avg_length": "$\delta_0\downarrow$",
        "avg_manhattan": "$\delta_1\downarrow$",
        "avg_dist": "$\delta_2\downarrow$",
        "per_valid": "$\%\\uparrow$",
    }
    pretty_names = {
        "FACET": "FACET",
        "MACE": "MACE",
        "AFT": "AFT",
        "OCEAN": "OCEAN",
        "RFOCSE": "RFOCSE",
    }
    all_metrics = ["sample_time", "avg_length", "avg_manhattan", "avg_dist", "per_valid"]

    def df_to_csv_latex(df_source, expls, metrics, fname, include_row_label=True):
        #df_ds = df_source["dataset"].unique()
        df_ds = ["compas", "glass", "vertebral"]
        df = df_source.copy()
        df = df.set_index(["dataset", "explainer"])
        with open(output_dir + fname + ".csv", "w") as csv:
            with open(output_dir + fname + ".tex", "w") as tab:
                # csv header row start
                if include_row_label:
                    csv.write("dataset,")
                # tab header row start
                tab.write("\\begin{table*}[t]\n\small\n\centering\n\\begin{tabularx}{0.95\\textwidth}{")
                if include_row_label:
                    tab.write("|X")
                tab.write("|")
                for expl in expls:
                    for m in metrics:
                        tab.write("c")
                    tab.write("|")
                tab.write("}\n\hline")
                if include_row_label:
                    tab.write("\\textbf{Dataset}")
                # csv and tab header row
                for expl in expls:
                    tab.write(" & \multicolumn{5}{c|}{\\textbf{" + pretty_names[expl] +"}}")
                    for m in metrics:
                        csv.write(pretty_names[expl] + ",")
                    csv.write(",")
                csv.write("\n")
                tab.write(" " + re.escape("\\") + "\n")
                # header row two
                csv.write(",")
                for expl in expls:
                    for m in metrics:
                        csv.write(m + ",")
                        tab.write("& " + metric_latex[m] + " ")
                    csv.write(",")
                csv.write("\n")
                tab.write(re.escape("\\") + "\n\hline\n")
                # csv and tab body row
                for ds in df_ds:
                    if include_row_label:
                        csv.write(ds + ",")
                        tab.write(ds)
                    for expl in expls:
                        for m in metrics:
                            if m == "per_valid":
                                val = df.loc[ds, expl][m] * 100
                                val_str = "{:0.1f}".format(val)
                                csv.write(val_str + ",")
                                if bold_opt and df.loc[ds, expl][m] == opt_vals[ds][m]:
                                    tab.write(" & \\textbf{" + val_str + "}")
                                else:
                                    tab.write(" & " + val_str)
                            elif m == "avg_length":
                                val = df.loc[ds, expl][m]
                                val_str = "{:0.2f}".format(val)
                                csv.write(val_str + ",")
                                if bold_opt and df.loc[ds, expl][m] == opt_vals[ds][m]:
                                    tab.write(" & \\textbf{" + val_str + "}")
                                else:
                                    tab.write(" & " + val_str)
                            elif m == "avg_dist":
                                val = df.loc[ds, expl][m] / df.loc[ds, "FACET"][m]
                                val_str = "{:0.2f}".format(val)
                                csv.write(val_str + ",")
                                if bold_opt and df.loc[ds, expl][m] == opt_vals[ds][m]:
                                    tab.write(" & \\textbf{" + val_str + "}")
                                else:
                                    tab.write(" & " + val_str)
                            elif m == "avg_manhattan":
                                val = df.loc[ds, expl][m] / df.loc[ds, "FACET"][m]
                                val_str = "{:0.2f}".format(val)
                                csv.write(val_str + ",")
                                if bold_opt and df.loc[ds, expl][m] == opt_vals[ds][m]:
                                    tab.write(" & \\textbf{" + val_str + "}")
                                else:
                                    tab.write(" & " + val_str)
                            elif m == "sample_time":
                                val = df.loc[ds, expl][m]
                                val_str = "{:0.4f}".format(val)
                                csv.write(val_str + ",")
                                if bold_opt and df.loc[ds, expl][m] == opt_vals[ds][m]:
                                    tab.write(" & \\textbf{" + val_str + "}")
                                else:
                                    tab.write(" & " + val_str)
                            else:
                                val_str = str(df.loc[ds, expl][m])
                                csv.write(val_str + ",")
                                tab.write(" & " + val_str)
                        csv.write(",")
                    csv.write("\n")
                    tab.write(" " + re.escape("\\") + "\n")
                # tab latex close
                tab.write("\hline\n\end{tabularx}\n")
                tab.write("\caption{Comparison to state-of-the art counterfactual example generation techniques in terms of explanation time $t$, explanation distance $\delta$, and percent of instances successfully explained. ($\\ast$) denotes cases which necessitated uncapped explanation time.}\n")
                tab.write("\label{tab.compare_methods}\n\\vspace{-7mm}\n\end{table*}\n")

    def df_to_csv_latex_transponse(df_source, expls, ds_names, metrics, fname, include_row_label=True):
        valid_expls = []
        for e in expls: 
            if e in found_expl:
                valid_expls.append(e)
        valid_expls
        print(expls)

        valid_ds = []
        for ds in ds_names: 
            if ds in found_ds:
                valid_ds.append(ds)
        ds_names = valid_ds
        print(ds_names)

        df_ds = df_source["dataset"].unique()
        df = df_source.copy()
        df = df.set_index(["dataset", "explainer"])

        with open(output_dir + fname + ".csv", "w") as csv:
            with open(output_dir + fname + ".tex", "w") as tab:
                # csv header row start
                if include_row_label:
                    csv.write("dataset,")
                # tab header row start
                tab.write("\\begin{table*}[t]\n\small\n\centering\n\\begin{tabularx}{0.95\\textwidth}{")
                if include_row_label:
                    tab.write("|X")
                tab.write("|")
                for ds in valid_ds:
                    for m in metrics:
                        tab.write("c")
                    tab.write("|")
                tab.write("}\n\hline")
                if include_row_label:
                    tab.write("\\textbf{Dataset}")
                # csv and tab header row
                for ds in valid_ds:
                    tab.write(" & \multicolumn{5}{c|}{\\textbf{" + ds.upper() +"}}")
                    for m in metrics:
                        csv.write(ds.upper() + ",")
                    csv.write(",")
                csv.write("\n")
                tab.write(" " + re.escape("\\") + "\n")
                # header row two
                csv.write(",")
                for ds in valid_ds:
                    for m in metrics:
                        csv.write(m + ",")
                        tab.write("& " + metric_latex[m] + " ")
                    csv.write(",")
                csv.write("\n")
                tab.write(re.escape("\\") + "\n\hline\n")
                # csv and tab body row
                for expl in valid_expls:
                    if include_row_label:
                        csv.write(pretty_names[expl] + ",")
                        tab.write(pretty_names[expl])
                    for ds in valid_ds:
                        for m in metrics:
                            if m == "per_valid":
                                val = df.loc[ds, expl][m] * 100
                                val_str = "{:0.1f}".format(val)
                                csv.write(val_str + ",")
                                if bold_opt and df.loc[ds, expl][m] == opt_vals[ds][m]:
                                    tab.write(" & \\textbf{" + val_str + "}")
                                else:
                                    tab.write(" & " + val_str)
                            elif m == "avg_length":
                                val = df.loc[ds, expl][m]
                                val_str = "{:0.2f}".format(val)
                                csv.write(val_str + ",")
                                if bold_opt and df.loc[ds, expl][m] == opt_vals[ds][m]:
                                    tab.write(" & \\textbf{" + val_str + "}")
                                else:
                                    tab.write(" & " + val_str)
                            elif m == "avg_dist":
                                val = df.loc[ds, expl][m] / df.loc[ds, "FACET"][m]
                                val_str = "{:0.2f}".format(val)
                                csv.write(val_str + ",")
                                if bold_opt and df.loc[ds, expl][m] == opt_vals[ds][m]:
                                    tab.write(" & \\textbf{" + val_str + "}")
                                else:
                                    tab.write(" & " + val_str)
                            elif m == "avg_manhattan":
                                val = df.loc[ds, expl][m] / df.loc[ds, "FACET"][m]
                                val_str = "{:0.2f}".format(val)
                                csv.write(val_str + ",")
                                if bold_opt and df.loc[ds, expl][m] == opt_vals[ds][m]:
                                    tab.write(" & \\textbf{" + val_str + "}")
                                else:
                                    tab.write(" & " + val_str)
                            elif m == "sample_time":
                                val = df.loc[ds, expl][m]
                                val_str = "{:0.4f}".format(val)
                                csv.write(val_str + ",")
                                if bold_opt and df.loc[ds, expl][m] == opt_vals[ds][m]:
                                    tab.write(" & \\textbf{" + val_str + "}")
                                else:
                                    tab.write(" & " + val_str)
                            else:
                                val_str = str(df.loc[ds, expl][m])
                                csv.write(val_str + ",")
                                tab.write(" & " + val_str)
                        csv.write(",")
                    csv.write("\n")
                    tab.write(" " + re.escape("\\") + "\n")
                # tab latex close
                tab.write("\hline\n\end{tabularx}\n")
                tab.write("\caption{Comparison to state-of-the art counterfactual example generation techniques in terms of explanation time $t$, explanation distance $\delta$, and percent of instances successfully explained. ($\\ast$) denotes cases which necessitated uncapped explanation time.}\n")
                tab.write("\label{tab.compare_methods}\n\\vspace{-7mm}\n\end{table*}\n")

    # create the table from the main paper
    main_paper_ds = ["adult", "cancer", "credit", "magic", "spambase"]
    output_expl = [_ for _ in expl_order if _ in found_expl]
    df_to_csv_latex_transponse(all_results, output_expl, main_paper_ds, all_metrics, table_save_name, True)

    # create the table from the appendix
    apdx_ds = ["compas", "glass", "vertebral"]
    output_expl = [_ for _ in expl_order if _ in found_expl]
    df_to_csv_latex_transponse(all_results, output_expl, apdx_ds, all_metrics, table_save_name + "_apdx", True)


def nb_index_evaluation() -> None:
    '''
    This contains code for generating the following figures
        - "Evaluation of FACETs explanation analytics using COREX, our counterfactual region explanation index." 
        - "Evaluation of query response time with and without COREX, FACETs bit-vector based counterfactual region explanation index. Varying Nr, the number of indexed counterfactual regions.

    Experiment results files needed: VaryNrects, VaryM

    Results used in the paper are provided in "../results/final" if generating new results run each experiment and update the results paths below
    '''
    # path to each result file
    path_nrects = 'results/vary_nrects_fig11.csv'
    path_nrects_vs_linear = 'results/vary_nrects_fig12.csv'
    path_m = "results/vary_m_fig11.csv"

    # path to output the figure
    export_figures = True
    output_dir = "figures/reproducibility/"
    fig_save_name_1 = "index_evaluation_line_fig11"
    fig_save_name_2 = "index_evaluation_bar_fig12"
    fig_type = ".pdf"

    import os
    import re
    import sys

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib import axis

    sys.path.append(os.path.abspath("../"))
    plot_fontsize = 20
    if export_figures and not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    have_nrects_data = os.path.exists(path_nrects) and os.path.exists(path_m)
    have_vs_linear_data = os.path.exists(path_nrects_vs_linear)

    if have_nrects_data:
        # load the results of vary_nrects for FACET w/bitvector index
        nrects_results = pd.read_csv(path_nrects).groupby(["dataset", "explainer", "n_rects"]).mean().reset_index()
        print("datasets", list(nrects_results["dataset"].unique()))
        # drop rows we don't want for clarity
        nrects_keep = [1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
        nrects_results = nrects_results[nrects_results["n_rects"].isin(nrects_keep)]
        nrects_results.head()

    if have_nrects_data:
        # load the results of vary_m
        m_results = pd.read_csv(path_m).groupby(["dataset", "explainer", "facet_m"]).mean().reset_index()
        print("datasets", list(nrects_results["dataset"].unique()))
        m_results.head()

    cmap = plt.get_cmap("tab10")
    colors = cmap.colors
    ds_colors = {
        "adult": colors[0],
        "cancer": colors[1],
        "credit": colors[2],
        "magic": colors[3],
        "spambase": colors[4],
        "compas": colors[0],
        "glass": colors[1],
        "vertebral": colors[2],
    }
    ds_markers = {
        "adult": "^",
        "cancer": "o",
        "credit": "v",
        "magic": "s",
        "spambase": "D",
        "compas": "^",
        "glass": "o",
        "vertebral": "v",
    }
    marker_size = 10
    line_width = 4
    marker_sizes = {
        "adult": marker_size,
        "cancer": marker_size,
        "credit": marker_size,
        "magic": marker_size,
        "spambase": marker_size,
        "compas": marker_size,
        "glass": marker_size,
        "vertebral": marker_size
    }
    nrects_pretty = {
        0: "0",
        100: "0.1",
        1000: "1",
        5000: "5",
        10000: "10",
        20000: "20",
        30000: "30",
        40000: "40",
        50000: "50",
        60000: "60",
        70000: "70",
        80000: "80",
        90000: "90",
        100000: "100",
    }

    if have_nrects_data:
        results_dict = {
            "n_rects" : nrects_results,
            "facet_m" : m_results,
        }
        def render_plot_ax(datasets, xfeats, yfeats, xlabels=None, ylabels=None, save_name=None, xrange=None, yrange=None, xlog=False, lloc=None, ylog=False, yticks_multi=None, yrange_multi=None, ytick_format="{:.2f}", titles=None):
            #datasets = results_dict[xfeats[0]]["dataset"].unique()
            #datasets = [dataset1, dataset2, dataset3]
            fig = plt.figure(figsize=[32, 4])
            ax = fig.subplots(nrows=1, ncols=5)
            
            for i in range(len(yfeats)):
                xfeature = xfeats[i]
                yfeature =  yfeats[i]
                for ds in datasets:
                    is_ds = results_dict[xfeature]["dataset"] == ds
                    matching_rows = is_ds

                    xvals = results_dict[xfeature][matching_rows][xfeature]
                    if xfeature == "n_rects":
                        xvals = [nrects_pretty[_] for _ in xvals]

                    ax[i].plot(xvals, results_dict[xfeature][matching_rows][yfeature], label=yfeature, marker=ds_markers[ds], lw=line_width, ms=marker_size)
                    
                    ax[i].tick_params(axis='both', which='major', labelsize=plot_fontsize)
                    if xfeature == "n_rects":
                        ax[i].set_xticks(xvals)
                        xtick_labels = [str(_) for _ in xvals]
                        ax[i].set_xticklabels(xtick_labels, fontdict={"horizontalalignment": "center"}, rotation=90)
                    if titles is not None:
                        ax[i].set_title("({:s}) {:s}".format(chr(ord('a')+i), titles[i]), fontdict={"fontsize": plot_fontsize}) 
                    if ylabels is not None:
                        ax[i].set_ylabel(ylabels[i], fontdict={"fontsize": plot_fontsize})
                    if xlabels is not None:
                        ax[i].set_xlabel(xlabels[i], fontdict={"fontsize": plot_fontsize})
                    ax[i].set_xlim(xrange)
                    if yticks_multi is not None:
                        ax[i].set_yticks(yticks_multi[i])
                        ax[i].set_yticklabels([ytick_format.format(_) for _ in yticks_multi[i]])
                    if yrange_multi is not None:
                        bottom = yrange_multi[i][0]
                        top = yrange_multi[i][1]
                        if bottom is not None:
                            ax[i].set_ylim(bottom=bottom)
                        if top is not None:
                            ax[i].set_ylim(top=top)

            lines, labels = ax[0].get_legend_handles_labels()
            fig.subplots_adjust(wspace=0.18)
            legend_labels = [datasets[_].upper() for _ in range(len(datasets))]
            fig.legend(lines, legend_labels, loc="upper center", bbox_to_anchor=(0.51, 1.14), handletextpad=0.5, columnspacing=1, handlelength=1.5, prop={"size": plot_fontsize}, ncol=len(legend_labels))
            if save_name is not None:
                fig.savefig(output_dir + save_name + fig_type, bbox_inches="tight", facecolor='w')

    # ########## MAIN PAPER DATASETS - VARYNRECTS AND M
    if have_nrects_data:
        datasets = ["adult", "cancer", "credit", "magic", "spambase"]
        # axis features and  labels
        ylabels = ["Preprocessing Time", "Explanation Time", "L2 Explanation Distance", "Explanation Sparsity", "Explanation Time"]
        xlabels = ["N Regions (thousands)", "N Regions (thousands)", "N Regions (thousands)", "N Regions (thousands)", "m (Bit-Vector Indices per Dim.)"]
        titles = ["Preproc Time vs N Regions", "Expl Time vs N Regions", "Expl Time vs N Regions", "Expl Sparsity vs N Regions", "Expl Timve vs m"]
        xfeats = ["n_rects", "n_rects", "n_rects", "n_rects", "facet_m"]
        yfeats = ["prep_time", "sample_time", "avg_dist", "avg_length", "sample_time"]
        # create the plot
        render_plot_ax(datasets, xfeats=xfeats, yfeats=yfeats, ylabels=None, xlabels=xlabels, titles=ylabels, save_name=fig_save_name_1)

    # ########## APPENDIX DATASETS - VARYNRECTS AND M
    if have_nrects_data:
        datasets_apdx = ["compas", "glass", "vertebral"]
        # axis features and  labels
        ylabels = ["Preprocessing Time", "Explanation Time", "L2 Explanation Distance", "Explanation Sparsity", "Explanation Time"]
        xlabels = ["N Regions (thousands)", "N Regions (thousands)", "N Regions (thousands)", "N Regions (thousands)", "m (Bit-Vector Indices per Dim.)"]
        titles = ["Preproc Time vs N Regions", "Expl Time vs N Regions", "Expl Time vs N Regions", "Expl Sparsity vs N Regions", "Expl Timve vs m"]
        xfeats = ["n_rects", "n_rects", "n_rects", "n_rects", "facet_m"]
        yfeats = ["prep_time", "sample_time", "avg_dist", "avg_length", "sample_time"]
        # create the plot
        render_plot_ax(datasets_apdx, xfeats=xfeats, yfeats=yfeats, ylabels=None, xlabels=xlabels, titles=ylabels, save_name=fig_save_name_1 + "_apdx")

    if have_vs_linear_data:
        # load the results of vary_nrects comparing FACET's bitvector to a simple linear scan
        nrects_vs_linear = pd.read_csv(path_nrects_vs_linear).groupby(["dataset", "explainer", "n_rects", "facet_search"]).mean().reset_index()
        print("datasets", list(nrects_vs_linear["dataset"].unique()))
        nrects_vs_linear.head()
        # drop rows we don't want for clarity
        nrects_keep = [1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
        nrects_vs_linear = nrects_vs_linear[nrects_vs_linear["n_rects"].isin(nrects_keep)]
        linear_results = nrects_vs_linear[nrects_vs_linear["facet_search"] == "Linear"]
        bitvector_results = nrects_vs_linear[nrects_vs_linear["facet_search"] == "BitVector"]
        nrects_vs_linear.head()

    if have_vs_linear_data:
        def render_bar_ax(datasets, xfeature, yfeature, xlabel, ylabel, title, save_name=None, xrange=None, yrange=None, xlog=False, lloc=None, ylog=False, yticks_multi=None, yrange_multi=None, ytick_format="{:.2f}"):
            print(datasets)
            fig = plt.figure(figsize=[2 + 6*len(datasets), 3.8])
            ax = fig.subplots(nrows=1, ncols=len(datasets), sharex="all") #nrows=1, ncols=5, sharex="all")
            fig.subplots_adjust(left=0.2)
            
            for i in range(len(datasets)):
                ds = datasets[i]
                        
                bitvector_match = (bitvector_results["dataset"] == ds) & (bitvector_results["explainer"] == "FACET")
                linear_match = (linear_results["dataset"] == ds) & (linear_results["explainer"] == "FACET")
                data1 = bitvector_results[bitvector_match][yfeature]
                data2 = linear_results[linear_match][yfeature]
                width =0.3
                ax[i].bar(np.arange(len(data1)), data1, width=width, )
                ax[i].bar(np.arange(len(data2))+ width, data2, width=width)
                
                xticks = [_ + width/2 for _ in range(len(linear_results[linear_match]["n_rects"]))]
                ax[i].set_xticks(xticks,)
                xtick_labels = [nrects_pretty[_] for _ in linear_results[linear_match]["n_rects"]]
                ax[i].set_xticklabels(xtick_labels, rotation=90, fontdict={"horizontalalignment": "center"})        
                ax[i].tick_params(axis='both', which='major', labelsize=plot_fontsize)
                
                ax[i].set_title("({:s}) {:s}".format(chr(ord('a')+i), datasets[i].upper()), fontdict={"fontsize": plot_fontsize}) 
                if xrange is not None:
                    ax[i].set_xlim(xrange)
                if yticks_multi is not None:
                    ax[i].set_yticks(yticks_multi[i])
                    ax[i].set_yticklabels([ytick_format.format(_) for _ in yticks_multi[i]])
                if yrange_multi is not None:
                    bottom = yrange_multi[i][0]
                    top = yrange_multi[i][1]
                    if bottom is not None:
                        ax[i].set_ylim(bottom=bottom)
                    if top is not None:
                        ax[i].set_ylim(top=top)

            ax[0].set_ylabel(ylabel, fontsize=plot_fontsize)
            
            fig.subplots_adjust(wspace=0.22)
            legend_labels = ["FACET COREX", "FACET Linear Scan"]
            fig.legend(labels=legend_labels, loc="upper center", handletextpad=0.5, columnspacing=1, handlelength=1.5, prop={"size": plot_fontsize}, bbox_to_anchor=(0.55, 1.15), ncol=len(legend_labels))
            if save_name is not None:
                fig.savefig(output_dir + save_name + fig_type, bbox_inches="tight", facecolor='w')

    # ########## MAIN PAPER DATASETS - COMPARING LINEAR TO BITVECTOR
    if have_vs_linear_data:
        datasets = ["adult", "cancer", "credit", "magic", "spambase"]
        # yticks_multi = [np.arange(0.0, 5.5, 1), np.arange(0.0, 1.02, 0.2), np.arange(0.0, 1.8, 0.2),
        #                 np.arange(0, 1.3, 0.2), np.arange(0.0, 1.5, 0.2)]
        # yrange_multi = [[0.0, 5.0001], [0.0, 1.01], [0.0, 1.61], [0.0, 1], [0.0, 1.2]]
        yticks_multi = None
        yrange_multi = None
        render_bar_ax(datasets, xfeature="n_rects", yfeature="sample_time", xlabel="N Regions (thousands)", ylabel="Explanation Time (sec)", title="Explanation Time vs NRegions", save_name=fig_save_name_2, yrange_multi=yrange_multi, yticks_multi=yticks_multi)

    # ########## APPENDIX DATASETS - COMPARING LINEAR TO BITVECTOR
    if have_vs_linear_data:
        datasets = ["compas", "glass", "vertebral"]
        # yticks_multi = [np.arange(0.0, 5.5, 1), np.arange(0.0, 1.02, 0.2), np.arange(0.0, 1.8, 0.2),
        #                 np.arange(0, 1.3, 0.2), np.arange(0.0, 1.5, 0.2)]
        # yrange_multi = [[0.0, 5.0001], [0.0, 1.01], [0.0, 1.61], [0.0, 1], [0.0, 1.2]]
        yticks_multi = None
        yrange_multi = None
        render_bar_ax(datasets, xfeature="n_rects", yfeature="sample_time", xlabel="N Regions (thousands)", ylabel="Explanation Time (sec)", title="Explanation Time vs NRegions", save_name=fig_save_name_2 + "_apdx", yrange_multi=yrange_multi, yticks_multi=yticks_multi)




def nb_perturbation_robustness() -> None:
    '''
    This contains code for generating "Evaluation of nearest explanation robustness to varying random perturbation size (percent of space)."

    Experiment results files needed: Perturb

    Results used in the paper are provided in "results/final" if generating new results run each experiment and update the results paths below
    '''
    # path to each result file
    results_path = "results/perturbations_fig9.csv"

    # path to output the figure
    export_figures = True
    output_dir = "figures/reproducibility/"
    fig_save_name = "perturbation_robustness_fig9"
    fig_type = ".pdf"

    import os
    import re
    import sys

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    sys.path.append(os.path.abspath("../"))
    #from dataset import DS_NAMES
    #from utilities.figure_tools import (get_latest_results_directory, load_results,
    #                                   make_fig_directory)

    cmap = plt.get_cmap("Set1")
    colors = cmap.colors

    if export_figures and not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    df = pd.read_csv(results_path)
    df.head()

    all_results = df.groupby(["dataset", "explainer", "perturbation_size"], as_index=False).mean()
    print("explainers:", list(all_results["explainer"].unique()))
    print("datasets:", list(all_results["dataset"].unique()))
    print("pert_sizes:", list(all_results["perturbation_size"].unique()))
    all_results.head()

    # drop some values for clarity
    idx_drop = (df["perturbation_size"].isin([2e-05, 3e-05, 4e-05, 5e-05, 6e-05, 7e-05, 8e-05, 9e-05, 1e-05, 0.00011, 0.00012, 0.00013, 0.00014, 0.00015, 0.00016, 0.00017, 0.00018, 0.00019, 0.0002, 0.00021, 0.00022, 0.00023, 0.00024, 0.00026, 0.00027, 0.00028, 0.00029, 0.0003, 0.00031, 0.00032, 0.00033, 0.00034, 0.00035, 0.00036, 0.00037, 0.00038, 0.00039, 0.0004, 0.00041, 0.00042, 0.00043, 0.00044, 0.00045, 0.00046, 0.00047, 0.00048, 0.00049, 0.105, 0.11, 0.115, 0.12, 0.125, 0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17, 0.175, 0.18, 0.185, 0.19, 0.195, 0.2, 0.205, 0.21, 0.215, 0.22, 0.225, 0.23, 0.235, 0.24, 0.245, 0.25, 0.255, 0.26, 0.265, 0.27, 0.275, 0.28, 0.285, 0.29, 0.295, 0.3, 0.305, 0.31, 0.315, 0.32, 0.325, 0.33, 0.335, 0.34, 0.345, 0.35, 0.355, 0.36, 0.365, 0.37, 0.375, 0.38, 0.385, 0.39, 0.395, 0.4, 0.405, 0.41, 0.415, 0.42, 0.425, 0.43, 0.435, 0.44, 0.445, 0.45, 0.455, 0.46, 0.465, 0.47, 0.475, 0.48, 0.485, 0.49, 0.495, 0.5, 0.0105, 0.011, 0.0115, 0.012, 0.0125, 0.013, 0.0135, 0.014, 0.0145, 0.015, 0.0155, 0.016, 0.0165, 0.017, 0.0175, 0.018, 0.0185, 0.019, 0.0195, 0.02, 0.0205, 0.021, 0.0215, 0.022, 0.0225, 0.023, 0.0235, 0.024, 0.0245, 0.025, 0.0255, 0.026, 0.0265, 0.027, 0.0275, 0.028, 0.0285, 0.029, 0.0295, 0.03, 0.0305, 0.031, 0.0315, 0.032, 0.0325, 0.033, 0.0335, 0.034, 0.0345, 0.035, 0.0355, 0.036, 0.0365, 0.037, 0.0375, 0.038, 0.0385, 0.039, 0.0395, 0.04, 0.0405, 0.041, 0.0415, 0.042, 0.0425, 0.043, 0.0435, 0.044, 0.0445, 0.045, 0.0455, 0.046, 0.0465, 0.047, 0.0475, 0.048, 0.0485, 0.049, 0.0495, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1]))
    df = df.loc[~idx_drop]

    all_results = df.groupby(["dataset", "explainer", "perturbation_size"], as_index=False).mean()
    print("explainers:", list(all_results["explainer"].unique()))
    print("datasets:", list(all_results["dataset"].unique()))
    print("pert_sizes:", list(all_results["perturbation_size"].unique()))
    all_results.head()

    cmap = plt.get_cmap("tab10")
    colors = cmap.colors
    marker_size = 10
    line_width = 4
    ds_colors = {
        "adult": colors[0],
        "cancer": colors[1],
        "credit": colors[2],
        "spambase": colors[3],
        "spambase" : colors[4],
        "compas": colors[0],
        "glass": colors[1],
        "vertebral": colors[2],
    }
    epl_colors = {
        "AFT": colors[0],
        "FACET": colors[1],
        "OCEAN": colors[2],
        "RFOCSE": colors[3],
        "MACE": colors[4]
    }
    explainer_markers = {
        "AFT": "o",
        "FACET": "^",
        "OCEAN": "s",
        "RFOCSE": "v",
        "MACE": "D"
    }
    marker_sizes = {
        "AFT": marker_size,
        "FACET": marker_size,
        "OCEAN": marker_size,
        "RFOCSE": marker_size,
        "MACE": marker_size
    }
    pretty_explainer_names = {
        "AFT": "AFT",
        "FACET": "FACET",
        "OCEAN": "OCEAN",
        "RFOCSE": "RFOCSE",
        "MACE": "MACE",
    }
    found_explainers = all_results["explainer"].unique()
    all_explainers = ["FACET", "OCEAN", "MACE", "RFOCSE", "AFT"]
    explainers = []
    for expl in all_explainers:
        if expl in found_explainers:
            explainers.append(expl)

    plot_fontsize = 20
    figure_widths = [10, 12, 8]

    def render_plot_ax(datasets, xfeature, yfeature, xlabel, ylabel, title, save_name=None, xrange=None, yrange=None, xlog=False, lloc=None, ylog=False, yticks_multi=None, yrange_multi=None, ytick_format="{:.2f}"):
        fig = plt.figure(figsize=[2 + len(datasets) * 6, 4.5])
        ax = fig.subplots(nrows=1, ncols=len(datasets), sharex="all")
        fig.subplots_adjust(left=0.2)
        
        xticks = np.arange(0, 1.01, 0.2)
        yticks = np.arange(0.1, 1.01, 0.2)
        xtick_labels = ["{:.1f}".format(_) for _ in xticks]
        ytick_labels = ["{:.1f}".format(_) for _ in yticks]
        
        for i in range(len(datasets)):
            ds = datasets[i]
            is_ds = all_results["dataset"] == ds
            for expl in explainers:
                is_expl = all_results["explainer"] == expl
                matching_rows = is_ds & is_expl
                ax[i].plot(all_results[matching_rows][xfeature] * 100, all_results[matching_rows][yfeature], label=expl, marker=explainer_markers[expl], lw=line_width, ms=marker_size)
                
                ax[i].tick_params(axis='both', which='major', labelsize=plot_fontsize)
                ax[i].set_xticks(xticks)
                ax[i].set_xticklabels(xtick_labels, fontdict={"horizontalalignment": "center"}, rotation=0)
                ax[i].set_title("({:s}) {:s}".format(chr(ord('a')+i), datasets[i].upper()), fontdict={"fontsize": plot_fontsize}) 
                ax[i].set_xlim(xrange)
                if yticks_multi is not None:
                    ax[i].set_yticks(yticks_multi[i])
                    ax[i].set_yticklabels([ytick_format.format(_) for _ in yticks_multi[i]])
                if yrange_multi is not None:
                    bottom = yrange_multi[i][0]
                    top = yrange_multi[i][1]
                    if bottom is not None:
                        ax[i].set_ylim(bottom=bottom)
                    if top is not None:
                        ax[i].set_ylim(top=top)

        ax[0].set_ylabel("Explanation Validity", fontsize=plot_fontsize)
        lines, labels = ax[0].get_legend_handles_labels()
        fig.subplots_adjust(wspace=0.2)
        legend_labels = [pretty_explainer_names[_] for _ in labels]
        fig.legend(lines, legend_labels, loc="upper center", bbox_to_anchor=(0.54, 1.11), handletextpad=0.5, columnspacing=1, handlelength=1.5, prop={"size": plot_fontsize}, ncol=len(legend_labels))
        fig.savefig(output_dir + save_name + fig_type, bbox_inches="tight", facecolor='w')


    # ######### MAIN PAPER DATASETS
    datasets = ["adult", "cancer", "credit", "magic", "spambase"]
    # axis ranges
    yrange_multi = [[0.15, 1.058], [0.15, 1.03], [0.16, .62], [0.07, 1.03], [0.34, 0.92]]
    yticks_multi = [np.arange(0.2, 1.1, 0.2), np.arange(0.2, 1.3, 0.2), np.arange(0.2, 1.1, 0.1),
                    np.arange(0.2, 1.1, 0.2), np.arange(0.4, 0.93, 0.1)]
    # create plots
    render_plot_ax(datasets, "perturbation_size", "per_valid", "Perturbation Size (% Space)", "Perturbed Validity", "Rate of Explanation Failure", save_name=fig_save_name, xrange=[-0.03, 1.03], yrange_multi=yrange_multi, yticks_multi=yticks_multi, ytick_format="{:.2f}")


    # ######### APPENDIX DATASETS
    datasets_apdx = ["compas", "glass", "vertebral"]
    # axis ranges
    yrange_multi = [[0.15, 1.05], [0.15, .83], [0.20, .68], [0.05, 1.05], [0.35, .92]]
    yticks_multi = [np.arange(0.2, 1.1, 0.2), np.arange(0.2, 1.1, 0.1), np.arange(.25, 1.1, 0.1), np.arange(0.2, 1.1, 0.2), np.arange(0.4, 0.91, 0.1)]
    # create the plot
    render_plot_ax(datasets_apdx, "perturbation_size", "per_valid", "Perturbation Size (% Space)", "Perturbed Validity", "Rate of Explanation Failure", save_name=fig_save_name + "_apdx", xrange=[-0.03, 1.03], yrange_multi=yrange_multi, yticks_multi=yticks_multi, ytick_format="{:.2f}")

def nb_user_workbooks() -> None:
    '''
    This contains code for generating "Evaluation of FACETs explanation analytics with diverse query workloads"

    Experiment results files needed: VaryRobustness, VaryNConstraints, and VaryK

    Results used in the paper are provided in "results/final" if generating new results run each experiment and update the results paths below
    '''
    # path to each result file
    robust_path = "results/vary_robustness_fig10.csv" # run json merge to obtain if new results
    constr_path = "results/vary_nconstraints_fig10.csv"
    k_path = "results/vary_k_fig10.csv"

    # path to output the figure
    export_figures = True
    output_dir = "figures/reproducibility/"
    fig_save_name = "user_workloads_fig10"
    fig_type = ".pdf"


    import glob
    import json
    import os
    import re
    import sys

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    import pandas as pd
    from matplotlib import axis
    from matplotlib.axis import Axis

    def load_json_results(results_dir):
        '''
        Utility to loads all results files in the given directory and merges them to a single pandas dataframe.
        '''
        results_df = None
        for name in glob.glob(results_dir):
            fname: str = os.path.basename(name)
            if "result" in fname:
                parsed_text = fname.split("_")
                with open(name, "r") as f:
                    results = json.load(f)    
                    results["dataset"] = parsed_text[0]
                    results["explainer"] = parsed_text[1]
                    results["min_robust"] = float(parsed_text[2][1:])
                    results["iteration"] = int(parsed_text[3])
                    if results_df is None:
                        results_df = pd.DataFrame([results])
                    else:
                        results_df = pd.concat([results_df, pd.DataFrame([results])])
        return results_df

    # set to true and update paths to merge if generating new results
    do_merge = False
    if do_merge:
        results_dir = "results/vary-robustness/*"
        output_path = "results/vary_robustness_merged.csv"
        results_df = load_json_results(results_dir)
        results_df.to_csv(output_path, index=False)
        results_df.head()
        robust_path = output_path

    # read in robustness data
    robust_results = pd.read_csv(robust_path)

    # Group by minimum robustness, dataset taking the average across each iteration
    robust_results = robust_results.groupby(["dataset", "min_robust"], as_index=False).mean() 
    robust_results["explainer"] = "facet"

    keep_values = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5000000000000004, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.000000000000001, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]) / 100
    idx_keep = robust_results["min_robust"].isin(keep_values)
    robust_results = robust_results[idx_keep]
    robust_results.head()

    # read in constraints data
    constr_results = pd.read_csv(constr_path)
    constr_results = constr_results.groupby(["dataset", "nconstr"], as_index=False).mean() 
    constr_results["explainer"] = "facet"
    # show results after 2
    constr_results = constr_results[(constr_results['nconstr'] > 1)]
    constr_results.head()

    # read in k data
    k_results = pd.read_csv(k_path)
    k_results = k_results.groupby(["dataset", "facet_k"], as_index=False).mean() 
    k_results["explainer"] = "facet"
    # idx_drop = ((k_results["facet_k"] % 2) > 0.00001)  # for even values
    idx_drop = ((k_results["facet_k"] % 2) < 0.00001)  # for odd values
    k_results = k_results[~idx_drop]
    k_results.head()

    # create viz
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    cmap = plt.get_cmap("tab10")
    colors = cmap.colors
    ds_colors = {    
        "adult" : colors[0],
        "cancer" : colors[1],
        "credit" : colors[2],
        "magic" : colors[3],
        "spambase" : colors[4],
        "compas" : colors[0],
        "glass" : colors[1],
        "vertebral" : colors[2],
    }
    ds_markers = {
        "adult" : "^",
        "cancer" : "o",
        "credit" : "v",
        "magic" : "s",
        "spambase" : "D",
        "compas": "^",
        "glass": "o",
        "vertebral": "v",
    }
    marker_size = 10
    line_width = 4
    lhorz = True
    loc = "upper center"
    marker_sizes = {
        "adult" : marker_size,
        "cancer" : marker_size,
        "credit" : marker_size,
        "magic" : marker_size,
        "spambase" : marker_size,
        "compas": marker_size,
        "glass": marker_size,
        "vertebral": marker_size,
    }

    plot_fontsize = 20

    results_dict = {
        "min_robust": robust_results,
        "nconstr": constr_results,
        "facet_k": k_results,
    }

    # plot features
    ylabels = ["Explanation Time", "L2 Explanation Distance", "Explanation Sparsity", "Explanation Time", "Explanation Time"]
    xlabels = ["Minimum Robustness (% Space)", "Minimum Robustness (% Space)", "Minimum Robustness (% Space)", "N Constraints", "k (Explanations Requested)"]
    titles = ["Explantion Time vs Min Robust", "L2-Norm vs Min Robust", "Sparsity vs Min Robust", "Explanation Time vs Nc", "Sample Time vs k"]
    xfeats = ["min_robust", "min_robust", "min_robust", "nconstr", "facet_k"]
    yfeats = ["sample_time", "avg_dist", "avg_length", "sample_time", "sample_time"]

    def render_plot_ax(datasets, xfeats, yfeats, xlabels=None, ylabels=None, save_name=None, xrange=None, yrange=None, xlog=False, lloc=None, ylog=False, yticks_multi=None, yrange_multi=None, ytick_format="{:.2f}", titles=None):
        fig = plt.figure(figsize=[32, 4])
        ax = fig.subplots(nrows=1, ncols=5) #nrows=1, ncols=5, sharex="all")
        
        for i in range(len(yfeats)):
            xfeature = xfeats[i]
            yfeature =  yfeats[i]
            for ds in datasets:
                is_ds = results_dict[xfeature]["dataset"] == ds
                matching_rows = is_ds

                xvals = results_dict[xfeature][matching_rows][xfeature]
                if xfeature == "min_robust":
                    xvals = [_ * 100 for _ in xvals]

                ax[i].plot(xvals, results_dict[xfeature][matching_rows][yfeature], label=yfeature, marker=ds_markers[ds], lw=line_width, ms=marker_size)
                ax[i].tick_params(axis='both', which='major', labelsize=plot_fontsize)

                if titles is not None:
                    ax[i].set_title("({:s}) {:s}".format(chr(ord('a')+i), titles[i]), fontdict={"fontsize": plot_fontsize})
                if ylabels is not None:
                    ax[i].set_ylabel(ylabels[i], fontdict={"fontsize": plot_fontsize})
                if xlabels is not None:
                    ax[i].set_xlabel(xlabels[i], fontdict={"fontsize": plot_fontsize})
                ax[i].set_xlim(xrange)
                if yticks_multi is not None:
                    ax[i].set_yticks(yticks_multi[i])
                    ax[i].set_yticklabels([ytick_format.format(_) for _ in yticks_multi[i]])
                if yrange_multi is not None:
                    bottom = yrange_multi[i][0]
                    top = yrange_multi[i][1]
                    if bottom is not None:
                        ax[i].set_ylim(bottom=bottom)
                    if top is not None:
                        ax[i].set_ylim(top=top)
        lines, labels = ax[0].get_legend_handles_labels()
        fig.subplots_adjust(wspace=0.18)


        #fig.subplots_adjust(wspace=0.25)
        legend_labels = [datasets[_].upper() for _ in range(len(datasets))]
        fig.legend(lines, legend_labels, loc="upper center", bbox_to_anchor=(0.51, 1.13), handletextpad=0.5, columnspacing=1, handlelength=1.5, prop={"size": plot_fontsize}, ncol=len(legend_labels))
        if save_name is not None:
            fig.savefig(output_dir + save_name + fig_type, bbox_inches="tight", facecolor='w')


    # ########### MAIN PAPER DATASETS
    datasets = ["adult", "cancer", "credit", "magic", "spambase"]
    # axes bounds
    yticks_multi = [[round(_,1) for _ in np.arange(0.0, .45, .1)], [round(_,1) for _ in np.arange(0.0, .45, 0.1)], np.arange(0, 24, 5), [round(_,2) for _ in np.arange(0, .18, 0.05)], np.arange(0.0, 1.54, 0.5)]
    yrange_multi = [[-.001, .47], [.13, .49], [1.8, 22], [-.001, .2], [0.0, 1.52]]
    # render the plot
    render_plot_ax(datasets, xfeats=xfeats, yfeats=yfeats, ylabels=None, xlabels=xlabels, titles=ylabels, save_name=fig_save_name)


    # ########### APPENDIX DATASETS
    datasets_apdx = ["compas", "glass", "vertebral"]
    # axes bounds
    yticks_multi = [np.arange(0.0, .5, .1), [round(_,2) for _ in np.arange(0.0, 1, 0.05)], np.arange(0, 7, 1), np.arange(0, .09, 0.005), np.arange(0.0, .06, 0.01)]
    yrange_multi = [[-.001, .5], [0.09, .38], [0.6, 6.6], [0.0, .018], [0.0, .041]]
    # render the plot
    render_plot_ax(datasets_apdx, xfeats=xfeats, yfeats=yfeats, ylabels=None, xlabels=xlabels, titles=ylabels, save_name=fig_save_name + "_apdx")


def nb_vary_ntrees() -> None:
    '''
    This contains code for generating the following figures
        - "Explanation time as a function of model complexity. Varying number of trees T" 
        - "Explanation distance as a function of model complexity. Varying number of trees T"

    Experiment results files needed: VaryNtrees

    Results used in the paper are provided in "results/final" if generating new results run each experiment and update the results paths below
    '''
    # path to each result file
    results_path = "results/vary_ntrees_fig13_14.csv"

    # path to output the figure
    export_figures = True
    output_dir = "figures/reproducibility/"
    fig_save_name_1 = "vary_ntrees_time_fig13_14"
    fig_save_name_2 = "vary_ntrees_dist_fig13_14"
    fig_type = ".pdf"

    import os
    import re
    import sys

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    sys.path.append(os.path.abspath("../"))
    cmap = plt.get_cmap("Set1")
    colors = cmap.colors
    if export_figures and not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    dataset1 = "adult"
    dataset2 = "cancer"
    dataset3 = "credit"
    dataset4 = "magic"
    dataset5 = "spambase"

    # dataset1 = "compas"
    # dataset2 = "glass"
    # dataset3 = "vertebral"

    cmap = plt.get_cmap("tab10")
    colors = cmap.colors
    marker_size = 12
    line_width = 4
    ds_colors = {
        "adult": colors[0],
        "cancer": colors[1],
        "credit": colors[2],
        "magic": colors[3],
        "spambase": colors[4],
        "compas": colors[0],
        "glass": colors[1],
        "spambase": colors[2],
    }
    epl_colors = {
        "AFT": colors[0],
        "FACET": colors[1],
        "OCEAN": colors[2],
        "RFOCSE": colors[3],
        "MACE": colors[4]
    }
    explainer_markers = {
        "AFT": "o",
        "FACET": "^",
        "OCEAN": "s",
        "RFOCSE": "v",
        "MACE": "D"
    }
    marker_sizes = {
        "AFT": marker_size,
        "FACET": marker_size,
        "OCEAN": marker_size,
        "RFOCSE": marker_size,
        "MACE": marker_size
    }
    pretty_explainer_names = {
        "AFT": "AFT",
        "FACET": "FACET",
        "OCEAN": "OCEAN",
        "RFOCSE": "RFOCSE",
        "MACE": "MACE",
    }

    all_results = pd.read_csv(results_path).groupby(["dataset", "explainer", "n_trees"], as_index=False).mean()
    found_explainers = all_results["explainer"].unique()
    all_explainers = ["FACET", "OCEAN", "MACE", "RFOCSE", "AFT"]
    explainers = []
    for expl in all_explainers:
        if expl in found_explainers:
            explainers.append(expl)
    plot_fontsize = 20
    all_results.head()

    def render_plot_ax(datasets, xfeature, yfeature, xlabel, ylabel, title, save_name=None, xrange=None, yrange=None, xlog=False, lloc=None, ylog=False, yticks_multi=None, yrange_multi=None, ytick_format="{:.2f}"):
        results = all_results.groupby(["dataset", "explainer"], as_index=False).mean()
        fig = plt.figure(figsize=[2 + 6*len(datasets), 4])
        ax = fig.subplots(nrows=1, ncols=len(datasets), sharex="all")
        xticks = np.arange(0, 501, 100)
        xtick_labels = ["{:d}".format(_) for _ in xticks]
        fig.subplots_adjust(left=0.2)

        for i in range(len(datasets)):
            ds = datasets[i]
            is_ds = all_results["dataset"] == ds
            for expl in explainers:
                is_expl = all_results["explainer"] == expl
                matching_rows = is_ds & is_expl
                ax[i].plot(all_results[matching_rows][xfeature], all_results[matching_rows][yfeature], label=expl, marker=explainer_markers[expl], lw=line_width, ms=marker_size)
                
                ax[i].tick_params(axis='both', which='major', labelsize=plot_fontsize)
                if xrange is not None:
                    ax[i].set_xticks(xticks)
                    ax[i].set_xticklabels(xtick_labels, fontdict={"horizontalalignment": "center"}, rotation=0)
                # optional letter: "(" + chr(ord('a')+i) + ") " 
                ax[i].set_title("({:s}) {:s}".format(chr(ord('a')+i), datasets[i].upper()), fontdict={"fontsize": plot_fontsize}) 
                if xrange is not None:
                    ax[i].set_xlim(xrange)
                if yticks_multi is not None:
                    ax[i].set_yticks(yticks_multi[i])
                    ax[i].set_yticklabels([ytick_format.format(_) for _ in yticks_multi[i]])
                if yrange_multi is not None:
                    bottom = yrange_multi[i][0]
                    top = yrange_multi[i][1]
                    if bottom is not None:
                        ax[i].set_ylim(bottom=bottom)
                    if top is not None:
                        ax[i].set_ylim(top=top)


        ax[0].set_ylabel(ylabel, fontsize=plot_fontsize)
        
        # ax[0].set_yticklabels(tick_labels)
        lines, labels = ax[0].get_legend_handles_labels()
        fig.subplots_adjust(wspace=0.2)
        legend_labels = [pretty_explainer_names[_] for _ in labels]
        fig.legend(lines,
            legend_labels, loc="upper center", bbox_to_anchor=(0.55, 1.13), handletextpad=0.5, columnspacing=1, handlelength=1.5, prop={"size": plot_fontsize}, ncol=len(legend_labels)
        )
        if save_name is not None:
            fig.savefig(output_dir + save_name + fig_type, bbox_inches="tight", facecolor='w')

    # ######### MAIN PAPER DATASETS - EXPLANATION TIME
    datasets = ["adult", "cancer", "credit", "magic", "spambase"]
    # axis ranges
    # yticks_multi = [np.arange(0, 720, 100,), np.arange(0, 65, 10), np.arange(0, 640, 100), np.arange(0, 161, 20), np.arange(0, 61, 10)]
    # yrange_multi = [[-3, 725], [-1.5, 66], [-20, 640], [-4, 165], [-3, 60]]
    yticks_multi = None
    yrange_multi = None
    # create the plot
    render_plot_ax(datasets, xfeature="n_trees", yfeature="sample_time", xlabel="T (Number of Trees)", ylabel="Explanation Time (sec)", title="Explanation Time vs Num Trees", save_name=fig_save_name_1, xrange=[-10, 520], yrange_multi=yrange_multi, yticks_multi=yticks_multi, ytick_format="{:d}")

    # ######### APPENDIX DATASETS - EXPLANATION TIME
    datasets_apdx = ["compas", "glass", "vertebral"]
    # axis ranges
    # yticks_multi = [np.arange(0, 720, 100,), np.arange(0, 65, 5), np.arange(0, 37, 5), np.arange(0, 161, 20), np.arange(0, 61, 10)]
    # yrange_multi = [[-3, 470], [-1.5, 23], [-1.5, 36], [-4, 165], [-3, 60]]
    yticks_multi = None
    yrange_multi = None
    # create the plot
    render_plot_ax(datasets_apdx, xfeature="n_trees", yfeature="sample_time", xlabel="T (Number of Trees)", ylabel="Explanation Time (sec)", title="Explanation Time vs Num Trees", save_name=fig_save_name_1 + "_apdx", xrange=[-10, 520], yrange_multi=yrange_multi, yticks_multi=yticks_multi, ytick_format="{:d}")

    # ###### MAIN PAPER DATASETS - EXPLANTION DISTANCE
    datasets = ["adult", "cancer", "credit", "magic", "spambase"]
    # axis parameters
    # yticks_multi = [np.arange(0, 0.45, 0.05), np.arange(0, 0.46, 0.05), np.arange(0, 0.46, 0.05), np.arange(0.1, 0.19, 0.02), np.arange(0, 0.13, 0.02)]
    # yrange_multi = [[0.06, 0.42], [0.14, 0.47], [0.06, 0.47], [0.09, 0.185], [-0.005, 0.13]]
    yticks_multi = None
    yrange_multi = None
    # create the parameters
    render_plot_ax(datasets, xfeature="n_trees", yfeature="avg_dist", xlabel="T (Number of Trees)", ylabel="Expl. Distance (L2)", title="Average Distance vs Num Trees", save_name=fig_save_name_2, xrange=[-10, 520], yticks_multi=yticks_multi, yrange=[None, None], yrange_multi=yrange_multi)

    # ###### APPENDIX DATASETS - EXPLANTION DISTANCE
    datasets_apdx = ["compas", "glass", "vertebral"]
    # axis parameters
    yticks_multi = None
    yrange_multi = None
    # create the parameters
    render_plot_ax(datasets_apdx, xfeature="n_trees", yfeature="avg_dist", xlabel="T (Number of Trees)", ylabel="Expl. Distance (L2)", title="Average Distance vs Num Trees", save_name=fig_save_name_2 + "_apdx", xrange=[-10, 520], yticks_multi=yticks_multi, yrange=[None, None], yrange_multi=yrange_multi)


