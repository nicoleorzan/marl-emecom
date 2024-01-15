import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wandb
api = wandb.Api()

sns.set()
sns.despine()
sns.set_context("paper", rc={"font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15,
                             "legend.fontsize": 16})
sns.set_style('white', {'axes.edgecolor': "0.5", "pdf.fonttype": 42})
plt.gcf().subplots_adjust(bottom=0.15, left=0.14)
entity = "nicoleorzan" # set to your entity and project 

def get_ids(project):
    print("\nGet ids from:", entity, "/", project)
    runs = api.runs(entity + "/" + project)
    print("There are ", len(runs), "runs")

    all_metrics = []
    all_configs = []

    summary_list, config_list, name_list, id_list = [], [], [], []
    for run in runs: 
        id_list.append(run.id)
        #print(run.summary)
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict)
        metrics_run = list(run.summary._json_dict.keys())
        for metric in metrics_run:
            if metric not in all_metrics:
                all_metrics.append(metric)
        config_cols = list(run.config.keys())
        for conf in config_cols:
            if conf not in all_configs:
                all_configs.append(conf)
        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
             if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)
    #print(run.name)
    metrics_cols = list(run.summary._json_dict.keys()) #+ list(run.config.keys())
    # print("met=", metrics_cols)
    config_cols = list(run.config.keys())
    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list
        })
    return id_list, all_metrics, all_configs #metrics_cols, config_cols

def create_dataset(project):
    api = wandb.Api(timeout=100000)
   
    id_list, metrics_cols, config_cols = get_ids(project)
    cols = metrics_cols + config_cols  + ["run_name"]
    print("cols=", cols)
    print("epoch" in cols)
    rows_data = []
    i = 0
    for _id in id_list:
        run = api.run(entity + "/" + project + "/" + _id) 
        print(i, run.name)
        #print(run.history())
        #if ("agent_0actions_eval_m_0.0" in run.history()):
        for _, row in run.history().iterrows():
            new_line_metrics = [row[metrics_cols[idx]] if metrics_cols[idx] in row else None for idx, _ in enumerate(metrics_cols)]
            new_line_config = [run.config[config_cols[idx]] if config_cols[idx] in run.config else None for idx, _ in enumerate(config_cols)]
            new_line = new_line_metrics + new_line_config + [run.name]
            rows_data.append(new_line)
        
        #print("rows_data=",rows_data)
        #if (jj == 10):
        #    break
        #jj += 1
        i += 1
    return pd.DataFrame(rows_data, columns=cols)


repos = [
   #"EPGG_10agents_no_unc_q-learning_mf[0.5, 1.0, 1.5, 2.5, 3.5]_rep0",
   #"EPGG_10agents_no_unc_q-learning_mf[0.5, 1.0, 1.5, 2.5, 3.5]_rep1",
   #"EPGG_10agents_no_unc_dqn_mf[0.5, 1.0, 1.5, 2.5, 3.5]_rep0",
   "EPGG_10agents_no_unc_dqn_mf[0.5, 1.0, 1.5, 2.5, 3.5]_rep1",
   #"EPGG_10agents_unc_dqn_mf[0.5, 1.0, 1.5, 2.5, 3.5]_rep0",
   #"EPGG_10agents_unc_dqn_mf[0.5, 1.0, 1.5, 2.5, 3.5]_rep1"
]


for repo in repos:
   df = create_dataset(repo)
   df.to_csv(repo+".csv")