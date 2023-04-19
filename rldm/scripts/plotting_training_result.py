
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def main():
    PPO_CC = pd.read_csv('/mnt/logs/selected_CSV/PPO_Centralized_Critic.csv')
    IMPALA = pd.read_csv('/mnt/logs/selected_CSV/IMPALA.csv')

    total_timestep_1 = PPO_CC['timesteps_total'].values
    total_timestep_2 = IMPALA['timesteps_total'].values
    print(total_timestep_1)
    print(total_timestep_2)
    timestep = [total_timestep_1, total_timestep_2]

    # assert len(PPO_CC['timesteps_total'].values) == len(IMPALA['timesteps_total'].values),"THE TRAINING TIME BETWEEN TWO ALGORITHM IS NOT EQUAL"

    custom_metrics = "custom_metrics/"
    metrics = ["episode_reward_mean",
            custom_metrics+"statics/teammate_spread_out_mean",
            custom_metrics+"statics/teammate_pass_team_1_mean",
            custom_metrics+"statics/teammate_pass_team_2_mean",
            
            custom_metrics+"statics/teammate_interception_team_1_mean",
            custom_metrics+"statics/teammate_interception_team_2_mean",
            custom_metrics+"statics/teammate_pass_interception_rate_team_1_mean",
            custom_metrics+"statics/teammate_pass_interception_rate_team_2_mean",


            custom_metrics+"game_result/score_reward_episode_mean",
            custom_metrics+"game_result/win_percentage_episode_mean",
            custom_metrics+"game_result/undefeated_percentage_episode_mean",
            custom_metrics+"statics/players_tired_factor_rate_mean",
    ]
    val_pass = {}
    val_interception = {}
    t_val = {}
    for m in metrics:
        
        plt.figure(figsize=(10,7))
        title = m.split('/')[-1]
        tt = title
        if '_team_2' in title:
            continue
        if "_team_1" in title:
            tt = title
            title = title.replace("_team_1", "_")

        title = (" ".join(title.split("_")[:-1])).capitalize()
        fig = plt.figure()
        for algo,name, t,c in zip([PPO_CC, IMPALA], ["PPO_CC", "IMPALA"],timestep,["tab:blue","tab:orange"]):
            val =  algo[m].values

            if "pass" in tt and "interception" in tt:
                val = [a/b for a,b in zip(val_pass[name],val_interception[name])]
                plt.plot(t_val[name],val,label = name, color=c)
                continue
            plt.plot(t,val,label = name,alpha=0.5, color=c)
            
            window = 100
            average_val = []
            average_t = []

            for ind in range(len(val) - window + 1):
                average_val.append(np.mean(val[ind:ind+window]))
                average_t.append(np.mean(t[ind:ind+window]))
            plt.plot(average_t,average_val,color=c)
            if "pass" not in tt and "interception" in tt:
                val_interception[name] = average_val
                t_val[name] = average_t
            if "pass" in tt and "interception" not in tt:
                val_pass[name] = average_val
                t_val[name] = average_t
        # if "pass" in tt and "interception" in tt:
        #     plt.ylim(0,1)
        plt.ylabel( title,size=12)
        plt.xlabel('Training Timesteps',size=12)
        plt.title(title,size=15)
        plt.legend()
        plt.show()
        
        
        fig.savefig('/mnt/logs/plots/'+tt+'.png',dpi=300)
        plt.close(fig)
if __name__ == '__main__':
    main()