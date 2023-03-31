#Plot All set, Intersection set, Random set performance comparision figures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import rc
%matplotlib inline

level='DEG_Level'

#Open files to dataframe
d1 = pd.read_csv(f'Multipartite_{level}_All_Set_Performance.csv')
d2 = pd.read_csv(f'3.Multipartite_{level}_Intersection_Set_Performance.csv' )
d3 = pd.read_csv(f'4.Multipartite_{level}_Random_Set_Performance.csv')

#Select classifier names
models = d1[['classifier']]

#extract the average from All set, Intersection set, Random set performances
def extract_average(a, b, c):
    a_s = d1[[f'{a}']]
    i_s = d2[[f'{b}']]
    r_s = d3[[f'{c}']]
    df = pd.concat([models, a_s, i_s, r_s], axis =1)
    df.set_index(['classifier'], inplace=True)
    return df

#extraact the standard deviation for y-error
def extract_stdev(a, b, c):
    a_s_std = d1[[f'{a}']]
    i_s_std = d2[[f'{b}']]
    r_s_std = d3[[f'{c}']]
    df1_std = pd.concat([models, a_s_std, i_s_std, r_s_std], axis =1)
    df1_std.set_index(['classifier'], inplace=True)
    yerr_df1 = df1_std.iloc[:, :].to_numpy().T
    return df1_std, yerr_df1

#Training precision
Training_precision = extract_average('tr_precision_avg_as','tr_precision_avg_is', 'tr_precision_avg_rs')
#Training precision yerror
Training_precision_yerror =  extract_stdev('tr_precision_stdev_as', 'tr_precision_stdev_is','tr_precision_stdev_rs')[1]
#Training recall
Training_recall = extract_average('tr_recall_avg_as','tr_recall_avg_is', 'tr_recall_avg_rs')
#Training recall yerror
Training_recall_yerror = extract_stdev('tr_precision_stdev_as', 'tr_precision_stdev_is','tr_precision_stdev_rs')[1]
#Training F1
Training_F1 = extract_average('tr_f1_avg_as','tr_f1_avg_is', 'tr_f1_avg_rs')
#Training F1 yerror
Training_F1_yerror = extract_stdev('tr_f1_stdev_as', 'tr_f1_stdev_is','tr_f1_stdev_rs')[1]
#Test precision
Test_precision = extract_average('te_precision_avg_as','te_precision_avg_is', 'te_precision_avg_rs')
#Test precision yerror
Test_precision_yerror =  extract_stdev('te_precision_stdev_as', 'te_precision_stdev_is','te_precision_stdev_rs')[1]
#Test recall
Test_recall = extract_average('te_recall_avg_as','te_recall_avg_is', 'te_recall_avg_rs')
#Test precision yerror
Test_recall_yerror =  extract_stdev('te_recall_stdev_as', 'te_recall_stdev_is','te_recall_stdev_rs')[1]
#Test F1
Test_F1 = extract_average('te_f1_avg_as','te_f1_avg_is', 'te_f1_avg_rs')
#Test precision stdev
Test_F1_stdev = extract_stdev('te_f1_stdev_as', 'te_f1_stdev_is','te_f1_stdev_rs')[0]
#Test precision yerror
Test_F1_yerror =  extract_stdev('te_f1_stdev_as', 'te_f1_stdev_is','te_f1_stdev_rs')[1]
#Ten fold CV
TenFold_CV = extract_average('Tf_CV_Avg_as','Tf_CV_Avg_is', 'Tf_CV_Avg_rs')
#Ten fold yerror
TenFold_CV_yerror =  extract_stdev('Tf_CV_stdev_as', 'Tf_CV_stdev_is','Tf_CV_stdev_rs')[1]

#Au_ROC
Au_ROC = extract_average('au_ROC_avg_as','au_ROC_avg_is', 'au_ROC_avg_rs')
#TAu_ROC yerror
Au_ROC_yerror =  extract_stdev('au_ROC_stdev_as', 'au_ROC_stdev_is','au_ROC_stdev_rs')[1]

#Au_PR
Au_PR = extract_average('au_PR_avg_as','au_PR_avg_is', 'au_PR_avg_rs')
#TAu_ROC yerror
Au_PR_yerror = extract_stdev('au_PR_stdev_as', 'au_PR_stdev_is','au_PR_stdev_rs')[1]

#plot figure
fig, axes = plt.subplots(nrows=3, ncols=3, sharex=False, sharey=False, figsize=(15,10))
ax1 = Training_precision.plot(kind='bar', ax=axes[0,0], 
                              title='a. Training precision',legend=False, rot=0, fontsize=8, 
                              xlabel='Machine Learning Algorithms', ylabel='Accuracy',
                             yerr= Training_precision_yerror, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5))

ax1 = Training_recall.plot(kind='bar', ax=axes[0,1], title='b. Training recall', legend=False, rot=0, fontsize=8, 
                           xlabel='Machine Learning Algorithms',
                          yerr= Training_recall_yerror, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5))
ax1 = Training_F1.plot(kind='bar', ax=axes[0,2], title='c. Training F1-score', legend=False, rot=0, fontsize=8, 
                       xlabel='Machine Learning Algorithms',
                      yerr= Training_F1_yerror, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5))

ax1 = Test_precision.plot(kind='bar', ax=axes[1,0], title='d. Test precision', legend=False, rot=0, fontsize=8, 
                          xlabel='Machine Learning Algorithms', ylabel='Accuracy',
                         yerr= Test_precision_yerror, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5))
ax1 = Test_recall.plot(kind='bar', ax=axes[1,1], title='e. Test recall', legend=False, rot=0, fontsize=8, 
                       xlabel='Machine Learning Algorithms',
                      yerr= Test_recall_yerror, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5))
ax1 = Test_F1.plot(kind='bar', ax=axes[1,2], title='f. Test F1-score', legend=False, rot=0, fontsize=8, 
                   xlabel='Machine Learning Algorithms',
                  yerr= Test_F1_yerror, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5))

ax1 = TenFold_CV.plot(kind='bar', ax=axes[2,0], title='g. 10-fold CV', legend=False, rot=0, fontsize=8, 
                      xlabel='Machine Learning Algorithms', ylabel='Accuracy',
                     yerr= TenFold_CV_yerror, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5))
ax8 = Au_ROC.plot(kind='bar', ax=axes[2,1], title='h. au ROC', legend=False, rot=0, fontsize=8, 
                  xlabel='Machine Learning Algorithms',
                 yerr= Au_ROC_yerror, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5))

ax8.legend(labels=['All Set', 'Intersection Set', 'Random Set'], loc='lower center', borderaxespad=0.1, ncol=3,
               bbox_to_anchor=(0.5, -0.3),
               fancybox=False, shadow=False, prop={'size': 8})
ax8.set_axisbelow(True)

ax1 = Au_PR.plot(kind='bar', ax=axes[2,2], title='i. au PR', legend=False, rot=0, fontsize=8, 
                 xlabel='Machine Learning Algorithms',
                yerr= Au_PR_yerror, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5))
fig.tight_layout()

print("Assessment of the performance of the machine learning algorithms in predicting Multipartite Vs. Unipartite, differentially present gene, genomes under 6-fold cross validation settings. The preformance metrics i) training precision, ii) training recall, iii) training F1, iv) test precision, v) test recall, vi) test F1, vii) 10f CV (ten-fold cross validation), viii) au PR (area under precision recall curve), and ix) au ROC (area under ROC curve). 'All' denotes all genes for taraining (as in the cross-validation partioning), 'Intersection' refers to genes that consistently ranked high across all 6 rounds of cross-validation, and 'Random' refers to randomly sampled genes.")

fig.savefig(f'ML_Plot_Multipartite_{level}.tiff', dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"});
