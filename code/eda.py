import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt


def categorical_analysis_plot(df, categorical_var, target, showlabel=False):
    plt.figure(dpi=80, figsize=(12,5))
    
    df_var = pd.crosstab(df[categorical_var], df[target])
    df_var['bad_rate'] = round(df_var[1] / ( df_var[0] + df_var[1]), 4)
    x_label = df_var.index
    data1 = df_var[0]
    data2 = df_var[1]
    data3 = df_var['bad_rate']
    x = range(len(x_label))
    bar_width = 0.2
    
    ax1 = plt.subplot(1,1,1) 
    ax1.bar(x, data1, width=bar_width, color='#3A669C', label="Good")
    ax1.bar([i + bar_width for i in x], data2, width=bar_width, color='#C0504D',label="Bad")
    plt.xticks([i + bar_width/2 for i in x], x_label)
    ax1.set_ylabel('Total Count',size=10)
    ax1.set_xlabel(categorical_var,size=10)

    text_heiht = 3
    # 为每个条形图添加数值标签
    for x1,y1 in enumerate(data1):
        ax1.text(x1, y1+text_heiht, y1,ha='center',fontsize=8)
 
    for x2,y2 in enumerate(data2):
        ax1.text(x2+bar_width,y2+text_heiht,y2,ha='center',fontsize=8)

    # bad_rate线 共用纵坐标轴
    ax2 = ax1.twinx()
    p3 = ax2.plot([i + bar_width/2 for i in x], data3, color="gray",linestyle='--', label="Bad_Rate")
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper center")
    ax1.yaxis.grid(True)
    ax1.xaxis.grid(True)
    plt.show()

def numerical_analysis_plot(raw_data, numerical_var, target, showlabel=False):
    plt.figure(dpi=80, figsize=(12, 5))
    ax=sbn.kdeplot(data=raw_data, x= numerical_var, hue=target, multiple='stack')
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    plt.show()


