import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def loadCSV(FileName):
    # load = np.genfromtxt(FileName, delimiter=',')
    load = pd.read_csv(FileName)
    return load


def main():
    # fig, ax = plt.subplots()
    data = loadCSV('Model_Speed_Accuracy.csv')

    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 8}

    plt.rc('font', **font)

    groups = data.groupby('label')
    fig, ax = plt.subplots()
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        ax.plot(group.Speed, group.Accuracy, marker='o', linestyle='', ms=8, label=name)
    ax.legend(numpoints=1, loc='upper left')

    for i, txt in enumerate(data.Model):
        ax.annotate(txt, (data.Speed.iat[i],data.Accuracy.iat[i]))
        # ax.data.plot.scatter(x='Speed', y='Accuracy', alpha=0.5, fontsize = 12)
    ax.set_xlabel("GPU Time", fontsize= 18)
    ax.set_ylabel("Accuracy (mAP)",fontsize= 18 )

    plt.show()


#call the main
main()
   
