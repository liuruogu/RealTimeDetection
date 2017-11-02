import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def loadCSV(FileName):
    # load = np.genfromtxt(FileName, delimiter=',')
    load = pd.read_csv(FileName)
    return load


def main():
    data = loadCSV('Model_Speed_Accuracy.csv')

    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 8}

    plt.rc('font', **font)

    # Group the dataframe by the 'label'
    groups = data.groupby('label')
    fig, ax = plt.subplots()
    ax.margins(0.05)
    for name, group in groups:
        # plot each point in the data frame
        ax.plot(group.Speed, group.Accuracy, marker='o', linestyle='', ms=8, label=name)
    # Add the legend to upperleft
    ax.legend(numpoints=1, loc='upper left')

    # Annotate each plotted data point with its labels
    for i, txt in enumerate(data.Model):
        ax.annotate(txt, (data.Speed.iat[i],data.Accuracy.iat[i]))
        # ax.data.plot.scatter(x='Speed', y='Accuracy', alpha=0.5, fontsize = 12)
    ax.set_xlabel("GPU Time", fontsize= 18)
    ax.set_ylabel("Accuracy (mAP)",fontsize= 18 )

    plt.show()

#call the main
main()
   
