from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

def plot_results(result):
    values = defaultdict(lambda: defaultdict(list))
    with open(result, 'r') as f:
        file = f.readlines()
        for line in file:
            split = line.split()
            values[split[5]][split[7]].append(float(split[-1]))
    plt.figure(figsize=(15,10))
    plt.suptitle(result.split('/')[-1].split('.')[0])
    for i,task in enumerate(values.keys(), start=1):
        plt.subplot(2,4,i).set_title(task)
        for cl in values[task].keys():
            plt.plot(values[task][cl], label=cl + ' (' + str(np.argmin(values[task][cl])) + '-' + str(np.argmax(values[task][cl])) + ')')
            plt.legend()
    plt.show()