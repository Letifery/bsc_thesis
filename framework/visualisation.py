import matplotlib.pyplot as plt
import numpy as np

def plot_loss(datapoints, save_path=None):
    plt.plot(np.array(datapoints)[:,0:2])
    plt.legend(['Train Loss', 'Validation Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0,1)
    if save_path is not None:
        plt.savefig(save_path+'\\_loss.png')
    plt.show()

def plot_acc(datapoints, save_path=None):
    plt.plot(np.array(datapoints)[:,2:4])
    plt.legend(['Train Accuracy', 'Validation Accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0,1)
    if save_path is not None:
        plt.savefig(save_path+'\\_accuracy.png')
    plt.show()
