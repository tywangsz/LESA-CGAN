

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np





def PCA_Analysis(dataX, dataX_hat):
    Sample_No = min(len(dataX) - 100, len(dataX_hat) - 100)

    for i in range(Sample_No):
        if (i == 0):
            arrayX = np.reshape(np.mean(np.asarray(dataX[0]), 1), [1, len(dataX[0][:, 0])])
            arrayX_hat = np.reshape(np.mean(np.asarray(dataX_hat[0]), 1), [1, len(dataX[0][:, 0])])
        else:
            arrayX = np.concatenate((arrayX, np.reshape(np.mean(np.asarray(dataX[i]), 1), [1, len(dataX[0][:, 0])])))
            arrayX_hat = np.concatenate(
                (arrayX_hat, np.reshape(np.mean(np.asarray(dataX_hat[i]), 1), [1, len(dataX[0][:, 0])])))


    No = len(arrayX[:, 0])
    colors = ["red" for i in range(No)] + ["blue" for i in range(No)]


    pca = PCA(n_components=2)
    pca.fit(arrayX)
    pca_results = pca.transform(arrayX)
    pca_hat_results = pca.transform(arrayX_hat)


    f, ax = plt.subplots(1)

    plt.scatter(pca_results[:, 0], pca_results[:, 1], c=colors[:No], alpha=0.2, label="Original")
    plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1], c=colors[No:], alpha=0.2, label="Synthetic")

    ax.legend()

    plt.title('PCA plot')
    plt.xlabel('x-pca')
    plt.ylabel('y_pca')
    plt.show()



def tSNE_Analysis(dataX, dataX_hat):
    Sample_No = min(len(dataX) - 100, len(dataX_hat) - 100)

    for i in range(Sample_No):
        if (i == 0):
            arrayX = np.reshape(np.mean(np.asarray(dataX[0]), 1), [1, len(dataX[0][:, 0])])
            arrayX_hat = np.reshape(np.mean(np.asarray(dataX_hat[0]), 1), [1, len(dataX[0][:, 0])])
        else:
            arrayX = np.concatenate((arrayX, np.reshape(np.mean(np.asarray(dataX[i]), 1), [1, len(dataX[0][:, 0])])))
            arrayX_hat = np.concatenate(
                (arrayX_hat, np.reshape(np.mean(np.asarray(dataX_hat[i]), 1), [1, len(dataX[0][:, 0])])))

    final_arrayX = np.concatenate((arrayX, arrayX_hat), axis=0)


    No = len(arrayX[:, 0])
    colors = ["red" for i in range(No)] + ["blue" for i in range(No)]


    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(final_arrayX)


    f, ax = plt.subplots(1)

    plt.scatter(tsne_results[:No, 0], tsne_results[:No, 1], c=colors[:No], alpha=0.2, label="Original")
    plt.scatter(tsne_results[No:, 0], tsne_results[No:, 1], c=colors[No:], alpha=0.2, label="Synthetic")

    ax.legend()

    plt.title('t-SNE plot')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    plt.show()

