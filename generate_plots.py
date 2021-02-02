import matplotlib.pyplot as plt
import numpy as np
import random
import math
from sklearn.metrics import mean_squared_error
def plot(y_train):


    plt.figure()
    plt.subplot(211)

    # TRAINING PREDICTIONS

    y_train = [[random.random()] for _ in range(10)]
    trainPred = [[random.random()] for _ in range(10)]

    y_train = np.reshape([y_train], (len(y_train), 1))
    trainPred = np.reshape(trainPred, (len(trainPred), 1))


    trainScore = math.sqrt(mean_squared_error(y_train, trainPred))
    print('Train Score: %.2f RMSE' % (trainScore))

    #testScore = math.sqrt(mean_squared_error(y_test[0], testPred[:, 0]))
    #print('Test Score: %.2f RMSE' % (testScore))

    #plt.plot(y_train, color='blue', label='Real Likes', linestyle="",marker="*",markersize=1)
    plt.plot(trainPred, color='green', label='Predicted Likes',linestyle="dashdot")
    plt.xlabel('samples')
    plt.ylabel('Likes')
    plt.ylim([0, 1])
    plt.title('Training')
    plt.figlegend(loc='upper right')
    plt.show()
    """
    # TESTING PREDICTIONS
    plt.subplot(212)

    y_train = [random.random() for _ in range(500)]
    trainPred = [random.random() for _ in range(500)]

    y_train = np.reshape([y_train], (len(y_train), 1))
    trainPred = np.reshape([trainPred], (len(trainPred), 1))

    plt.plot(y_train, color='black', label='Real Likes (Training)')
    plt.plot(trainPred, color='green', label='Predicted Likes (Training)')
    plt.xlabel('samples')
    plt.ylabel('Likes')
    plt.title('Testing')
    plt.ylim([0,1])

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=1, hspace=1)

    plt.savefig("plts.png")
    """
plot([1,1])