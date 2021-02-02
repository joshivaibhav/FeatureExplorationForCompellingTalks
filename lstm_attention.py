import numpy as np
import pandas as pd
import tensorflow as tf
import os
import math
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from attention import Attention

from pose_feature_extractor import PoseFeatureExtractor

def prepare_metadata():

    video_metadata = pd.read_csv("likes.csv")
    video_metadata = video_metadata.sort_values(by=['Video_Title'])
    video_metadata = video_metadata.iloc[:,1:]
    scaler = StandardScaler()
    data = video_metadata['Likes'].values.reshape(-1, 1)
    standardize_likes = scaler.fit_transform(data)
    video_metadata['Standardized_Likes'] = np.ravel(standardize_likes)
    video_metadata.to_csv("stats.csv",index=False)

def create_dataset():

    datafiles = 'C:\\Users\\vaibh\\PycharmProjects\\TedxCapstoneDataSet\\body_segments_normalized_new'
    video_metadata = pd.read_csv('C:\\Users\\vaibh\\PycharmProjects\\TedxCapstoneDataSet\\stats.csv')

    data = []
    seg_list = []

    for file, idx in zip(os.listdir(datafiles), video_metadata.index):

        video = pd.read_csv(datafiles + "\\" + file)
        likes = video_metadata['Popularity'][idx]
        select_cols = [col for col in video if "c" not in col]
        video_segment = video[select_cols]
        print(len(video_segment))

        if len(video_segment) > 30:
            length, mod = divmod(len(video_segment), 30)

            for part in range(0, (length * 30), 30):

                segment = video_segment.iloc[part:part + 30, :38]

                if len(segment) < 30:
                    print("bad segment")
                #print(idx, " : >>>>" ,segment.shape)
                data.append([segment, likes])

            seg_list.append(len(data))


    X = np.stack([data[i][0] for i in range(len(data))])
    y = np.stack([data[i][1] for i in range(len(data))])

    return X, y, seg_list


def test_train_split(X, y, segment_splits):

    split = int(len(segment_splits) * 0.8)

    X_train, X_test = X[:segment_splits[:split][-1]], X[segment_splits[:split][-1]:]
    y_train, y_test = y[:segment_splits[:split][-1]], y[segment_splits[:split][-1]:]

    return X_train, X_test, y_train, y_test

def model(X,y,seg_list):

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


    X_train, X_test, y_train, y_test = test_train_split(
        X, y, seg_list)


    for units,epochs,batch_size in [(128, 200, 64),(128, 250, 64),(128, 250, 32),(128, 300, 64), (256, 300, 48), (128, 450, 32), (128, 500, 32),
                                    (128, 500, 64)]:

        model = Sequential()
        model.add(LSTM(units, input_shape=(30, 38), return_sequences=True))
        model.add(Attention())
        #model.add(LSTM(64))
        model.add(Dropout(0.2))
        model.add(Dense(64))
        model.add(Dense(32,activation="sigmoid"))
        model.add(Dropout(0.2))
        model.add(Dense(1,activation="softmax"))


        #print(model.summary())

        model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])

        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2,validation_split=0.1,shuffle=True)
        plot_error(history, units, epochs, batch_size)

        test_loss,test_acc=model.evaluate(X_test, y_test, verbose=2)
        text_file = open(str(epochs)+ "--" +str(batch_size) +"=="+ str(units) +".txt", "w+")
        text_file.write("Test Loss: {}' --- Test Accuracy: {}".format(test_loss,test_acc))
        text_file.close()

        trainPredict = model.predict(X_train)
        testPredict = model.predict(X_test)

        print(confusion_matrix(y_test, testPredict))
        print(classification_report(y_test, testPredict))
        print(accuracy_score(y_test, testPredict))

        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        plot_graphs(history, 'accuracy')
        plt.subplot(1, 2, 2)
        plot_graphs(history, 'loss')
        plt.savefig('{0} -- {1} -- {2}.png'.format(units,epochs,batch_size))



def plot_error(history, units, epochs, batch_size):

    loss = [loss_per_epoch / batch_size for loss_per_epoch in history.history['loss']]
    plt.plot(loss, label='loss',linestyle="--")
    plt.plot([loss_per_epoch / batch_size for loss_per_epoch in history.history['val_loss']], label='val_loss')
    #yhat = savgol_filter((history.history['loss'] / batch_size), 5, 3)
    #plt.plot(yhat, label='smoothed', linestyle="-")
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title("lstm units {0}|epochs {1}|batch size {2} att \n".format(units, epochs, batch_size))
    plt.legend()
    plt.grid(True)
    plt.savefig("loss({0} ;; {1} ;; {2}).png".format(units, epochs, batch_size))
    plt.close()


def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])



def plot_predictions(X_train, y_train, X_test, y_test, trainPred, testPred, units, epochs, batch_size):

    plt.figure()
    plt.subplot(211)

    # TRAINING PREDICTIONS
    n = len(y_train)
    y_train = np.reshape([y_train], (n, 1))

    trainScore = math.sqrt(mean_squared_error(y_train, trainPred))
    testScore = math.sqrt(mean_absolute_error(y_test, testPred))

    plt.plot(y_train, color='black', label='Actual', linestyle='dashdot')
    plt.plot(trainPred, color='green', label='Predicted', linestyle="-")
    plt.xlabel('samples')
    plt.ylabel('Likes')
    plt.ylim([0, 1])
    plt.title("lstm units {0}|epochs {1}| batch size {2} att \n RMSE acheived {3} (training)".format(units, epochs, batch_size,trainScore))
    plt.figlegend(loc='upper right')

    # TESTING PREDICTIONS
    plt.subplot(212)
    n = len(y_test)
    y_test = np.reshape([y_test], (n, 1))

    plt.plot(y_test,color='black', label='Actual',linestyle='dashdot')
    plt.plot(testPred, color='green', label='Predicted',linestyle='-')
    plt.xlabel('samples')
    plt.ylabel('Likes')
    plt.ylim([0, 1])
    plt.title(
        "lstm units {0}|epochs {1}| batch size {2} \n MAE acheived {3} (Testing)".format(units, epochs, batch_size, testScore))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=1, hspace=1)

    plt.savefig("predictions({0} ;; {1} ;; {2}).png".format(units,epochs,batch_size))
    plt.close()


if __name__ == '__main__':

    #prepare_metadata()
    X,y,seg_list = create_dataset()
    #features,labels = generate_RF_features(X, y)
    model(X,y,seg_list)


