import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from utils import Statistics
from pose_feature_extractor import PoseFeatureExtractor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def create_dataset():
    datafiles = 'C:\\Users\\vaibh\\PycharmProjects\\TedxCapstoneDataSet\\body_segments_normalized_new'
    video_metadata = pd.read_csv('C:\\Users\\vaibh\\PycharmProjects\\TedxCapstoneDataSet\\stats.csv')

    data = []
    seg_list = []

    for file, idx in zip(os.listdir(datafiles), video_metadata.index):

        video = pd.read_csv(datafiles + "\\" + file)
        likes = video_metadata['Popularity'][idx]
        select_cols = [col for col in video.columns if "_c" not in col]
        conf_cols = [col for col in video.columns if "_c" in col]
        video_segment = video[select_cols]
        conf_values = video[conf_cols]
        avg_conf = np.mean(conf_values.iloc[:, :18].values)

        if len(video_segment) > 100:
            length, mod = divmod(len(video_segment), 100)

            for part in range(0, (length * 100), 100):

                segment = video_segment.iloc[part:part + 100, :]

                if len(segment) < 100:
                    print("bad segment")
                # print(idx, " : >>>>" ,segment.shape)
                data.append([segment, likes, avg_conf])

            seg_list.append(len(data))

    X = np.stack([data[i][0] for i in range(len(data))])
    y = np.stack([data[i][1] for i in range(len(data))])
    c = np.stack([data[i][2] for i in range(len(data))])

    return X, y, c, seg_list


def test_train_split(X, y, segment_splits):
    split = int(len(segment_splits) * 0.8)
    X_train, X_test = X[:segment_splits[:split][-1]], X[segment_splits[:split][-1]:]
    y_train, y_test = y[:segment_splits[:split][-1]], y[segment_splits[:split][-1]:]

    return X_train, X_test, y_train, y_test


def generate_RF_features(X, y, c):
    SPEAKER_ACTION_FEATURE_POINTS = [#[2, True, False, False, True, True, True, True, True],
                                     #[3, True, False, False, True, True, True, True, True],
                                     #[4, True, False, False, True, True, True, True, True],
                                     #[0, True, False, False, True, True, True, True, True],
                                     #[16, True, False, False, True, True, True, True, True],
                                     #[5, True, False, False, True, True, True, True, True],
                                     #[6, True, False, False, True, True, True, True, True],
                                     #[7, True, False, False, True, True, True, True, True],
                                     #[(2, 3), False, True, True, False, True],
                                     #[(3, 4), False, True, True, False, True],
                                     [(2, 4), False, True, True, False, True],
                                     [(2, 5),False, True, True, False, True],
                                     #[(5, 6), False, True, True, False, True],
                                    #[(5, 7), False, True, True, False, True],
                                     [(5, 7), False, True, True, False, True]]

    features = []
    labels = []
    for segment, likes, conf in zip(X, y, c):
        norm_factor = segment[0, -1:][0]
        segment_features = segment[:, :38]

        feat_extractor = PoseFeatureExtractor(feature_points=SPEAKER_ACTION_FEATURE_POINTS, segment_length=100,
                                              norm_factor=norm_factor)
        segment_feature_vector = feat_extractor.extract(np.array([segment_features]))[0]

        #segment_feature_vector = np.append(segment_feature_vector, conf)

        features.append(segment_feature_vector)

        labels.append(likes)

    return features, labels


def rf_regressor(features, labels):
    X_train, X_test, y_train, y_test = test_train_split(
        features, labels, seg_list)

    plt.plot(y_test, color='blue')

    model = RandomForestRegressor(random_state=1, n_estimators=64)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(y_pred)

    # print(mean_absolute_error(y_test, y_pred))
    # print(mean_squared_error(y_test, y_pred))

    plt.plot(y_pred, color='red')

    plt.show()


def get_models():

    models = dict()
    # define number of trees to consider
    rf_hyper_parameters = [(64,"auto",None),(64,"log2",None), (64,10,20)] #[(64, 8, 2), (64,None,3),(64, 12, 10), (64, None, None)]
    for n_trees, max_features, depth in rf_hyper_parameters:
        models["" + str(n_trees) + "" + str(max_features) + "" + str(depth)] = RandomForestClassifier(n_estimators=n_trees,random_state=1, max_features=max_features,
                                                max_depth=depth)
    return models

def evaluate_model(model, X, y):

    # define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model and collect the results
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores

def rf_classifier(features, labels):
    X_train, X_test, y_train, y_test = test_train_split(
        features, labels, seg_list)

    print(len(X_train[0]))
    #model = RandomForestClassifier(random_state=2, n_estimators=64)
    models = get_models()
    #cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    #n_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # report performance

    for name, model in models.items():
        # evaluate the model
        results, names = list(), list()
        scores = evaluate_model(model, X_train, y_train)
        # store the results
        results.append(scores)
        names.append(name)
        # summarize the performance along the way
        print('{0} ---  Accuracy: {1} , ({2})'.format(name, np.mean(scores), np.std(scores)))

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print("# OF ESTIMATORS ", name)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print(accuracy_score(y_test, y_pred))

        print(" _________________________________________________")


if __name__ == '__main__':
    X, y, c, seg_list = create_dataset()
    features, labels = generate_RF_features(X, y, c)
    print(len(features))
    rf_classifier(features, labels)
    # rf_regressor(features, labels)
