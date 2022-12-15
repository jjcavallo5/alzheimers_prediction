import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

def load_data(test_size=0.2):
    df = pd.read_csv("alzheimer.csv")
    df = df.dropna()
    df['M/F'] = df['M/F'].replace({'M': 0, 'F': 1})
    df = df.sample(frac=1, random_state=1).reset_index()

    # groups = df.groupby('Group')
    # for name, group in groups:
    #     plt.plot(group.eTIV, group.ASF, marker='o', linestyle='', label=name)

    # plt.legend()
    # plt.title('eTIV v. ASF')
    # plt.xlabel('eTIV')
    # plt.ylabel('ASF')
    # plt.show()

    test_len = int(len(df) * test_size)
    train_len = len(df) - test_len
    train = df.head(train_len)
    test = df.tail(test_len)

    y_train = train["Group"]
    y_test = test["Group"]
    x_train = train.loc[:, df.columns != "Group"]
    x_test = test.loc[:, df.columns != "Group"]


    return x_train, y_train, x_test, y_test

def standardize(x_train, x_test):
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test


def visualize(x_train, y_train):
    pass

    plt.scatter(x_train['Age'], y_train)
    plt.show()

    y_train.hist()
    plt.title("Demented v. Non-Demented")
    plt.ylabel("Count")
    plt.show()

    x_train['M/F'].hist()
    plt.show()

    x_train.boxplot()
    plt.show()



def save_conf_matrix(model, y, pred, save_str):
    conf = confusion_matrix(y, pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(conf, display_labels=model.classes_)
    disp.plot()
    plt.title(f'{save_str} Confusion Matrix')
    plt.savefig(f'{save_str}.png')

    precision, recall, fscore, _ = precision_recall_fscore_support(y, pred, labels=model.classes_)

    with open(f'{save_str}.csv', 'w') as f:
        f.write(f',Precision,Recall,FScore\n')
        for p, r, fs, l in zip(precision, recall, fscore, model.classes_):
            f.write(f'{l},{p},{r},{fs}\n')

