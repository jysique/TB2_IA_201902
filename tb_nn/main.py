from datetime import date

import matplotlib.pyplot as plt
import numpy as np

seed = np.random.seed
import pandas as pd
import seaborn as sns
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import cross_validate
from Adaline import AdalineGD
from MLP import MLP
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
import tkinter.ttk as ttk
import csv

# filename = ""
df = np.zeros((3, 1))

filename = "ph-data.csv"


def Initdata(filename):
    dataframe = pd.read_csv(filename)
    dataframe.columns = ['Blue', 'Green', 'Red', 'Label']
    dataframe['Type'] = dataframe.apply(lambda row: Acidnotrbase(row), axis=1)
    return dataframe


def Acidnotrbase(row):
    if row['Label'] < 7:
        return 0
    elif row['Label'] > 7:
        return 1
    else:
        return 2


def Plot3d():
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    xs = df["Blue"]
    ys = df['Green']
    zs = df['Red']
    ax.scatter(xs, ys, zs, label=df["Type"], cmap='viridis')
    ax.set_xlabel('Blue')
    ax.set_ylabel('Red')
    ax.set_zlabel('Green')
    plt.show()


def PlotHist():
    sns.countplot(x=df["Type"])
    plt.show()


##Split
def ElementsDataframe(dataframe):
    X = dataframe[['Blue', 'Green', 'Red']].values
    y = pd.factorize(df["Type"])[0]
    return X, y


def ParseAdaline():
    inputR = int(numberR.get())
    inputG = int(numberG.get())
    inputB = int(numberB.get())
    return np.array([inputB, inputG, inputR])


def ParseMLP():
    inputR = int(numberR.get())
    inputG = int(numberG.get())
    inputB = int(numberB.get())
    return np.array([[inputB, inputG, inputR]])


def ParseIter():
    inputIter = int(n_iter.get())
    return inputIter


def ParseEta():
    inputEta = float(eta.get())
    return inputEta


def Split(dataframe):
    X, y = ElementsDataframe(dataframe)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    print('#Training data : {}'.format(X_train.shape[0]))
    print('#Testing data : {}'.format(X_test.shape[0]))
    print('Class labels: {} (mapped from {}'.format(np.unique(y), np.unique(df['Type'])))

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    return X_train_std, y_train, X_test_std, y_test


def TestAdeline():
    inputs = ParseAdaline()
    n_iter = ParseIter()
    eta = ParseEta()
    ada = AdalineGD(eta, n_iter)
    X_train_std, y_train, X_test_std, y_test = Split(df)
    ada.fit(X_train_std, y_train)
    # print("RGB entrada:   ", inputR, inputG, inputB)
    activation = ada.activation(inputs)
    print("Test Adeline ---------------------------")
    print("Salida:        ", activation)
    messagebox.showinfo("Clase", ada.think(inputs))


def SplitTrain(dataframe, X):
    X, y = ElementsDataframe(dataframe)
    target = np.zeros((len(X), 3))
    target[np.arange(len(X)), dataframe["Type"]] = 1
    feature_train, feature_test, target_train, target_test = model_selection.train_test_split(X, target, test_size=.25)
    feature_train, feature_validate, target_train, target_validate = model_selection.train_test_split(feature_train,
                                                                                                      target_train,
                                                                                                      test_size=.33)
    return feature_train, feature_validate, target_train, target_validate


def TestMLP():
    inputs = ParseMLP()
    n_iter = ParseIter()
    eta = ParseEta()
    net = MLP(5, eta, outtype="softmax")
    X, y = ElementsDataframe(df)
    print(inputs)
    feature_train, feature_validate, target_train, target_validate = SplitTrain(df, X)
    net.earlystopping(feature_train, target_train, feature_validate, target_validate, n_iter, disp=False)
    print("Test MLP ---------------------------")
    print("Salida:           ", net.predict(inputs))
    max_position = net.predict(inputs).argmax()
    print("Inferencia: clase ", max_position)
    messagebox.showinfo("Clase", net.think(inputs))


def GetFile():
    filename = filedialog.askopenfilename(title="Select file", filetypes=(("csv files", "*.csv"), ("all files", "*.*")))
    return filename


def Print():
    inputR = int(numberR.get())
    inputG = int(numberG.get())
    inputB = int(numberB.get())
    print(np.array([inputB, inputG, inputR]))


window = Tk()
window.geometry("400x400")

filename = GetFile()
df = Initdata(filename)
##print(df)
lbl1 = Label(window, text="Generar Graficos").place(x=20, y=20)
btn3 = Button(window, text="Grafico 3d", command=Plot3d).place(x=130, y=50)
btn2 = Button(window, text="Grafico de Barras", command=PlotHist).place(x=20, y=50)

numberR = StringVar()
numberG = StringVar()
numberB = StringVar()
n_iter = StringVar()
eta = StringVar()

lblRGB = Label(window, text="Inputs").place(x=20, y=80)
lblR = Label(window, text="R   ").place(x=20, y=110)
lblG = Label(window, text="G   ").place(x=20, y=140)
lblB = Label(window, text="B   ").place(x=20, y=170)
lblIter = Label(window, text="Iter").place(x=20, y=200)
lblEta = Label(window, text="Eta  ").place(x=20, y=230)

lblNN = Label(window, text="NN ").place(x=20, y=260)

entry_boxR = Entry(window, textvariable=numberR, width=35).place(x=50, y=110)
entry_boxG = Entry(window, textvariable=numberG, width=35).place(x=50, y=140)
entry_boxB = Entry(window, textvariable=numberB, width=35).place(x=50, y=170)
entry_boxIter = Entry(window, textvariable=n_iter, width=35).place(x=50, y=200)
entry_eta = Entry(window, textvariable=eta, width=35).place(x=50, y=230)

btn2 = Button(window, text="Adaline", command=TestAdeline).place(x=20, y=290)
btn2 = Button(window, text="MLP", command=TestMLP).place(x=20, y=320)

window.title("TB2 Redes Neuronales")
window.mainloop()
