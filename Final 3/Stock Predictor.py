# Machine learning
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score

# For data manipulation
import pandas as pd
import numpy as np
import seaborn as sb

# To plot
import matplotlib.pyplot as plt

# To ignore warnings
import warnings

warnings.filterwarnings("ignore")

data_arr = ['RELIANCE.CSV', 'TSLA.csv', 'MSFT.csv', 'TCS.NS.csv', 'IBN.csv',
            'CXO.csv', 'DG.csv', 'EA.csv', 'GOOGL.csv', 'IBM.csv', 'MAC.csv',
            'NOC.csv']


def data_function(data_name):
    df = pd.read_csv(data_name)
    print(df)

    # Changes The Date column as index columns
    df.index = pd.to_datetime(df['Date'])
    print(df)

    splitted = df['Date'].str.split('-', expand=True)

    # drop The original date column
    df = df.drop(['Date'], axis='columns')
    print(df)

    # Create predictor variables
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low

    # Store all predictor variables in a variable X
    X = df[['Open-Close', 'High-Low']]
    print(X.head())

    # Target variables
    y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    print(y)

    split_percentage = 0.8
    split = int(split_percentage * len(df))

    df['day'] = splitted[1].astype('int')
    df['month'] = splitted[0].astype('int')
    df['year'] = splitted[2].astype('int')
    print(df.head())

    df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)
    print(df.head())

    df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    # Train data set
    X_train = X[:split]
    y_train = y[:split]

    # Test data set
    X_test = X[split:]
    y_test = y[split:]

    #def training_model(df, X_train, y_train, X):
    # Support vector classifier
    cls = SVC().fit(X_train, y_train)

    df['Pred'] = cls.predict(X)
    print(df['Pred'])

    # Calculate daily returns
    df['Return'] = df.Close.pct_change()

    # Calculate strategy returns
    df['Str_Ret'] = df.Return * df.Pred.shift(1)

    # Calculate Cumulutive returns
    df['Cum_Ret'] = df['Return'].cumsum()
    print(df)

    # Plot Strategy Cumulative returns
    df['Cum_Str'] = df['Str_Ret'].cumsum()
    print(df)



    features = df[['Open-Close', 'High-Low', 'is_quarter_end']]
    target = df['target']

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    X_train, X_valid, Y_train, Y_valid = train_test_split(
        features, target, test_size=0.1, random_state=2022)
    print(X_train.shape, X_valid.shape)

    models = [LogisticRegression(), SVC(
        kernel='poly', probability=True), XGBClassifier()]

    for i in range(3):
        models[i].fit(X_train, Y_train)

        print(f'{models[i]} " ')
        print('Training Accuracy : ', metrics.roc_auc_score(Y_train, models[i].predict_proba(X_train)[:, 1]))
        print('Validation Accuracy : ', metrics.roc_auc_score(Y_valid, models[i].predict_proba(X_valid)[:, 1]))
        print()

    cls.fit(X_valid, Y_valid)
    predictions = cls.predict(X_valid)
    cm = metrics.confusion_matrix(Y_valid, predictions)

    fig = plt.figure(figsize=(8, 6), facecolor='#222222')
    ax = fig.add_subplot(1, 1, 1, facecolor='#282828')
    ax.grid(True, color='#222222', linewidth=2)
    df_col = ['Open', 'High', 'Close', 'Low', 'Volume']
    if image_index == 0:
        ax.plot(df['Cum_Ret'], label='Actual', color='cyan')
    elif image_index == 1:
        ax.plot(df['Cum_Str'], label='Prediction', color='magenta')
    elif image_index == 2:
        ax.plot(df['Cum_Ret'], label='Actual', color='cyan')
        ax.plot(df['Cum_Str'], label='Prediction', color='magenta')
    elif image_index == 3:
        sb.heatmap(df.corr() > 0.9, annot=True, cbar=False, cmap="RdGy"
                   , linewidth=2, linecolor='#222222')
    elif image_index == 4:
        x = sb.heatmap(cm, annot=True, cmap='viridis')
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
    elif image_index == 5:
        colors = ['cyan', 'magenta']
        patches, texts, autotexts = ax.pie(df['target'].value_counts().values,
                                           labels=[0, 1], autopct='%1.1f%%', colors=colors,
                                           textprops={'size': 'x-large'},
                                           wedgeprops={'linewidth': 5.0, 'edgecolor': '#222222'},
                                           explode=(0, 0.03))
        for autotext in autotexts:
            autotext.set_color('black')
        for i, patch in enumerate(patches):
            texts[i].set_color(patch.get_facecolor())
    if image_index >= 6 & image_index <= 9:
        for _ in range(6, 10):
            if image_index == _:
                data_grouped = df.groupby('year').mean()
                data_grouped[df_col[_ - 6]].plot.bar(color='#daa520', width=0.2)
    if image_index >= 10 & image_index <= 14:
        for _ in range(10, 15):
            if image_index == _:
                c = '#9acd32'
                ax.boxplot(df[df_col[_ - 10]], notch=True, patch_artist=True,
                           boxprops=dict(facecolor=c, color=c), capprops=dict(color=c),
                           whiskerprops=dict(color=c), widths=0.6,
                           flierprops=dict(color=c, markeredgecolor=c),
                           medianprops=dict(color=c))
    if image_index >= 15 & image_index <= 19:
        for _ in range(15, 20):
            if image_index == _:
                sb.distplot(df[df_col[_ - 15]], color='cyan')
    for _ in ['top', 'bottom', 'left', 'right']:
        ax.spines[_].set_color('#222222')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white', length=0)
    ax.tick_params(axis='y', colors='white', length=0)
    l = plt.legend(framealpha=0)
    for text in l.get_texts():
        text.set_color("white")
    plt.savefig('fig.png', dpi=180)


# GUI

import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QWidget, QMainWindow, QLabel,
                             QPushButton, QSizePolicy, QDesktopWidget, QGridLayout,
                             QVBoxLayout, QAction, QScrollArea, QFrame,
                             QGroupBox, QHBoxLayout)
from PyQt5.QtGui import QIcon

widget_index = 0
data_index = 0
image_index = 2

dfr = pd.read_excel('Data/Data.xlsx', sheet_name='home')
image = dfr['image'].tolist()
data_name = dfr['Name'].tolist()
info = dfr['info'].tolist()


def factory1(dummy):
    def _pic_():
        global data_index, widget_index
        data_index = dummy
        widget_index = 1
        stock_prediction('Data/CSV/' + data_arr[dummy])
        win.call_widget()

    return _pic_


pic_functions = []
for i in range(len(data_arr)):
    pic_functions.append(factory1(i))


def factory2(dummy):
    def _action_():
        global image_index, data_index
        image_index = dummy
        new_action = pic_functions
        new_action[data_index]()

    return _action_


action_functions = []
for i in range(20):
    action_functions.append(factory2(i))


class __HomeWidget__(QWidget):

    def __init__(self):
        super().__init__()
        self.func_widget()

    def func_widget(self):
        self.action = pic_functions

        grid = QGridLayout()
        grid.setSpacing(16)

        scroll = QScrollArea()
        scroll.setStyleSheet("border:0px;")
        scroll.setFrameShape(QFrame.Box)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)

        groupbox = QGroupBox("")
        groupbox.setMinimumSize(200, 1000)
        groupbox.setStyleSheet('border:0px; border-radius:10px;')
        groupbox.setCheckable(False)
        vbox = QVBoxLayout()
        groupbox.setLayout(vbox)
        grid.addWidget(scroll, 0, 0, 3, 1)
        grid.setContentsMargins(0, 0, 0, 0)

        self.setLayout(grid)

        for i in range(len(data_arr)):
            qss = 'QPushButton\
                    {border-image:url("Images/Icons/' + image[i] + '"); background: rgba(0, 0, 0, 0);}\
                    QPushButton::hover\
                    {border:5px solid black;}'

            inner_grid = QGridLayout()
            inner_grid.setColumnStretch(0, 1)
            inner_grid.setColumnStretch(1, 5)
            g_box = QGroupBox("")
            g_box.setCheckable(False)
            g_box.setLayout(inner_grid)
            g_box.setStyleSheet("QGroupBox{background-color:qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,\
                                    stop: 0 #2b2b2b, stop: 1 #070720);}")

            lbl = QLabel(self)
            lbl.setWordWrap(True)
            lbl.setAlignment(Qt.AlignLeft)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            lbl.setStyleSheet('QLabel{color:#ffff99; font-size:24px; padding-left:10px;\
                                  background : rgba(255, 255, 255, 0.0)}\
                                  QLabel::hover{background:rgba(0, 0, 0, 0.08)}')
            lbl.setText('<font face=Azonix>' + data_name[i] + '</font><br><br>\
            <font size=20px color=#bfbf99>' + info[i] + '</font>')

            btn = QPushButton(self)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            btn.setStyleSheet(qss)
            btn.clicked.connect(self.action[i])

            inner_grid.addWidget(btn, 0, 0, 1, 1)
            inner_grid.addWidget(lbl, 0, 1, 1, 5)
            vbox.addWidget(g_box)

        scroll.setWidget(groupbox)


class __PredWidget__(QWidget):

    def __init__(self):
        super().__init__()
        self.func_widget()

    def func_widget(self):
        self.action = action_functions

        btn_txt = ['Actual', 'Prediction', 'Both', 'Correlation', 'Confusion Matrix',
                   'Target Pie', 'Open Barplot', 'High Barplot', 'Low Barplot', 'Close Barplot',
                   'Open Boxplot', 'High Boxplot', 'Low Boxplot', 'Close Boxplot', 'Volume Boxplot',
                   'Open Distribution', 'High Distribution', 'Low Distribution', 'Close Distribution',
                   'Volume Distribution']

        grid = QGridLayout()
        grid.setSpacing(16)
        grid.setContentsMargins(0, 0, 0, 0)

        groupbox = QGroupBox("")
        groupbox.setMinimumSize(200, 600)
        groupbox.setStyleSheet('border:0px; border-radius:10px;')
        groupbox.setCheckable(False)
        # groupbox.setStyleSheet('background: white;')
        # groupbox.setContentsMargins(0, 0, 0, 0)
        vbox = QVBoxLayout()
        vbox.setSpacing(0)
        vbox.setContentsMargins(0, 0, 0, 0)

        inner_grid1 = QGridLayout()
        inner_grid1.setColumnStretch(0, 1)
        inner_grid1.setColumnStretch(1, 12)
        inner_grid1.setSpacing(0)
        g_box1 = QGroupBox("")
        g_box1.setCheckable(False)
        g_box1.setLayout(inner_grid1)
        g_box1.setStyleSheet('QGroupBox{background-color:qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,\
                                stop: 0 #161616, stop: 1 #070718); border-radius: 0px; color: #ffffff;}')

        lbl1 = QLabel(self)
        lbl1.setWordWrap(True)
        lbl1.setAlignment(Qt.AlignLeft)
        lbl1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # lbl1.setStyleSheet('background: #909090')
        lbl1.setStyleSheet('QLabel{font-size:48px; padding-left:10px; color: #ffe68c;\
                                          background : rgba(255, 255, 255, 0.0); font-family: Anurati;}')
        lbl1.setText(data_name[data_index])

        lbl0 = QLabel(self)
        lbl0.setWordWrap(True)
        lbl0.setAlignment(Qt.AlignLeft)
        lbl0.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lbl0.setStyleSheet("border-radius:10px; background: #000000; \
                            border-image:url('Images/Icons/" + image[data_index] + "');")
        lbl0.setText("")

        inner_grid1.addWidget(lbl0, 0, 0, 1, 1)
        inner_grid1.addWidget(lbl1, 0, 1, 1, 5)
        vbox.addWidget(g_box1, 3)

        inner_grid2 = QGridLayout()
        inner_grid2.setSpacing(5)
        # inner_grid2.setContentsMargins(0, 0, 0, 0)

        g_box2 = QGroupBox("")
        g_box2.setCheckable(False)
        g_box2.setLayout(inner_grid2)
        g_box2.setStyleSheet("background: #161616; border-raidius: 0px;")

        qss_btn = "QPushButton{background: rgba(255, 255, 255, 0.08); color: #bbbbbb;\
                                    border: 1px solid #151515;font-family: Bahnschrift; \
                                        font-size:20px; border-radius: 3px;}\
                                        QPushButton::hover{background:#191940;color:white;}"

        qss_btn1 = "QPushButton{background: rgba(255, 255, 255, 0.08); color: #bbbbbb;\
                                            border: 1px solid #151515;font-family: Bahnschrift;\
                                               border-radius: 2px;\
                                                font-size: 18px;}\
                                                QPushButton::hover{background:#373737;color:white;}"

        for i in range(5):
            inner_grid2.setColumnStretch(i, 1)
            btn = QPushButton(self)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            btn.setText(btn_txt[i])
            btn.setStyleSheet(qss_btn)
            btn.clicked.connect(self.action[i])
            inner_grid2.addWidget(btn, 0, i, 1, 1)

        vbox.addWidget(g_box2, 2)

        inner_grid3 = QGridLayout()
        inner_grid3.setColumnStretch(0, 6)
        inner_grid3.setColumnStretch(1, 22)
        inner_grid3.setSpacing(0)
        inner_grid3.setContentsMargins(0, 0, 0, 0)
        g_box3 = QGroupBox("")
        g_box3.setCheckable(False)
        g_box3.setLayout(inner_grid3)
        g_box3.setStyleSheet("border-radius: 0px; border-left: 5px solid #161616;\
                             border-right: 15px solid #161616; border-bottom: 20px solid #161616;")

        lbl2 = QLabel(self)
        lbl2.setWordWrap(True)
        lbl2.setAlignment(Qt.AlignLeft)
        lbl2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # lbl2.setStyleSheet('background: #909090')
        lbl2.setStyleSheet("border-radius:0px; border-image:url('fig.png')")
        lbl2.setText("")

        inner_grid4 = QGridLayout()
        inner_grid4.setContentsMargins(0, 0, 0, 0)
        inner_grid4.setSpacing(0)
        g_box4 = QGroupBox("")
        g_box4.setCheckable(False)
        g_box4.setLayout(inner_grid4)
        g_box4.setStyleSheet("background: #161616; border-radius: 0px; border-bottom: 0px;")

        for i in range(5, 20):
            inner_grid4.setRowStretch(i, 1)
            btn = QPushButton(self)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            btn.setText(btn_txt[i])
            btn.setStyleSheet(qss_btn1)
            btn.clicked.connect(self.action[i])
            inner_grid4.addWidget(btn, i, 0, 1, 1)

        inner_grid3.addWidget(g_box4, 0, 0, 1, 1)
        inner_grid3.addWidget(lbl2, 0, 1, 1, 5)
        vbox.addWidget(g_box3, 18)

        groupbox.setLayout(vbox)
        grid.addWidget(groupbox)
        self.setLayout(grid)


class __main__window__(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setMinimumSize(1000, 800)
        self.setWindowTitle('Stock Market Prediction')
        self.setWindowIcon(QIcon('Images/icon.png'))
        self.setWindowOpacity(0.95)
        self.statusbar = self.statusBar()
        self.statusBar().showMessage('v 0.0.1')
        self.back = QAction(QIcon('Images/back.png'), 'Back', self)
        self.back.setShortcut('Backspace')
        self.back.triggered.connect(self.action_back)

        self.toolbar = self.addToolBar('Back')
        self.toolbar.addAction(self.back)
        self.toolbar.setStyleSheet("QToolBar{background-color:qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,\
                                stop: 0 #070718, stop: 1 #070718);}")
        self.center()
        self.call_widget()
        self.show()

    def call_widget(self):
        global widget_index

        if widget_index == 0:
            widget = __HomeWidget__()
        elif widget_index == 1:
            widget = __PredWidget__()
        self.setCentralWidget(widget)
        self.hbox = QHBoxLayout()
        self.centralWidget().setLayout(self.hbox)

    def action_back(self):
        global widget_index, image_index
        image_index = 2
        if widget_index == 0:
            widget_index = 0
        elif widget_index == 1:
            widget_index = 0
            self.call_widget()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


def main():
    global win
    app = QApplication(sys.argv)
    qss = "Data/stylesheet.qss"
    with open(qss, "r") as fh:
        app.setStyleSheet(fh.read())
    win = __main__window__()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
