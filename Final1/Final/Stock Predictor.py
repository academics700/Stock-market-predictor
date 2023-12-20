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


def stock_prediction(data_name):
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

    # Train data set
    X_train = X[:split]
    y_train = y[:split]

    # Test data set
    X_test = X[split:]
    y_test = y[split:]

    # Support vector classifier
    cls = SVC().fit(X_train, y_train)

    df['Pred'] = cls.predict(X)

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

    df['day'] = splitted[1].astype('int')
    df['month'] = splitted[0].astype('int')
    df['year'] = splitted[2].astype('int')

    print(df.head())

    df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)
    print(df.head())

    df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

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
    print(cm)

    fig = plt.figure(figsize=(8, 6), facecolor='#222222')
    ax = fig.add_subplot(1, 1, 1, facecolor='#282828')
    ax.grid(True, color='#222222', linewidth=2)
    if image_index == 0:
        ax.plot(df['Cum_Ret'], label='Actual', color='cyan')
    elif image_index == 1:
        ax.plot(df['Cum_Str'], label='Prediction', color='magenta')
    elif image_index == 2:
        ax.plot(df['Cum_Ret'], label='Actual', color='cyan')
        ax.plot(df['Cum_Str'], label='Prediction', color='magenta')
    elif image_index == 3:
        # plt.tight_layout()
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
    elif image_index == 6:
        data_grouped = df.groupby('year').mean()
        data_grouped['Open'].plot.bar(color='#805050')
    elif image_index == 7:
        data_grouped = df.groupby('year').mean()
        data_grouped['High'].plot.bar(color='#805050')
    elif image_index == 8:
        data_grouped = df.groupby('year').mean()
        data_grouped['Low'].plot.bar(color='#805050')
    elif image_index == 9:
        data_grouped = df.groupby('year').mean()
        data_grouped['Close'].plot.bar(color='#805050')
    elif image_index == 10:
        sb.boxplot(df['Open'], notch=True)
    elif image_index == 11:
        sb.boxplot(df['High'], notch=True)
    elif image_index == 12:
        sb.boxplot(df['Low'], notch=True)
    elif image_index == 13:
        sb.boxplot(df['Close'], notch=True)
    elif image_index == 14:
        sb.boxplot(df['Volume'], notch=True)
    elif image_index == 15:
        sb.distplot(df['Open'], color='cyan')
    elif image_index == 16:
        sb.distplot(df['High'], color='cyan')
    elif image_index == 17:
        sb.distplot(df['Low'], color='cyan')
    elif image_index == 18:
        sb.distplot(df['Close'], color='cyan')
    elif image_index == 19:
        sb.distplot(df['Volume'], color='cyan')
    ax.spines['bottom'].set_color('#222222')
    ax.spines['top'].set_color('#222222')
    ax.spines['left'].set_color('#222222')
    ax.spines['right'].set_color('#222222')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white', length=0)
    ax.tick_params(axis='y', colors='white', length=0)
    ax.tick_params(color='white', labelcolor='white')
    l = plt.legend(framealpha=0)
    for text in l.get_texts():
        text.set_color("white")
    '''if image_index == 3:
        plt.savefig('fig.png', dpi=180, transparent=True)
    else:'''
    plt.savefig('fig.png', dpi=180)


# GUI

import sys
from PyQt5.QtCore import Qt, QUrl
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


def factory(dummy):
    def _pic_():
        global data_index, widget_index
        data_index = dummy
        widget_index = 1
        stock_prediction('Data/CSV/' + data_arr[dummy])
        win.call_widget()

    return _pic_


pic_functions = []
for i in range(len(data_arr)):
    pic_functions.append(factory(i))


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
                    {border-image:url("Images/Icons/' + image[i] + '");}\
                    QPushButton::hover\
                    {border:5px solid black;}'

            inner_grid = QGridLayout()
            inner_grid.setColumnStretch(0, 1)
            inner_grid.setColumnStretch(1, 5)
            g_box = QGroupBox("")
            g_box.setCheckable(False)
            g_box.setLayout(inner_grid)
            g_box.setStyleSheet("QGroupBox{background-color:qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,\
                                    stop: 0 #2b2b2b, stop: 1 #2f4f4f);}")

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
        grid = QGridLayout()
        grid.setSpacing(16)

        groupbox = QGroupBox("")
        groupbox.setMinimumSize(200, 600)
        groupbox.setStyleSheet('border:0px; border-radius:10px;')
        groupbox.setCheckable(False)
        vbox = QVBoxLayout()

        inner_grid1 = QGridLayout()
        inner_grid1.setColumnStretch(0, 1)
        inner_grid1.setColumnStretch(1, 12)
        g_box1 = QGroupBox("")
        g_box1.setCheckable(False)
        g_box1.setLayout(inner_grid1)

        g_box1.setStyleSheet("QGroupBox{background:rgba(100, 100, 100, 0.08);}")

        lbl1 = QLabel(self)
        lbl1.setWordWrap(True)
        lbl1.setAlignment(Qt.AlignLeft)
        lbl1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # lbl1.setStyleSheet('background: #909090')
        lbl1.setStyleSheet('QLabel{color:#afafaf; font-size:48px; padding-left:10px;\
                                          background : rgba(255, 255, 255, 0.0); font-family: BLACK;}')
        lbl1.setText(data_name[data_index])

        lbl0 = QLabel(self)
        lbl0.setWordWrap(True)
        lbl0.setAlignment(Qt.AlignLeft)
        lbl0.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lbl0.setStyleSheet("border-radius:10px; border-image:url('Images/Icons/" + image[data_index] + "')")
        lbl0.setText("")

        inner_grid1.addWidget(lbl0, 0, 0, 1, 1)
        inner_grid1.addWidget(lbl1, 0, 1, 1, 5)
        vbox.addWidget(g_box1, 3)

        inner_grid2 = QGridLayout()
        inner_grid2.setColumnStretch(0, 1)
        inner_grid2.setColumnStretch(1, 1)
        inner_grid2.setColumnStretch(2, 1)
        inner_grid2.setColumnStretch(3, 1)
        inner_grid2.setColumnStretch(4, 1)
        g_box2 = QGroupBox("")
        g_box2.setCheckable(False)
        g_box2.setLayout(inner_grid2)
        g_box2.setStyleSheet("background: #191919;")

        qss_btn = "QPushButton{background-color:qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,\
                                stop: 0 #222222, stop: 1 #444c3f); color: #cfcf2f;\
                                    border: 1px solid #151515;font-family: Antonio; \
                                        font-size:17px;}\
                                        QPushButton::hover{background:#444c3f;color:white;}"

        qss_btn1 = "QPushButton{background-color:qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,\
                                        stop: 0 #222222, stop: 1 #564c4d); color: #40b5ad;\
                                            border: 1px solid #151515;font-family: Antonio;\
                                               border-radius: 6px;\
                                                font-size: 15px;}\
                                                QPushButton::hover{background:#373737;color:white;}"

        btn1 = QPushButton(self)
        btn1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn1.setText('Actual')
        btn1.setStyleSheet(qss_btn)
        btn1.clicked.connect(self.action_btn1)

        btn2 = QPushButton(self)
        btn2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn2.setText('Prediction')
        btn2.setStyleSheet(qss_btn)
        btn2.clicked.connect(self.action_btn2)

        btn3 = QPushButton(self)
        btn3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn3.setText('Both')
        btn3.setStyleSheet(qss_btn)
        btn3.clicked.connect(self.action_btn3)

        btn4 = QPushButton(self)
        btn4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn4.setText('Correlation')
        btn4.setStyleSheet(qss_btn)
        btn4.clicked.connect(self.action_btn4)

        btn5 = QPushButton(self)
        btn5.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn5.setText('Confusion matrix')
        btn5.setStyleSheet(qss_btn)
        btn5.clicked.connect(self.action_btn5)

        inner_grid2.addWidget(btn1, 0, 0, 1, 1)
        inner_grid2.addWidget(btn2, 0, 1, 1, 1)
        inner_grid2.addWidget(btn3, 0, 2, 1, 1)
        inner_grid2.addWidget(btn4, 0, 3, 1, 1)
        inner_grid2.addWidget(btn5, 0, 4, 1, 1)
        vbox.addWidget(g_box2, 2)

        inner_grid3 = QGridLayout()
        inner_grid3.setColumnStretch(0, 5)
        inner_grid3.setColumnStretch(1, 22)
        g_box3 = QGroupBox("")
        g_box3.setCheckable(False)
        g_box3.setLayout(inner_grid3)

        lbl2 = QLabel(self)
        lbl2.setWordWrap(True)
        lbl2.setAlignment(Qt.AlignLeft)
        lbl2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # lbl2.setStyleSheet('background: #909090')
        lbl2.setStyleSheet("border-radius:0px; border-image:url('fig.png')")
        lbl2.setText("")

        inner_grid4 = QGridLayout()
        for i in range(15):
            inner_grid4.setRowStretch(i, 1)

        g_box4 = QGroupBox("")
        g_box4.setCheckable(False)
        g_box4.setLayout(inner_grid4)
        g_box4.setStyleSheet("background: #191919;")

        btn6 = QPushButton(self)
        btn6.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn6.setText('Target Pie')
        btn6.setStyleSheet(qss_btn1)
        btn6.clicked.connect(self.action_btn6)

        btn7 = QPushButton(self)
        btn7.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn7.setText('Open barplot')
        btn7.setStyleSheet(qss_btn1)
        btn7.clicked.connect(self.action_btn7)

        btn8 = QPushButton(self)
        btn8.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn8.setText('High barplot')
        btn8.setStyleSheet(qss_btn1)
        btn8.clicked.connect(self.action_btn8)

        btn9 = QPushButton(self)
        btn9.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn9.setText('Low Barplot')
        btn9.setStyleSheet(qss_btn1)
        btn9.clicked.connect(self.action_btn9)

        btn10 = QPushButton(self)
        btn10.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn10.setText('Close barplot')
        btn10.setStyleSheet(qss_btn1)
        btn10.clicked.connect(self.action_btn10)

        btn11 = QPushButton(self)
        btn11.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn11.setText('Open boxplot')
        btn11.setStyleSheet(qss_btn1)
        btn11.clicked.connect(self.action_btn11)

        btn12 = QPushButton(self)
        btn12.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn12.setText('High boxplot')
        btn12.setStyleSheet(qss_btn1)
        btn12.clicked.connect(self.action_btn12)

        btn13 = QPushButton(self)
        btn13.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn13.setText('Low boxplot')
        btn13.setStyleSheet(qss_btn1)
        btn13.clicked.connect(self.action_btn13)

        btn14 = QPushButton(self)
        btn14.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn14.setText('Close boxplot')
        btn14.setStyleSheet(qss_btn1)
        btn14.clicked.connect(self.action_btn14)

        btn15 = QPushButton(self)
        btn15.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn15.setText('Volume boxplot')
        btn15.setStyleSheet(qss_btn1)
        btn15.clicked.connect(self.action_btn15)

        btn16 = QPushButton(self)
        btn16.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn16.setText('Open distribution')
        btn16.setStyleSheet(qss_btn1)
        btn16.clicked.connect(self.action_btn16)

        btn17 = QPushButton(self)
        btn17.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn17.setText('High distribution')
        btn17.setStyleSheet(qss_btn1)
        btn17.clicked.connect(self.action_btn17)

        btn18 = QPushButton(self)
        btn18.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn18.setText('Low distribution')
        btn18.setStyleSheet(qss_btn1)
        btn18.clicked.connect(self.action_btn18)

        btn19 = QPushButton(self)
        btn19.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn19.setText('Close distribution')
        btn19.setStyleSheet(qss_btn1)
        btn19.clicked.connect(self.action_btn19)

        btn20 = QPushButton(self)
        btn20.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        btn20.setText('Volume distribution')
        btn20.setStyleSheet(qss_btn1)
        btn20.clicked.connect(self.action_btn20)

        inner_grid4.addWidget(btn6, 0, 0, 1, 1)
        inner_grid4.addWidget(btn7, 1, 0, 1, 1)
        inner_grid4.addWidget(btn8, 2, 0, 1, 1)
        inner_grid4.addWidget(btn9, 3, 0, 1, 1)
        inner_grid4.addWidget(btn10, 4, 0, 1, 1)
        inner_grid4.addWidget(btn11, 5, 0, 1, 1)
        inner_grid4.addWidget(btn12, 6, 0, 1, 1)
        inner_grid4.addWidget(btn13, 7, 0, 1, 1)
        inner_grid4.addWidget(btn14, 8, 0, 1, 1)
        inner_grid4.addWidget(btn15, 9, 0, 1, 1)
        inner_grid4.addWidget(btn16, 10, 0, 1, 1)
        inner_grid4.addWidget(btn17, 11, 0, 1, 1)
        inner_grid4.addWidget(btn18, 12, 0, 1, 1)
        inner_grid4.addWidget(btn19, 13, 0, 1, 1)
        inner_grid4.addWidget(btn20, 14, 0, 1, 1)

        inner_grid3.addWidget(g_box4, 0, 0, 1, 1)
        inner_grid3.addWidget(lbl2, 0, 1, 1, 5)
        vbox.addWidget(g_box3, 18)

        groupbox.setLayout(vbox)
        grid.addWidget(groupbox)
        self.setLayout(grid)

    def action_btn1(self):
        global image_index
        image_index = 0
        new_action = pic_functions
        new_action[data_index]()

    def action_btn2(self):
        global image_index
        image_index = 1
        new_action = pic_functions
        new_action[data_index]()

    def action_btn3(self):
        global image_index
        image_index = 2
        new_action = pic_functions
        new_action[data_index]()

    def action_btn4(self):
        global image_index
        image_index = 3
        new_action = pic_functions
        new_action[data_index]()

    def action_btn5(self):
        global image_index
        image_index = 4
        new_action = pic_functions
        new_action[data_index]()

    def action_btn6(self):
        global image_index
        image_index = 5
        new_action = pic_functions
        new_action[data_index]()

    def action_btn7(self):
        global image_index
        image_index = 6
        new_action = pic_functions
        new_action[data_index]()

    def action_btn8(self):
        global image_index
        image_index = 7
        new_action = pic_functions
        new_action[data_index]()

    def action_btn9(self):
        global image_index
        image_index = 8
        new_action = pic_functions
        new_action[data_index]()

    def action_btn10(self):
        global image_index
        image_index = 9
        new_action = pic_functions
        new_action[data_index]()

    def action_btn11(self):
        global image_index
        image_index = 10
        new_action = pic_functions
        new_action[data_index]()

    def action_btn12(self):
        global image_index
        image_index = 11
        new_action = pic_functions
        new_action[data_index]()

    def action_btn13(self):
        global image_index
        image_index = 12
        new_action = pic_functions
        new_action[data_index]()

    def action_btn14(self):
        global image_index
        image_index = 13
        new_action = pic_functions
        new_action[data_index]()

    def action_btn15(self):
        global image_index
        image_index = 14
        new_action = pic_functions
        new_action[data_index]()

    def action_btn16(self):
        global image_index
        image_index = 15
        new_action = pic_functions
        new_action[data_index]()

    def action_btn17(self):
        global image_index
        image_index = 16
        new_action = pic_functions
        new_action[data_index]()

    def action_btn18(self):
        global image_index
        image_index = 17
        new_action = pic_functions
        new_action[data_index]()

    def action_btn19(self):
        global image_index
        image_index = 18
        new_action = pic_functions
        new_action[data_index]()

    def action_btn20(self):
        global image_index
        image_index = 19
        new_action = pic_functions
        new_action[data_index]()


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
