from flask import Flask, jsonify, render_template, request,Response,send_from_directory,session,make_response
from flask_login import LoginManager
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib as mpl
import io
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.cluster import KMeans
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.figure import Figure
from datetime import timedelta

import os
from werkzeug.utils import secure_filename
import seaborn as sns


import os




ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','csv'}

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=0) 
LoginManager.session_protection = "strong"
app.secret_key = os.urandom(12)

def create_time_steps(length):
        time_steps = []
        for i in range(-length, 0, 1):
            time_steps.append(i)
        return time_steps              

def univariate_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i+target_size])
  return np.array(data), np.array(labels)


def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    print(time_steps)
    if delta:
        future = delta
    else:
        future = 0
    plt.title(title)
    
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.grid(color='lightgrey')
    plt.xlim([time_steps[0], (future+5)*1])
    plt.xlabel('Time-Step')
    return plt

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload',methods=[ "GET",'POST'])
def upload():
    print("上船中....")
    uploaded_file=request.files['inputFile']
    # uploaded_file = request.files.getlist("inputFile")    
    
    if uploaded_file.filename == '':
        print('No selected file')    
        return ("",204)

    if uploaded_file and allowed_file(uploaded_file.filename):
        filename = secure_filename(uploaded_file.filename)
        print(uploaded_file.filename)
        print (filename)
        uploaded_file.save(os.path.join("./", uploaded_file.filename))
        try:
            session['uploadFile']=uploaded_file.filename
        except:
            pass
        
        return ("",204)


@app.route('/gotoAI-Lstm',methods=[ "GET",'POST'])
def gotoAI():
    print("gotoAI")
    # mpl.rcParams['figure.figsize'] = (8, 6)
    # mpl.rcParams['axes.grid'] = False

    df = pd.read_csv("./jena_climate_2009_2016.csv")
    # print(df.head())
    TRAIN_SPLIT = 300000
    
    uni_data = df['T (degC)']
    uni_data.index = df['Date Time']
    uni_data.head()
    uni_data = uni_data.values
    uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
    uni_train_std = uni_data[:TRAIN_SPLIT].std()
    uni_data = (uni_data-uni_train_mean)/uni_train_std
    univariate_past_history = 20
    univariate_future_target = 0

    x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)
    x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)
    # print ('Single window of past history')
    # print (x_train_uni[0])
    # print ('\n Target temperature to predict')
    # print (y_train_uni[0])
                         


    BATCH_SIZE = 256
    BUFFER_SIZE = 10000

    train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
    train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
    val_univariate = val_univariate.batch(BATCH_SIZE).repeat()    

    simple_lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
        tf.keras.layers.Dense(1)
    ])

    simple_lstm_model.compile(optimizer='adam', loss='mae')
    for x, y in val_univariate.take(1):
        print(simple_lstm_model.predict(x).shape)

    EVALUATION_INTERVAL = 200
    EPOCHS = 10

    simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50)
    

    simple_lstm_model.evaluate(x_val_uni,  y_val_uni,verbose=2)
    
    

    for x, y in val_univariate.take(0):
        show_plot([x[0].numpy(), y[0].numpy(),
                    simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
    
    global LSTM_PLT
    LSTM_PLT=show_plot([x[5].numpy(), y[5].numpy(),simple_lstm_model.predict(x)[5]], 0, 'Simple LSTM model')
       


    return render_template('gotoAI.html')

@app.route('/plot.png',methods=[ "GET",'POST'])
def plot_png():
    try:    
        global LSTM_PLT
        
        fig = LSTM_PLT.gcf()
        # fig = plt.gcf()
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')
    except:
        return send_from_directory('./static',"loloading.png")    


@app.route('/gotoAI_Kmeans',methods=[ "GET",'POST'])
def gotoAI_Kmeans():
    dataset = pd.read_csv('./Mall_Customers.csv')
    print(dataset.head(10))
    print(dataset.shape)

    X= dataset.iloc[:, [3,4]].values
    wcss=[]

    for i in range(1,11):
        kmeans = KMeans(n_clusters= i, init='k-means++')
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    kmeansmodel = KMeans(n_clusters= 5, init='k-means++')
    y_kmeans= kmeansmodel.fit_predict(X) 

    global Kmeansfig
    Kmeansfig = Figure()
    axis = Kmeansfig.add_subplot(1, 1, 1)
    axis.grid(color='lightgrey')
    axis.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
    axis.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
    axis.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
    axis.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
    axis.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
    axis.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
    axis.set_title("K-means demo")
    axis.legend()
    


    
    return render_template('gotoAI.html')


@app.route('/gotoAI_Kmeans.png',methods=["GET",'POST'])
def gotoAIKmeansPNG():
    try:
        global Kmeansfig


        output = io.BytesIO()
        FigureCanvas(Kmeansfig).print_png(output)
        
        return Response(output.getvalue(), mimetype="image/png")

    except:
        return send_from_directory('./static',"loloading.png")


@app.route('/LinearRegression',methods=[ "GET",'POST'])
def LinearRegression():
    print("LinearRegression")
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
    diabetes_X = diabetes_X[:, 2, np.newaxis]

    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]
    regr = linear_model.LinearRegression()
    regr.fit(diabetes_X_train, diabetes_y_train)
    diabetes_y_pred = regr.predict(diabetes_X_test)
    print('Coefficients: \n', regr.coef_)
    print('Mean squared error: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    print('Coefficient of determination: %.2f'
      % r2_score(diabetes_y_test, diabetes_y_pred))  

    global LinearRegressionfig
    LinearRegressionfig= Figure()
    axis = LinearRegressionfig.add_subplot(1, 1, 1) 

    axis.scatter(diabetes_X_test, diabetes_y_test,  color='black')
    
    axis.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
    axis.set_title("linear regression demo")
    axis.grid(color='lightgrey',alpha=0.7)



    return render_template('gotoAI.html')

@app.route('/gotoAI_LinearRegression.png',methods=["GET",'POST'])
def gotoAILinearRegressionPNG():
    try:
        global LinearRegressionfig


        output = io.BytesIO()
        FigureCanvas(LinearRegressionfig).print_png(output)
        
        return Response(output.getvalue(), mimetype="image/png")

    except:
        return send_from_directory('./static',"loloading.png")


@app.route('/CustomDataBar.png',methods=["GET",'POST'])
def CustomDataBarPNG():
    try:
        print(session['uploadFile'])
        df=pd.read_csv(session['uploadFile'])
        Barfig= Figure()
        axis = Barfig.add_subplot(1,1,1)
        df.plot(kind='bar', ax=axis)

        axis.legend(loc='best')
        

        
        output = io.BytesIO()
        FigureCanvas(Barfig).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')
    except:
        return send_from_directory('./static',"loloading.png")




@app.route('/CustomDataLine.png',methods=["GET",'POST'])
def CustomDataLinePNG():

    try:

        print(session['uploadFile'])
        df=pd.read_csv(session['uploadFile'])
        CLinefig= Figure()
        axis = CLinefig.add_subplot(1,1,1)
        df.plot(kind='line', ax=axis)

        axis.legend(loc='best')
        

        
        output = io.BytesIO()
        FigureCanvas(CLinefig).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')

    except:
        print("there is no data to line plot yet")

    
        return send_from_directory('./static',"loloading.png")



@app.route('/CustomDataBox.png',methods=["GET",'POST'])
def CustomDataBoxPNG():

    try:

        print(session['uploadFile'])
        df=pd.read_csv(session['uploadFile'])
        Boxfig= Figure()
        axis = Boxfig.add_subplot(1,1,1)
        df.plot(kind='box', ax=axis)

        axis.legend(loc='best')
        

        
        output = io.BytesIO()
        FigureCanvas(Boxfig).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')

    except:
        print("there is no data to box plot yet")

    
        return send_from_directory('./static',"loloading.png")





@app.route('/',methods=[ "GET",'POST'])
def index():
    return render_template('gotoAI.html')
	





if __name__ == "__main__":
    # app.run(host="0.0.0.0",port=12000,threaded=True)
    app.run(debug=True,port=5000)