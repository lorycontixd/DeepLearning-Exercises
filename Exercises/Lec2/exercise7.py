from matplotlib import pyplot as plt
import numpy as np
from keras import layers, optimizers, losses, models

def true_function(x):
    return np.cos(1.5 * np.pi * x)


if __name__=='__main__':
    x = np.linspace(0, 1, 30)
    y = true_function(x) + np.random.rand() * 0.1
    
    ###### Polynomial fit
    degrees = [1,4,14]
    
    ####  Numpy
    fig, ax = plt.subplots(1, len(degrees), figsize=(15, 5))
    for i, degree in enumerate(degrees):
        ax[i].set_title("Polynomial fit of degree {}".format(degree))
        mymodel = np.poly1d(np.polyfit(x, y, degree))
        
        line = np.linspace(0, max(x), 100)
        ax[i].scatter(x, y, label="Data", color='navy')
        ax[i].plot(line, mymodel(line), label="Fit", color='lightcoral')
        ax[i].legend()
    plt.show()
    
    #### My Model
    
    ## Linear
    print("\n\nLinear Model")
    inp = layers.InputLayer(input_shape=(1,)) 
    out = layers.Dense(1)
    linearmodel = models.Sequential(layers=[inp, out])
    
    linearmodel.compile(loss=losses.mean_squared_error)
    linearmodel.fit(x,y, epochs=500)
    plt.title("Linear fit with custom model")
    plt.scatter(x,y, label="Data")
    plt.plot(x, linearmodel.predict(x), label="Fit", color='red')
    plt.legend()
    plt.show()
    
    ## Polynomial
    print("\n\nPolynomial Model")
    fig, ax = plt.subplots(1, len(degrees), figsize=(15, 6))
    for i, degree in enumerate(degrees):
        model = models.Sequential()
        model.add(layers.Dense(units=200, input_dim=1, activation='relu'))
        model.add(layers.Dense(units=45, activation='relu'))
        model.add(layers.Dense(units=1))
        model.compile(loss=losses.mean_squared_error)
        model.fit(x,y, epochs=500)
        ax[i].set_title("Polynomial fit of degree {}".format(degree))
        ax[i].scatter(x,y, label="Data")
        ax[i].plot(x, model.predict(x.reshape((degree+1,))), label="Fit", color='red')
        ax[i].legend()
    plt.show()