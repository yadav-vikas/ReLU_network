# ReLU neural network

### This is in development phase it may not work.
rectified linear activation function

## What is ReLU ?

ReLU is defined as g(x) = max(0,x). It is 0 when x is negative and equal to x when positive. Due to it’s lower saturation region, it is highly trainable and decreases the cost function far more quickly than sigmoid.

### acitation function

Inline-style: 
![alt text](https://images.app.goo.gl/9umj1oAi4Y5DmecV6 "activation functions")

### ReLU acitation function

Inline-style: 
![alt text](https://images.app.goo.gl/gkYh64M1wRKtjGuSA "ReLU activation functions")

note:please check out links if images are broken
https://images.app.goo.gl/9umj1oAi4Y5DmecV6


https://images.app.goo.gl/gkYh64M1wRKtjGuSA


direct implementation of ReLU neural networks

### install

```sh
pip install ReLUs
```
#### or

```sh
pip3 install ReLUs
```

#### parameters for the model to train
```
layers_sizes   (e.g.  layers_size=[13,5,5,1])
num_iters      (e.g. num_iters=1000)
learning_rate (e.g. learning_rate=0.03)
```

#### training the model
```sh
model_name =  model(X_train, Y_train, layer_sizes, num_iters, learning_rate)
```
#### train and test accuracy
```sh
train_acc, test_acc = compute_accuracy(X_train, X_test, Y_train, Y_test, model_name)
```
#### making predictions
```
predict(X_train,your_model)
```

#### REFRENCES

https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/

https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

https://www.kaggle.com/dansbecker/rectified-linear-units-relu-in-deep-learning
