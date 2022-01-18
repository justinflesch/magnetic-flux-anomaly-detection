import numpy as np

import os

import matplotlib.pyplot as plt
import logging

# configurations for the basic logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
import matplotlib
font = {'weight' : 'normal',
        'size'   : 22}
matplotlib.rc('font', **font)


# GLOBAL PARAMETERS FOR STOCHASTIC GRADIENT DESCENT
# (we won't neccessarily be adjusting all of these)
np.random.seed(0)
batch_size = 200 # the batch size for stochastic gradient descent
max_epochs = 50 # number of total epochs to train the neural network

# new hyperparameters
# hyperparameter tuple = {(activation_callback, width, step_size)}
# EXAMPLE: hp_tuple = {(LinearLayer, 16, 0.01), (ReLU, 16, 0.01), (LinearLayer, 16, 0.01)} # make two layer neural network with two different activation functions

# create an dictionary with it's associated callback

def main():

  highest_seed = 0
  highest_acc = 0

  X_train, Y_train, X_val, Y_val, X_test = loadData(True)

  # Load data and display an example
  displayExample(X_train[np.random.randint(0,len(X_train))])

  # Build a network with input feature dimensions, output feature dimension,
  # hidden dimension, and number of layers as specified below
  print("INPUT DIM:", X_train.shape[1])
  hp_tuple = [(LinearLayer, X_train.shape[1], 0.01), (ReLU, 1024, 0.01), (LinearLayer, 10, 0.01)]

  for i in range(len(hp_tuple)):
    print(hp_tuple[i][0].__name__)
#   net = FeedForwardNeuralNetwork(X_train.shape[1],10,width_of_layers,number_of_layers, activation=activation)
  net = FeedForwardNeuralNetwork(hp_tuple)
  # Some lists for book-keeping for plotting later
  losses = []
  val_losses = []
  accs = []
  val_accs = []

  # Loss function
  lossFunc = CrossEntropySoftmax()

  # Indicies we will use to shuffle data randomly
  inds = np.arange(len(X_train))
  for i in range(max_epochs):
    
    # Shuffled indicies so we go through data in new random batches
    np.random.shuffle(inds)

    # Go through all datapoints once (aka an epoch)
    j = 0
    acc_running = loss_running = 0

    # training the batch size (increment j by batch size)
    while j < len(X_train):

      # Select the members of this random batch
      # print("Batch size: ", batch_size, i)
      b = min(batch_size, len(X_train)-j)
      X_batch = X_train[inds[j:j+b]]
      Y_batch = Y_train[inds[j:j+b]].astype(np.int)
    
      # Compute the scores for our 10 classes using our model
      logits = net.forward(X_batch)
      loss = lossFunc.forward(logits, Y_batch)
      acc = np.mean( np.argmax(logits,axis=1)[:,np.newaxis] == Y_batch)

      # Compute gradient of Cross-Entropy Loss with respect to logits
      loss_grad = lossFunc.backward()

      # Pass gradient back through networks
      net.backward(loss_grad)

      # Take a step of gradient descent
      net.step()
      # print("step_size", step_size)

      #Record losses and accuracy then move to next batch
      losses.append(loss)
      accs.append(acc)
      loss_running += loss*b
      acc_running += acc*b

      j+=batch_size

    # Evaluate performance on validation. This function looks very similar to the training loop above, 
    vloss, vacc = evaluateValidation(net, X_val, Y_val, batch_size)
    val_losses.append(vloss)
    val_accs.append(vacc)
    
    # Print out the average stats over this epoch
    logging.info("[Epoch {:3}]   Loss:  {:8.4}     Train Acc:  {:8.4}%      Val Acc:  {:8.4}%".format(i,loss_running/len(X_train), acc_running / len(X_train)*100,vacc*100))
    # if (vacc*100 > highest_acc):
    #   # step_size = step_size - 0.001 if step_size > 0 else 0
    #   highest_acc = vacc* 100
    #   with open("FINAL", "a") as f:
    #     f.write("[Rand:{:3}] [Epoch {:3}]   Loss:  {:8.4}     Train Acc:  {:8.4}%      Val Acc:  {:8.4}%\n".format(r, i,loss_running/len(X_train), acc_running / len(X_train)*100,vacc*100))
    
    print(batch_size, highest_acc)
    
  # calculate cross-entropy loss
  fig, ax1 = plt.subplots(figsize=(16,9))
  color = 'tab:red'
  ax1.plot(range(len(losses)), losses, c=color, alpha=0.25, label="Train Loss")
  ax1.plot([np.ceil((i+1)*len(X_train)/batch_size) for i in range(len(val_losses))], val_losses,c="red", label="Val. Loss")
  ax1.set_xlabel("Iterations")
  ax1.set_ylabel("Avg. Cross-Entropy Loss", c=color)
  ax1.tick_params(axis='y', labelcolor=color)
  ax1.set_ylim(-0.01,3)

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  # calculate the accuracy
  color = 'tab:blue'
  ax2.plot(range(len(losses)), accs, c=color, label="Train Acc.", alpha=0.25)
  ax2.plot([np.ceil((i+1)*len(X_train)/batch_size) for i in range(len(val_accs))], val_accs,c="blue", label="Val. Acc.")
  ax2.set_ylabel(" Accuracy", c=color)
  ax2.tick_params(axis='y', labelcolor=color)
  ax2.set_ylim(-0.01,1.01)

  fig.tight_layout()  # otherwise the right y-label is slightly clipped
  ax1.legend(loc="center")
  ax2.legend(loc="center right")
  plt.show()

  # test performance of the neural network with output
  logits = net.forward(X_test)

  print(np.argmax(logits,axis=1)[:,np.newaxis])
  test_Y = np.argmax(logits,axis=1)[:,np.newaxis]
  print(test_Y.size)
    # add index and header then save to file
  test_out = np.concatenate((np.expand_dims(np.array(range(4220),dtype=np.int), axis=1), test_Y), axis=1)
  header = np.array([["id", "digit"]])
  test_out = np.concatenate((header, test_out))
  np.savetxt('mnist_test_output.csv', test_out, fmt='%s', delimiter=',')




class LinearLayer:

  # Initialize our layer with (input_dim, output_dim) weight matrix and a (1,output_dim) bias vector
  def __init__(self, input_dim, output_dim, step_size):
    self.weights = np.random.randn(input_dim, output_dim)* np.sqrt(2. / input_dim)
    self.bias = np.ones( (1,output_dim) )*0.5
    self.step_size = step_size

  # During the forward pass, we simply compute Xw+b
  def forward(self, input):
    self.input = input #Storing X
    return  self.input@self.weights + self.bias

  def backward(self, grad):
    self.grad_weights = np.transpose(self.input)@grad
    self.grad_bias = np.sum(grad, axis = 0)
    return grad@np.transpose(self.weights)
    
  def step(self):
    self.weights -= self.step_size*self.grad_weights
    self.bias -= self.step_size*self.grad_bias


# 2 rens, 128, 0.1, 5, seed 102

class FeedForwardNeuralNetwork:

  def __init__(self, layer_table):

    # add the first layer
    self.layers = [LinearLayer(layer_table[0][1], layer_table[2][1], layer_table[0][2])]
    # odd values of "i" must be linearLayers
    for i in range(1, len(layer_table)):
    # append the hidden layer activation function
      self.layers.append(LinearLayer(layer_table[i][1], layer_table[i-2][1], layer_table[i][2])) if i % 2 == 0 else self.layers.append(layer_table[i][0]())
    
    #append the last layer
    # check if it works
    for i in range(len(self.layers)):
      print(type(self.layers[i]))
    
  def forward(self, X):
    for layer in self.layers:
      X = layer.forward(X)
    return X

  def backward(self, grad):
    for layer in reversed(self.layers):
      grad = layer.backward(grad)

  def step(self):
    for layer in self.layers:
      layer.step()





# Sigmoid or Logistic Activation Function
class Sigmoid:

  # Given the input, apply the sigmoid function
  # store the output value for use in the backwards pass
  def forward(self, input):
    self.act = 1/(1+np.exp(-input))
    return self.act
  
  # Compute the gradient of the output with respect to the input
  # self.act*(1-self.act) and then multiply by the loss gradient with 
  # respect to the output to produce the loss gradient with respect to the input
  def backward(self, grad):
    return grad * self.act * (1-self.act)

  # The Sigmoid has no parameters so nothing to do during a gradient descent step
  def step(self):
    return

# Rectified Linear Unit Activation Function
class ReLU:

  # Forward pass is max(0,input)
  def forward(self, input):
    self.mask = (input > 0)
    return input * self.mask
  
  # Backward pass masks out same elements
  def backward(self, grad):
    return grad * self.mask

  # No parameters so nothing to do during a gradient descent step
  def step(self):
    return

# create an dictionary with it's associated callback
activation_callback = {
  "ReLU": ReLU,
  "Sigmoid": Sigmoid
}



#####################################################
# Utility Functions for Computing Loss / Val Metrics
#####################################################
def softmax(x):
  x -= np.max(x,axis=1)[:,np.newaxis]  # Numerical stability trick
  return np.exp(x) / (np.sum(np.exp(x),axis=1)[:,np.newaxis])


class CrossEntropySoftmax:

  def forward(self, logits, labels):
    self.probs = softmax(logits)
    self.labels = labels
    return -np.mean(np.log(self.probs[np.arange(len(self.probs))[:,np.newaxis],labels]+0.00001))

  def backward(self):
    grad = self.probs
    grad[np.arange(len(self.probs))[:,np.newaxis],self.labels] -=  1
    return  grad.astype(np.float64)/len(self.probs)


def evaluateValidation(model, X_val, Y_val, batch_size):
  val_loss_running = 0
  val_acc_running = 0
  j=0

  lossFunc = CrossEntropySoftmax()

  while j < len(X_val):
    b = min(batch_size, len(X_val)-j)
    X_batch = X_val[j:j+b]
    Y_batch = Y_val[j:j+b].astype(np.int)
   
    logits = model.forward(X_batch)
    loss = lossFunc.forward(logits, Y_batch)
    acc = np.mean( np.argmax(logits,axis=1)[:,np.newaxis] == Y_batch)

    val_loss_running += loss*b
    val_acc_running += acc*b
       
    j+=batch_size

  return val_loss_running/len(X_val), val_acc_running/len(X_val)







#####################################################
# Utility Functions for Loading and Displaying Data #
#####################################################
def loadData(normalize = True):
  cwd = os.getcwd()
  print(cwd + "\\autoencoder\\mnist_small_train.csv")
  train = np.loadtxt(cwd + "\\autoencoder\\mnist_small_train.csv", delimiter=",", dtype=np.float64)
  val = np.loadtxt(cwd + "\\autoencoder\\mnist_small_val.csv", delimiter=",", dtype=np.float64)
  test = np.loadtxt(cwd + "\\autoencoder\\mnist_small_test.csv", delimiter=",", dtype=np.float64)

  # Normalize Our Data
  if normalize:
    X_train = train[:,:-1]/256-0.5
    X_val = val[:,:-1]/256-0.5
    X_test = test/256-0.5
  else:
    X_train = train[:,:-1]
    X_val = val[:,:-1]
    X_test = test

  Y_train = train[:,-1].astype(np.int)[:,np.newaxis]
  Y_val = val[:,-1].astype(np.int)[:,np.newaxis]

  logging.info("Loaded train: " + str(X_train.shape))
  logging.info("Loaded val: " + str(X_val.shape))
  logging.info("Loaded test: "+ str(X_test.shape)) 

  return X_train, Y_train, X_val, Y_val, X_test


def displayExample(x):
  plt.imshow(x.reshape(28,28),cmap="gray")
  plt.show()


if __name__=="__main__":
  main()