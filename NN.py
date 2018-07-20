import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

num_examples = 50
x=np.linspace(-2, 4, num_examples)
y=np.linspace(-6, 6, num_examples)

#print x
#plt.figure(figsize=(4,4))
#plt.scatter(x, y)
#plt.show()

# Generate random pertubation
# Perturbation theory comprises mathematical methods for finding an approximate solution to a problem, by starting from the exact solution of a related, simpler problem.
randnum = np.random.random([num_examples])

x += randnum #an 1-d array with random numbers
y += np.random.random([num_examples])

x_with_bias = np.array[(1., a) for a in x]).astype(np.float32)
#print x_with_bias

#Train a neural network with Gradient Descent, The object is minimizing L2 loss

losses = []
training_steps = 50 # 50번을 돌린다
learning_rate = 0.002

with tf.Session() as sess:
  # Set up all the tensors, variables, and operations.
  input = tf.constant(x_with_bias)
  target = tf.constant(np.transpose([y]).astype(np.float32))
  weights = tf.Variable(tf.random_normal([2, 1], 0, 0.1))
  
  tf.initialize_all_variables().run()
                      
  yhat = tf.matmul(input, weights)
  yerror = tf.subtract(yhat, target) # yhat = input * weight , target = ax + b , yerror = target - (ax + b)
  loss = tf.nn.l2_loss(yerror) # make loss function from error
  
  update_weights = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss) # minimize error
  
  for _ in range(training_steps):
    update_weights.run()
    losses.append(loss.eval())
    #print _  #It takes on value from 0 to 49
    #print losses  #this shows losses array keep increasing in size: [18] , [18, 13],...

  # Training is done, get the final values for the graphs
  betas = weights.eval()
  yhat = yhat.eval()


# Show the actual and predicted data points

  plt.figure(figsize=(4,4))

plt.scatter(x, y, alpha=.9)  #plot original x and y
plt.scatter(x, np.transpose(yhat)[0], c="g", alpha=.6) #plot x and yhat

x_range = (-4, 6)
plt.plot(x_range, [betas[0] + a * betas[1] for a in x_range], "g", alpha=0.6)

plt.show()

#fitted. data training output이 나옴. target을 두고 training을 했더니 y값이 비슷하게 나왔다.

# Plot the prediction error over time
# Show the loss over time.
plt.figure(figsize=(4,4))

plt.plot(range(0, training_steps), losses)
#plt.set_ylabel("Loss")
#plt.set_xlabel("Training steps")

plt.show()

# 처음에는 loss가 컸지만 (random을 이용했기 때문에) training을 반복하면서 loss 값이 minimize됨. target과 output이 비슷해짐. weight trained.
