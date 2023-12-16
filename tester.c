

1. You have been hired to automate an audible response unit from a call center company. Every time a new customer's call comes in, the system must be able to understand the current load of the service as well as the goal of the call and recommend the right path in the audible response unit. The company does not have labeled data to supervise the model; it must take an approach to learn by experience (trial and error) and every time the algorithm makes a good recommendation of path, it will be rewarded. Which type of machine learning approach would best fit this project?
a) Unsupervised learning

b) Reinforcement learning

c) Supervised learning

d) DL

ANSWER

b, Since there is no labeled data and the agent needs to learn by experience, reinforcement learning is more appropriate for this use case. Another important fact in the question is that the agent is rewarded for good decisions.


2. You are working in a marketing department of a big company and you need to segment your customers based on their buying behavior. Which type of ML approach would best fit this project?
a) Unsupervised learning

b) Reinforcement learning

c) Supervised learning

d) DL

ANSWER

a, Clustering (which is an unsupervised learning approach) is the most common type of algorithm to work with data segmentation/clusters.



3. You are working in a retail company that needs to forecast sales for the next few months. Which type of ML approach would best fit this project?
a) Unsupervised learning

b) Reinforcement learning

c) Supervised learning

d) DL

ANSWER

c, Forecasting is a type of supervised learning that aims to predict a numerical value; hence, it might be framed as a regression problem and supervised learning.


4. A manufacturing company needs to understand how much money they are spending on each stage of their production chain. Which type of ML approach would best fit this project?
a) Unsupervised learning.

b) Reinforcement learning.

c) Supervised learning.

d) ML is not required.

ANSWER

d, ML is everywhere, but not everything needs ML. In this case, there is no need to use ML since the company should be able to collect their costs from each stage of the production chain and sum it up.


  
5. Which one of the following learning approaches gives us state-of-the-art algorithms to implement chatbots?
a) Unsupervised learning

b) Reinforcement learning

c) Supervised learning

d) DL

ANSWER

d, DL has provided state-of-the-art algorithms in the field of natural language processing.




6. You receive a training set from another team to create a binary classification model. They have told you that the dataset was already shuffled and ready for modeling. You have decided to take a quick look at how a particular algorithm, based on a neural network, would perform on it. You then split the data into train and test sets and run your algorithm. However, the results look very odd. It seems that the algorithm could not converge to an optimal solution. What would be your first line of investigation on this issue?
a) Make sure the algorithm used is able to handle binary classification models.

b) Take a look at the proportion of data of each class and make sure they are balanced.

c) Shuffle the dataset before starting working on it.

d) Make sure you are using the right hyperparameters of the chosen algorithm.

ANSWER

c, Data scientists must be skeptical about their work. Do not make assumptions about the data without prior validation. At this point in the book, you might not be aware of the specifics of neural networks, but you know that ML models are very sensitive to the data they are training on. You should double-check the assumptions that were passed to you before taking other decisions. By the way, shuffling your training data is the first thing you should do. This is likely to be present in the exam.



7. You have created a classification model to predict whether a banking transaction is fraud or not. During the modeling phase, your model was performing very well on both the training and testing sets. When you executed the model in a production environment, people started to complain about the low accuracy of the model. Assuming that there was no overfitting/underfitting problem during the training phase, what would be your first line of investigation?
a) The training and testing sets do not follow the same distribution.

b) The training set used to create this model does not represent the real environment where the model was deployed.

c) The algorithm used in the final solution could not generalize enough to identify fraud cases in production.

d) Since all ML models contain errors, we can't infer their performance in production systems.

ANSWER

b, Data sampling is very challenging, and you should always make sure your training data can represent the production data as precisely as possible. In this case, there was no evidence that the training and testing sets were invalid, since the model was able to perform well and consistently on both sets of data. Since the problem happens to appear only in production systems, there might have been a systematic issue in the training that is causing the issue.



8. You are training a classification model with 500 features that achieves 90% accuracy in the training set. However, when you run it in the test set, you get only 70% accuracy. Which of the following options are valid approaches to solve this problem (select all that apply)?
a) Reduce the number of features.

b) Add extra features.

c) Implement cross-validation during the training process.

d) Select another algorithm.

ANSWER

a, c, This is clearly an overfitting issue. In order to solve this type of problem, you could reduce the excessive number of features (which will reduce the complexity of the model and make it less dependent on the training set). Additionally, you could also implement cross-validation during the training process.




  9. You are training a neural network model and want to execute the training process as quickly as possible. Which of the following hardware architectures would be most helpful to you to speed up the training process of neural networks?
a) Use a machine with a CPU that implements multi-thread processing.

b) Use a machine with GPU processing.

c) Increase the amount of RAM of the machine.

d) Use a machine with SSD storage.

ANSWER

b, Although you might take some benefits from multi-thread processing and large amounts of RAM, using a GPU to train a neural network will give you the best performance. You will learn much more about neural networks in later chapters of this book, but you already know that they perform a lot of matrix calculations during training, which is better supported by the GPU rather than the CPU.




  10. Which of the following statements is not true about data resampling?
a) Cross-validation is a data resampling technique that helps to avoid overfitting during model training.

b) Bootstrapping is a data resampling technique often embedded in ML models that needs resampling capabilities to estimate the target function.

c) The parameter k in k-fold cross-validation specifies how many samples will be created.

d) Bootstrapping works without replacement.

ANSWER

d, All the statements about cross-validation and bootstrapping are correct except option d, since bootstrapping works with replacement (the same observations might appear on different splits).






