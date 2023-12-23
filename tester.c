
FROM Shreyas Subramanian

Assessment Test

1. You are building a supervised ML model for predicting housing prices in the United States. However, you notice that your dataset has a lot of highly correlated features. What are some methods you can use to reduce the number of features in your dataset? (Choose all that apply.)
Use principal component analysis to perform dimensionality reduction.
Add an L2 regularization term to your loss function.
Add an L1 regularization term to your loss function.
Add an L3 regularization term to your loss function.

2. Which of the following is an unsupervised learning algorithm useful with tabular data?
K-nearest neighbors
K-means clustering
Latent Dirichlet Allocation (LDA)
Random forest

3. Which of the following ML instance types is ideally suited for deep learning training?
EC2 M family instances
EC2 Inf1 instances powered by AWS Inferentia
EC2 G4 family of instances
EC2 P3 family of instances

4. Your company has a vast number of documents that contain some personally identifiable information (PII). The company is looking for a solution where the documents can be uploaded to the cloud and for a service that will extract the text from the documents and redact the PII. The company is concerned about the costs to train deep learning models for text and entity extraction. What solution would you recommend for this use case?
Upload the data to S3. Train a custom SageMaker model for text extraction from raw documents, followed by an entity extraction algorithm to extract the PII entities.
Use an off-the-shelf optical character recognition (OCR) tool to extract the text. Then use an entity detection algorithm to extract PII entities.
Use Amazon Textract to extract text from documents and Amazon Comprehend PII detection to detect PII entities.
Use Amazon Textract to extract text from documents and Amazon Rekognition PII detection to detect PII entities.
  
5. You have written some algorithm code on a local IDE like PyCharm and uploaded that script to your SageMaker environment. The code is the entry point to a training container, which contains all the relevant packages you need for training. However, before kicking off a full training job, you want to quickly and interactively test whether the code is working as expected. What can you do to achieve this?
Kick off a SageMaker processing job to test your code. Once it is working, then kick off a SageMaker training job.
Kick off a SageMaker training job on a t3.medium. Once you are convinced it is working, then switch to a larger instance type.
Use SageMaker local mode to kick off a job locally on your SageMaker notebook instance. Debug your scripts, and once they are working, start a SageMaker training job.
Kick off a SageMaker Batch Transform job to test your code. Once it is working, then kick off a SageMaker training job.

6. You are building a supervised ML model for forecasting average sales for your products based on product metadata and prior month sales. The data is arranged in a tabular format, where each row corresponds to a different product. Which machine learning algorithms might you choose for this task? (Choose all that apply.)
Random forest classifier
DeepAR forecasting
Random forest regressor
Linear regression

7. A business stakeholder from a solar energy company comes to you with a business problem to identify solar panels on roofs from aerial footage data. Currently, the business stakeholder does not have much labeled data available. What advice would you give them to proceed with this use case?
Semantic segmentation is an unsupervised ML problem that doesn't require labeled data. You can use a clustering algorithm to discover the roofs.
Semantic segmentation requires labels. Since this use case is very domain specific, you will need to train a custom model to detect them. For this, you will need to first develop a strategy to acquire labels. Advise the business stakeholder that you will need to factor in data labeling as part of this project.
Semantic segmentation requires labels. Simply pick up an off-the-shelf object detection model that is trained on ImageNet corpus for detecting the roofs.
Semantic segmentation is not an ML problem. Advise them to write a set of rules to detect solar panels on roofs based on the geometry of the solar panels.

8. Consider the same problem as the use case in Question 7. What AWS solution would you recommend to the stakeholder for generating labeled data?
Use Amazon Rekognition custom labels to label the rooftops.
Use SageMaker Data Wrangler.
Use Amazon Augmented AI.
Use SageMaker Ground Truth.

9. Which AWS service would you use to optimize your ML models to run on a specific hardware platforms or edge devices with processors from ARM, NVIDIA, Xilinx, and Texas Instruments?
Amazon CodeGuru
Amazon DevOps Guru
SageMaker Neuron SDK
SageMaker Neo

10. Which AWS AI/ML service would you use to detect anomalies in retail transaction data? (Choose all that apply.)
Amazon SageMaker Random Cut Forest
Amazon SageMaker DeepAR
Amazon Lookout for Metrics
Amazon Forecast

11. You have set up your S3 buckets in such a way that they cannot be accessed outside of your VPC using an S3 bucket policy. You are now passing the S3 prefix for your training dataset to SageMaker's training estimator to kick off training but find that SageMaker is unable to access your S3 buckets and give a Permission Denied Error. How can you resolve this issue?
Remove the bucket policy to allow the bucket to be accessed by SageMaker from outside of your VPC.
Modify the IAM role passed to SageMaker training estimator to make sure it has access to the S3 bucket.
Provide your network settings using the security_group_ids and subnets parameters for the VPC. Make sure to create an S3 VPC endpoint.
Migrate your dataset over to EFS and try again.

12. Which of the following is the customer's responsibility when it comes to security of Amazon Comprehend? (Choose all that apply.)
Patching of the instances used to run Comprehend custom entity detection jobs
Maintaining the availability of Comprehend Detect Entities endpoints
Creating an IAM role that provides permissions for the user to call Amazon Comprehend's APIs
Setting up a Comprehend VPC endpoint to ensure that network traffic flows through your VPC

13. Your team uses Amazon S3 for storing input datasets and would like to use PySpark code to preprocess the raw data before training. Which of the following solutions will require the least amount of setup and maintenance?
Create an EMR cluster with Spark installed. Then use a notebook to prepare data.
Use SageMaker Processing for Spark preprocessing.
Create a Glue crawler to populate a Glue data catalog, then write an ETL script in PySpark to be run in Glue.
Use AWS Data Pipeline with an Apache Hive Metastore for preprocessing.

14. A classification model has the following confusion matrix: true positives (tp) = 90, false positives (fp) = 4, true negatives (tn) = 96, false negatives (fn) = 10, as shown below. What is the recall for this model?
Predicted/Actuals	Positive	Negative
Positive	90	4
Negative	10	96
0.9
0.85
0.8
0.95
  
15. What methods of hyperparameter optimization does Amazon SageMaker provide? (Choose all that apply.)
Grid Search
Random Search
Matrix Search
Bayesian Optimization

16. You are helping a company that uses segmentation models in PyTorch to identify ships from high-resolution satellite images. The customer feels that the average IoU across all classes of ships is low and would like to explore hyperparameter optimization. What is the easiest way to set this up on SageMaker?
Use a custom container in SageMaker that explores multiple hyperparameters within a single training job.
Use SageMaker's built-in HPO functionality with PyTorch.
Implement your own HPO code since SageMaker's built-in HPO functionality does not work with PyTorch.
Use Ray Tune along with PyTorch with SageMaker in Script mode.
  
17. A financial services customer holds 20 years of stock market data in Amazon Redshift. They have a suite of ML-based algorithms trained on SageMaker and would like to back test these algorithms using 20 years of data. Which of the following services or features can be used? (Choose all that apply.)
SageMaker Processing
SageMaker Backtest
SageMaker Batch Transform
SageMaker Feature Store

18. You are helping a customer set up an update to their existing machine learning API. Their new model has a higher accuracy than the existing model but has not been tested with live traffic yet. What advice will you have for the customer for next steps?
Perform canary testing
Perform blue/green testing
Perform shadow testing
Perform A/B testing

19. A logistics company would like to group similar invoices together using an API call. An upstream process processes raw invoices and creates metadata in a tabular format. Which of the following built-in algorithms in SageMaker can help this customer?
DeepAR
K-means
Neural topic model
Sequence to sequence

20. You are writing code for a regression model using decision trees. Which of the following metrics will you use to evaluate your model?
RMSE close to 0
RMSE close to 1
AuC close to 1
F1 score close to 1
  
21. Your customer primarily uses Scikit Learn and XGBoost for developing ensemble models. In many cases, the selected ensemble models involve both XGBoost and Scikit Learn models working together. What is the easiest way to implement this using features available in SageMaker?
Build and use a container that includes both XGBoost and Scikit Learn for Inference.
Use Scikit Learn script mode and include XGBoost as a dependency to use both types of models in the ensemble.
Deploy two separate endpoints, one for all Scikit Learn models and another for XGBoost. Use a Lambda function to obtain results from both endpoints and collate the results.
Use multimodel endpoints.
  
22. Your company released a product 6 months ago and has been collecting customer reviews. Which of the following services can help with detecting sentiment in these reviews? (Choose all that apply.)
Amazon Sentiment
Amazon Comprehend
Amazon SageMaker BlazingText
Amazon Connect

23. A healthcare customer of yours has developed a model to detect skin cancers but is confused as to which metric to use for evaluating these models. What will your advice to them be?
Use Recall.
Use Precision.
Use Accuracy.
Use F1 score.
  
24. You and your teammates use the PyCharm IDE on your laptop to develop models. Management has now asked you to figure out how to deploy these models on the cloud using Amazon SageMaker. Which of the following options will you test before going back with an answer to your management?
Use the PyCharm SageMaker plug-in.
Upload the models to Amazon S3, and then use SageMaker for model hosting.
Deploy the models locally using PyCharm and provide credentials to SageMaker.
None of the above options are correct.




FROM Somanath Nanda

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






