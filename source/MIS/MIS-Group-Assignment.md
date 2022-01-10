---
title: MIS Group Assignment
date: 2021-12-21 19:51:30
tags: HW
---

**目录 Table of Contents**

1. 什么是机器学习 What is Machine Learning (ML)

   1. 实际应用领域 Real-World Applications

2. 机器学习的类别 Types of ML problems

   1. Supervised Learning
      1. Classification
      2. Regression
   2. Unsupervised Learning
      1. Clustering

3. 如何选择合适的算法 How to choose the appropriate algorithm

4. 适用机器学习的情景 When to appropriately use ML

   



## Introduction

1. **What is Machine Learning?**
   1. Machine learning teaches computers to do what comes naturally to humans and animals: learn from experience. Machine learning algorithms use computational methods to “learn” information directly from data without relying on a predetermined equation as a model. The algorithms adaptively improve their performance as the number of samples available for learning increases. 机器学习教计算机从过往的经验中总结和学习。**机器学习算法**使用计算方法直接**从数据中“学习”**信息，而**不依赖**预先确定的方程作为模型。随着可用于学习的样本数量的增加，这些算法自适应地提高其性能。
2. **More Data, More Questions, Better Answers**
   1. Machine learning algorithms find natural patterns in data that generate insight and help you make better decisions and predictions. They are used every day to make critical decisions in medical diagnosis, stock trading, energy load forecasting, and more. Media sites rely on machine learning to sift through millions of options to give you song or movie recommendations. Retailers use it to gain insight into their customers’ purchasing behavior.
   2. （Real-World Applications）With the rise in big data, machine learning has become particularly important for solving problems in areas like these:
      - Computational finance, for credit scoring and algorithmic trading
      - Image processing and computer vision, for face recognition, motion detection, and object detection
      - Computational biology, for tumor detection, drug discovery, and DNA sequencing
      - Energy production, for price and load forecasting
      - Automotive, aerospace, and manufacturing, for predictive maintenance
      - Natural language processing
3. **How Machine Learning Works**
   1. Machine learning uses two types of techniques: 
      1. Supervised learning: 
         1. trains a model on known input and output data so that it can predict future outputs
         2. Develop predictive model based on both input and output data
         3. **Classification**
            1. Support Vector Machines
            2. Discriminal Analysis
            3. Naive Bayes
            4. Nearest Neighbor
         4. Regression
            1. Linear Regression, GLM
            2. SVR, GPR
            3. Ensemble Methods
            4. Decision Trees
            5. Neutural Networks
         5. Choose supervised learning if you need to train a model to make a prediction
            1. for example, the future value of a continuous variable, such as temperature or a stock price, or a classification—for example, identify makes of cars from webcam video footage.
      2. Unsupervised learning: 
         1. finds hidden patterns or intrinsic structures in input data.
         2. Group and interpret data based on input data
         3. **Clustering**
            1. K-Means, K-Medoids, Fuzzy C-Means
            2. Hierarchical
            3. Gaussian Mixture
            4. Neutral Networks
            5. Hidden Markov Model
         4. Choose unsupervised learning if you need to explore your data and want to train a model to find a good internal representation, 
            1. such as splitting data up into clusters.
4. **Supervised Learning**
   1. The aim of supervised machine learning is to build a model that makes predictions based on evidence in the presence of uncertainty. A supervised learning algorithm takes a known set of input data and known responses to the data (output) and trains a model to generate reasonable predictions for the response to new data.
   2. Supervised learning uses classification and regression techniques to develop predictive models.
      - Classification techniques predict discrete responses—for example, whether an email is genuine or spam, or whether a tumor is cancerous or benign. Classification models classify input data into categories. Typical applications include medical imaging, speech recognition, and credit scoring.
      - Regression techniques predict continuous responses— for example, changes in temperature or fluctuations in power demand. Typical applications include electricity load forecasting and algorithmic trading.
   3. Using Supervised Learning to Predict Heart Attacks
      1. Suppose clinicians want to predict whether someone will have a heart attack within a year. They have data on previous patients, including age, weight, height, and blood pressure. They know whether the previous patients had heart attacks within a year. So the problem is combining the existing data into a model that can predict whether a new person will have a heart attack within a year.
5. **Unsupervised Learning**
   1. Unsupervised learning finds hidden patterns or intrinsic structures in data. It is used to draw inferences from datasets consisting of input data without labeled responses.
   2. Clustering is the most common unsupervised learning technique. It is used for exploratory data analysis to find hidden patterns or groupings in data.
   3. Applications for clustering include gene sequence analysis, market research, and object recognition.
6. **How Do You Decide Which Algorithm to Use?**
   1. Choosing the right algorithm can seem overwhelming—there are dozens of supervised and unsupervised machine learning algorithms, and each takes a different approach to learning.
   2. There is no best method or one size fits all. Finding the right algorithm is partly just trial and error—even highly experienced data scientists can’t tell whether an algorithm will work without trying it out. But algorithm selection also depends on the size and type of data you’re working with, the insights you want to get from the data, and how those insights will be used.
7. **When Should You Use Machine Learning?**
   1. Consider using machine learning when you have a complex task or problem involving a large amount of data and lots of variables, but no existing formula or equation. 
   2. For example, machine learning is a good option if you need to handle situations like these:
      - Hand-written rules and equations are too complex—as in face recognition and speech recognition.
      - The rules of a task are constantly changing—as in fraud detection from transaction records.
      - The nature of the data keeps changing, and the program needs to adapt—as in automated trading, energy demand forecasting, and predicting shopping trends.
8. **Real-World Examples**
   1. ...P14 for more

## Getting Started

1. **Rarely a Straight Line**

   1. With machine learning there’s rarely a straight line from start to finish—you’ll find yourself constantly iterating and trying different ideas and approaches. This chapter describes a systematic machine learning workflow, highlighting some key decision points along the way.

2. **Machine Learning Challenges**

   1. Most machine learning challenges relate to handling your data and finding the right model.
   2. Data comes in all shapes and sizes. 
      1. Real-world datasets can be messy, incomplete, and in a variety of formats. You might just have simple numeric data. But sometimes you’re combining several different data types, such as sensor signals, text, and streaming images from a camera.
   3. Preprocessing your data might require specialized knowledge and tools. 
      1. For example, to select features to train an object detection algorithm requires specialized knowledge of image processing. Different types of data require different approaches to preprocessing.
   4. It takes time to find the best model to fit the data. 
      1. Choosing the right model is a balancing act. Highly flexible models tend to overfit data by modeling minor variations that could be noise. On the other hand, simple models may assume too much. There are always tradeoffs between model speed, accuracy, and complexity.
   5. Sounds daunting? Don’t be discouraged. Remember that trial and error is at the core of machine learning—if one approach or algorithm doesn’t work, you simply try another. But a systematic workflow will help you get off to a smooth start.

3. **Questions to Consider Before You Start**

   1. Every machine learning workflow begins with three questions:
      1. What kind of data are you working with?
      2. What insights do you want to get from it?
      3. How and where will those insights be applied?
   2. Your answers to these questions help you decide whether to use supervised or unsupervised learning.

4. **Workflow at a Glance**

   1. ACCESS and load the data.
   2. PREPROCESS the data.
   3. DERIVE features using the preprocessed data.
   4. TRAIN models using the features derived in step 3.
   5. ITERATE to find the best model.
   6. INTEGRATE the best-trained model into a production system.

5. **Training a Model to Classify Physical Activities**

   1. In the next sections we’ll look at the steps in more detail, using a health monitoring app for illustration. The entire workflow will be completed in MATLAB.
   2. This example is based on a cell phone health-monitoring app. The input consists of three-axial sensor data from the phone’s accelerometer and gyroscope. The responses, (or output), are the activities performed–walking, standing, running, climbing stairs, or lying down.
   3. We want to use the input data to train a classification model to identify these activities. Since our goal is classification, we’ll be applying supervised learning.
   4. The trained model (or classifier) will be integrated into an app to help users track their activity levels throughout the day.

6. **Step One: Load the Data**

   1. To load data from the accelerometer and gyroscope we do the following:
      1. Sit down holding the phone, log data from the phone, and store it in a text file labeled “Sitting.”
      2. Stand up holding the phone, log data from the phone, and store it in a second text file labeled “Standing.”
      3. Repeat the steps until we have data for each activity we want to classify.
   2. We store the labeled data sets in a text file. A flat file format such as text or CSV is easy to work with and makes it straightforward to import data.
   3. Machine learning algorithms aren’t smart enough to tell the difference between noise and valuable information. Before using the data for training, we need to make sure it’s clean and complete.

7. **Step Two: Preprocess the Data**

   1. Look for outliers–data points that lie outside the rest of the data.
      1. This is what we do to process the imported data in MATLAB and plot each labeled set.
      2. We must decide whether the outliers can be ignored or whether they indicate a phenomenon that the model should account for. In our example, they can safely be ignored (it turns out that we moved unintentionally while recording the data).

   2. Check for missing values (perhaps we lost data because the connection dropped during recording).
      1. We could simply ignore the missing values, but this will reduce the size of the data set. Alternatively, we could substitute approximations for the missing values by interpolating or using comparable data from another sample.
   3. Remove gravitational effects from the accelerometer data
      1. so that our algorithm will focus on the movement of the subject, not the movement of the phone. A simple high- pass filter such as a biquad filter is commonly used for this.
   4. Divide the data into two sets. 
      1. We save part of the data for testing (the test set) and use the rest (the training set) to build models. 
      2. This is referred to as holdout, and is a useful cross- validation technique.

8. **Step Three: Derive Features**

   1. Deriving features (also known as feature engineering or feature extraction) is one of the most important parts of machine learning. It turns raw data into information that a machine learning algorithm can use.
   2. For the activity tracker, we want to extract features that capture the frequency content of the accelerometer data. These features will help the algorithm distinguish between walking (low frequency) and running (high frequency). We create a new table that includes the selected features.
   3. Use feature selection to:
      - Improve the accuracy of a machine learning algorithm
      - Boost model performance for high-dimensional data sets
      - Improve model interpretability
      - Prevent overfitting

9. **Step Four: Build and Train the Model**

   1. The number of features that you could derive is limited only by your imagination. However, there are a lot of techniques commonly used for different types of data.
      1. Sensor data
         1. Feature Selection Task: 
            1. Extract signal properties from raw sensor data to create higher-level information
         2. Techniques:
            1. Peak analysis
               1. perform an fft and identify dominant frequencies 
            2. Pulse and transition metrics
               1. derive signal characteristics such as rise time, fall time, and settling time
            3. Spectral measurements
               1. plot signal power, bandwidth, mean frequency, and median frequency
      2. Image and video data
         1. Feature Selection Task: 
            1. Extract features such as edge locations, resolution, and color
         2. Techniques: 
            1. Bag of visual words
               1. create a histogram of local image features, such as edges, corners, and blobs
            2. Histogram of oriented gradients (HOG)
               1. create a histogram of local gradient directions
            3. Minimum eigenvalue algorithm
               1. detect corner locations in images
            4. Edge detection
               1. identify points where the degree of brightness changes sharply
      3. Transactional data
         1. Feature Selection Task: 
            1. Calculate derived values that enhance the information in the data
         2. Techniques：
            1. Timestamp decomposition
               1. break timestamps down into components such as day and month
            2. Aggregate value calculation
               1. create higher-level features such as the total number of times a particular event occurred
   2. When building a model, it’s a good idea to start with something simple; it will be faster to run and easier to interpret.
   3. To see how well it performs, we plot the confusion matrix, a table that compares the classifications made by the model with the actual class labels that we created in step 1.
   4. The confusion matrix shows that our model is having trouble distinguishing between dancing and running. Maybe a decision tree doesn’t work for this type of data. We’ll try a few different algorithms.
   5. We start with a K-nearest neighbors (KNN), a simple algorithm that stores all the training data, compares new points to the training data, and returns the most frequent class of the “K” nearest points. That gives us 98% accuracy compared to 94.1% for the simple decision tree. The confusion matrix looks better, too:
   6. However, KNNs take a considerable amount of memory to run, since they require all the training data to make a prediction. 
   7. We try a linear discriminant model, but that doesn’t improve the results. Finally, we try a multiclass support vector machine (SVM). The SVM does very well—we now get 99% accuracy:
   8. We achieved our goal by iterating on the model and trying different algorithms. If our classifier still couldn’t reliably differentiate between dancing and running, we’d look into ways to improve the model.

10. **Step Five: Improve the Model** 

    1. Improving a model can take two different directions: make the model simpler or add complexity.
    2. **Simplify**
    3. First, we look for opportunities to reduce the number of features. Popular feature reduction techniques include:
       - Correlation matrix – shows the relationship between variables, so that variables (or features) that are not highly correlated can be removed.
       - Principal component analysis (PCA) – eliminates redundancy by finding a combination of features that captures key distinctions between the original features and brings out strong patterns in the dataset.
       - Sequential feature reduction – reduces features iteratively on the model until there is no improvement in performance.
    4. Next, we look at ways to reduce the model itself. We can do this by

    - - Pruning branches from a decision tree
      - Removing learners from an ensemble

    5. A good model includes only the features with the most predictive power. A simple model that generalizes well is better than a complex model that may not generalize or train well to new data.
    6. In machine learning, as in many other computational processes, simplifying the model makes it easier to understand, more robust, and more computationally efficient.
    7. **Add Complexity**
    8. If our model can’t differentiate dancing from running because it is over-generalizing, then we need find ways to make it more fine-tuned. To do this we can either:
       - Use model combination – merge multiple simpler models into a larger model that is better able to represent the trends in the data than any of the simpler models could on their own.
       - Add more data sources – look at the gyroscope data as well as the acceleromter data. The gyroscope records the orientation of the cell phone during activity. This data might provide unique signatures for the different activities; for example, there might be a combination of acceleration and rotation that’s unique to running.
    9. Once we’ve adjusted the model, we validate its performance on the test data that we set aside during preprocessing.
    10. If the model can reliably classify activities on the test data, we’re ready to move it to the phone and start tracking.

## Applying Unsupervised Learning 

1. **When to Consider Unsupervised Learning**
   1. Unsupervised learning is useful when you want to explore your data but don’t yet have a specific goal or are not sure what information the data contains. It’s also a good way to reduce the dimensions of your data.
2. **Unsupervised Learning Techniques**
   1. As we saw in section 1, most unsupervised learning techniques are a form of cluster analysis.
   2. In cluster analysis, data is partitioned into groups based on some measure of similarity or shared characteristic. Clusters are formed so that objects in the same cluster are very similar and objects in different clusters are very distinct.
   3. Clustering algorithms fall into two broad groups:
      1. Hard clustering, 
         1. where each data point belongs to only one cluster
      2. Soft clustering, 
         1. where each data point can belong to more than one cluster
   4. You can use hard or soft clustering techniques if you already know the possible data groupings.
   5. If you don’t yet know how the data might be grouped:
      - Use self-organizing feature maps or hierarchical clustering to look for possible structures in the data.
      - Use cluster evaluation to look for the “best” number of groups for a given clustering algorithm.
3. **Common Hard Clustering Algorithms**
   1. **k-Means**
      1. How it Works
         1. Partitions data into k number of mutually exclusive clusters. How well a point fits into a cluster is determined by the distance from that point to the cluster’s center.
      2. Best Used...
         1. When the number of clusters is known
         2. For fast clustering of large data sets
      3. Result: 
         1. Cluster centers
      4. Example: 
         1. Using k-Means Clustering to Site Cell Phone Towers
            1. A cell phone company wants to know the number and placement of cell phone towers that will provide the most reliable service. For optimal signal reception, the towers must be located within
               clusters of people.
            2. The workflow begins with an initial guess at the number of clusters that will be needed. To evaluate this guess, the engineers compare service with three towers and four towers to see how well they’re able to cluster for each scenario (in other words, how well the towers provide service).
            3. A phone can only talk to one tower at a time, so this is a hard clustering problem. The team uses k-means clustering because k-means treats each observation in the data as an object having a location in space. It finds a partition in which objects within each cluster are as close to each other as possible and as far from objects in other clusters as possible.
            4. After running the algorithm, the team can accurately determine the results of partitioning the data into three and four clusters.
   2. **k-Medoids**
      1. How It Works
         1. Similar to k-means, but with the requirement that the cluster centers coincide with points in the data.
      2. Best Used...
         1. When the number of clusters is known
         2. For fast clustering of categorical data
         3. To scale to large data sets
      3. Result: 
         1. Cluster centers that coincide with data points
   3. **Hierarchical Clustering**
      1. How it Works
         1. Produces nested sets of clusters by analyzing similarities between pairs of points and grouping objects into a binary, hierarchical tree.
      2. Best Used...
         1. When you don’t know in advance how many clusters are in your data
         2. You want visualization to guide your selection
      3. Result: 
         1. Dendrogram showing the hierarchical relationship between clusters
   4. **Self-Organizing Map**
      1. How It Works
         1. Neural-network based clustering that transforms a dataset into a topology-preserving 2D map.
      2. Best Used...
         1. To visualize high-dimensional data in 2D or 3D
         2. To deduce the dimensionality of data by preserving its topology (shape)
      3. Result:
         1. Lower-dimensional (typically 2D) representation
4. **Common Soft Clustering Algorithms**
   1. **Fuzzy c-Means**
      1. How it Works
         1. Partition-based clustering when data points may belong to more than one cluster.
      2. Best Used...
         1. When the number of clusters is known
         2. For pattern recognition
         3. When clusters overlap
      3. Result: 
         1. Cluster centers (similar to k-means) but with fuzziness so that points may belong to more than one cluster
      4. Example: 
         1. Using Fuzzy c-Means Clustering to Analyze Gene Expression Data
            1. A team of biologists is analyzing gene expression data from microarrays to better understand the genes involved in normal and abnormal cell division. (A gene is said to be “expressed” if it is actively involved in a cellular function such as protein production.)
            2. The microarray contains expression data from two tissue samples. The researchers want to compare the samples to determine whether certain patterns of gene expression are implicated in cancer proliferation.
            3. After preprocessing the data to remove noise, they cluster the data. Because the same genes can be involved in several biological processes, no single gene is likely to belong to one cluster only.
            4. The researchers apply a fuzzy c-means algorithm to the data. They then visualize the clusters to identify groups of genes that behave in a similar way.
   2. **Gaussian Mixture Model**
      1. How It Works
         1. Partition-based clustering where data points come from different multivariate normal distributions with certain probabilities.
      2. Best Used...
         1. When a data point might belong to more than one cluster
         2. When clusters have different sizes and correlation structures within them
      3. Result: 
         1. A model of Gaussian distributions that give probabilities of a point being in a cluster
5. **Improving Models with Dimensionality Reduction**
   1. Machine learning is an effective method for finding patterns in number of features, or dimensionality.
   2. As datasets get bigger, you frequently need to reduce the big datasets. But bigger data brings added complexity.
   3. Example: EEG Data Reduction
      1. Suppose you have electroencephalogram (EEG) data that captures electrical activity of the brain, and you want to use this data to predict a future seizure. The data was captured using dozens of leads, each corresponding to a variable in your original dataset. Each of these variables contains noise. To make your prediction algorithm more robust, you use dimensionality reduction techniques to derive a smaller number of features. Because these features are calculated from multiple sensors, they will be less susceptible to noise in an individual sensor than would be the case if you used the raw data directly.
6. **Common Dimensionality Reduction Techniques**
   1. The three most commonly used dimensionality reduction techniques are:
      1. **Principal Component Analysis** (PCA)—
         1. performs a linear transformation on the data so that most of the variance or information in your high-dimensional dataset is captured by the first few principal components. The first principal component will capture the most variance, followed by the second principal component, and so on.
         2. Using Principal Component Analysis
            1. In datasets with many variables, groups of variables often move together. PCA takes advantage of this redundancy of information by generating new variables via linear combinations of the original variables so that a small number of new variables captures most of the information.
            2. Each principal component is a linear combination of the original variables. Because all the principal components are orthogonal to each other, there is no redundant information.
            3. Example: Engine Health Monitoring
               1. You have a dataset that includes measurements for different sensors on an engine (temperatures, pressures, emissions, and so on). While much of the data comes from a healthy engine, the sensors have also captured data from the engine when it needs maintenance.
               2. You cannot see any obvious abnormalities by looking at any individual sensor. However, by applying PCA, you can transform this data so that most variations in the sensor measurements are captured by a small number of principal components. It is easier to distinguish between a healthy and unhealthy engine by inspecting these principal components than by looking at the raw sensor data.
      2. **Factor Analysis**—
         1. identifies underlying correlations between variables in your dataset to provide a representation in terms of a smaller number of unobserved latent, or common, factors.
         2. Using Factor Analysis
            1. Your dataset might contain measured variables that overlap, meaning that they are dependent on one another. Factor analysis lets you fit a model to multivariate data to estimate this sort of interdependence.
            2. In a factor analysis model, the measured variables depend on a smaller number of unobserved (latent) factors. Because each factor might affect several variables, it is known as a common factor. Each variable is assumed to be dependent on a linear combination of the common factors.
            3. Example: Tracking Stock Price Variation
               1. Over the course of 100 weeks, the percent change in stock prices has been recorded for ten companies. Of these ten, four are technology companies, three are financial, and a further three are retail. It seems reasonable to assume that the stock prices for companies in the same sector will vary together as economic conditions change. Factor analysis can provide quantitative evidence to support this premise.
      3. **Nonnegative Matrix Factorization**—
         1. used when model terms must represent nonnegative quantities, such as physical quantities.
         2. Using Nonnegative Matrix Factorization
            1. This dimension reduction technique is based on a low-rank approximation of the feature space. In addition to reducing the number of features, it guarantees that the features are nonnegative, producing models that respect features such as the nonnegativity of physical quantities.
            2. Example: Text Mining
               1. Suppose you want to explore variations in vocabulary and style among several web pages. You create a matrix where each row corresponds to an individual web page and each column corresponds to a word (“the”,”a”,”we”, and so on). The data will be the number of times a particular word occurs on a particular page.
               2. Since there more than a million words in the English language, you apply nonnegative matrix factorization to create an arbitrary number of features that represent higher-level concepts rather than individual words. These concepts make it easier to distinguish between, say, news, educational content, and online retail content.
7. **Next Steps**
   1. In this section we took a closer look at hard and soft clustering algorithms for unsupervised learning, offered some tips on selecting the right algorithm for your data, and showed how reducing the number of features in your dataset improves model performance. 
   2. As for your next steps:
      - Unsupervised learning might be your end goal. 
        - For example, if you are doing market research and want to segment consumer groups to target based on web site behavior, a clustering algorithm will almost certainly give you the results you’re looking for.
      - On the other hand, you might want to use unsupervised learning as a preprocessing step for supervised learning. 
        - For example, apply clustering techniques to derive a smaller number of features, and then use those features as inputs for training a classifier.
   3. In section 4 we’ll explore supervised learning algorithms and techniques, and see how to improve models with feature selection, feature reduction, and parameter tuning.

## Applying Supervised Learning

1. **When to Consider Supervised Learning**
   1. A supervised learning algorithm takes a known set of input data (the training set) and known responses to the data (output), and trains a model to generate reasonable predictions for the response to new input data. Use supervised learning if you have existing data for the output you are trying to predict.
2. **Supervised Learning Techniques**
   1. All supervised learning techniques are a form of **classification** or **regression**.
      1. Classification techniques predict discrete responses—for example, whether an email is genuine or spam, or whether a tumor is small, medium, or large. Classification models are trained to classify data into categories. Applications include medical imaging, speech recognition, and credit scoring.
         1. Can your data be tagged or categorized? 
         2. If your data can be separated into specific groups or classes, use classification algorithms.
      2. Regression techniques predict continuous responses—for example, changes in temperature or fluctuations in electricity demand. Applications include forecasting stock prices, handwriting recognition, and acoustic signal processing.
         1. Working with a data range? 
         2. If the nature of your response is a real number—such as temperature, or the time until failure for a piece of equipment—use regression techniques.
3. **Selecting the Right Algorithm**
   1. As we saw in section 1, selecting a machine learning algorithm is a process of trial and error. It’s also a trade-off between specific characteristics of the algorithms, such as:
      1. Speed of training
      2. Memory usage
      3. Predictive accuracy on new data
      4. Transparency or interpretability (how easily you can understand the reasons an algorithm makes its predictions)
   2. Using larger training datasets often yield models that generalize well for new data.
   3. Let’s take a closer look at the most commonly used classification and regression algorithms.
4. **Binary vs. Multiclass Classification**
   1. When you are working on a classification problem, begin by determining whether the problem is binary or multiclass. In a binary classification problem, a single training or test item (instance) can only be divided into two classes—for example, if you want to determine whether an email is genuine or spam. In a multiclass classification problem, it can be divided into more than two—for example, if you want to train a model to classify an image as a dog, cat, or other animal.
   2. Bear in mind that a multiclass classification problem is generally more challenging because it requires a more complex model.
   3. Certain algorithms (for example, logistic regression) are designed specifically for binary classification problems. During training, these algorithms tend to be more efficient than multiclass algorithms.
5. **Common Classification Algorithms**
   1. **Logistic Regression**
      1. How it Works
         1. Fits a model that can predict the probability of a binary response belonging to one class or the other. Because of its simplicity, logistic regression is commonly used as a starting point for binary classification problems.
      2. Best Used...
         1. When data can be clearly separated by a single, linear boundary
         2. As a baseline for evaluating more complex classification methods
   2. **k Nearest Neighbor (kNN)**
      1. How it Works
         1. kNN categorizes objects based on the classes of their nearest neighbors in the dataset. kNN predictions assume that objects near each other are similar. Distance metrics, such as Euclidean, city block, cosine, and Chebychev, are used to find the nearest neighbor.
      2. Best Used...
         1. When you need a simple algorithm to establish benchmark learning rules
         2. When memory usage of the trained model is a lesser concern
         3. When prediction speed of the trained model is a lesser concern
   3. **Support Vector Machine (SVM)**
      1. How It Works
         1. Classifies data by finding the linear decision boundary (hyperplane) that separates all data points of one class from those of the other class. The best hyperplane for an SVM is the one with the largest margin between the two classes, when the data is linearly separable. If the data is not linearly separable, a loss function is used to penalize points on the wrong side of the hyperplane. SVMs sometimes use a kernel transform to transform nonlinearly separable data into higher dimensions where a linear decision boundary can be found.
      2. Best Used...
         1. For data that has exactly two classes (you can also use it for multiclass classification with a technique called error-correcting output codes)
         2. For high-dimensional, nonlinearly separable data
         3. When you need a classifier that’s simple, easy to interpret, and accurate
   4. **Neural Network**
      1. How it Works
         1. Inspired by the human brain, a neural network consists of highly connected networks of neurons that relate the inputs to the desired outputs. The network is trained by iteratively modifying the strengths of the connections so that given inputs map to the correct response.
      2. Best Used...
         1. For modeling highly nonlinear systems
         2. When data is available incrementally and you wish to constantly update the model
         3. When there could be unexpected changes in your input data
         4. When model interpretability is not a key concern
   5. **Naïve Bayes**
      1. How It Works
         1. A naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. It classifies new data based on the highest probability of its belonging to a particular class.
      2. Best Used...
         1. For a small dataset containing many parameters
         2. When you need a classifier that’s easy to interpret
         3. When the model will encounter scenarios that weren’t in the training data, as is the case with many financial and medical applications
   6. **Discriminant Analysis**
      1. How It Works
         1. Discriminant analysis classifies data by finding linear combinations of features. Discriminant analysis assumes that different classes generate data based on Gaussian distributions. Training a discriminant analysis model involves finding the parameters for a Gaussian distribution for each class. The distribution parameters are used to calculate boundaries, which can be linear or quadratic functions. These boundaries are used to determine the class of new data.
      2. Best Used...
         1. When you need a simple model that is easy to interpret
         2. When memory usage during training is a concern
         3. When you need a model that is fast to predict
   7. **Decision Tree**
      1. How it Works
         1. A decision tree lets you predict responses to data by following the decisions in the tree from the root (beginning) down to a leaf node. A tree consists of branching conditions where the value of a predictor is compared to a trained weight. The number of branches and the values of weights are determined in the training process. Additional modification, or pruning, may be used to simplify the model.
      2. Best Used...
         1. When you need an algorithm that is easy to interpret and fast to fit
         2. To minimize memory usage
         3. When high predictive accuracy is not a requirement
   8. **Bagged and Boosted Decision Trees**
      1. How They Work
         1. In these ensemble methods, several “weaker” decision trees are combined into a “stronger” ensemble.
         2. A bagged decision tree consists of trees that are trained independently on data that is bootstrapped from the input data.
         3. Boosting involves creating a strong learner by iteratively adding “weak” learners and adjusting the weight of each weak learner to focus on misclassified examples.
      2. Best Used...
         1. When predictors are categorical (discrete) or behave nonlinearly
         2. When the time taken to train a model is less of a concern
6. **Common Classification Algorithms Example** 
   1. Example: Predictive Maintenance for Manufacturing Equipment
      1. A plastic production plant delivers about 18 million tons of plastic and thin film products annually. The plant’s 900 workers operate 24 hours a day, 365 days a year.
      2. To minimize machine failures and maximize plant efficiency, engineers develop a health monitoring and predictive maintenance application that uses advanced statistics and machine learning algorithms to identify potential issues with the machines so that operators can take corrective action and prevent serious problems from occurring.
      3. After collecting, cleaning, and logging data from all the machines in the plant, the engineers evaluate several machine learning techniques, including neural networks, k-nearest neighbors, bagged decision trees, and support vector machines (SVMs). For each technique, they train a classification model using the logged machine data and then test the model’s ability to predict machine problems. The tests show that an ensemble of bagged decision trees is the most accurate model for predicting the production quality.
7. **Common Regression Algorithms**
   1. **Linear Regression**
      1. How it Works
         1. Linear regression is a statistical modeling technique used to describe a continuous response variable as a linear function of one or more predictor variables. Because linear regression models are simple to interpret and easy to train, they are often the first model to be fitted to a new dataset.
      2. Best Used...
         1. When you need an algorithm that is easy to interpret and fast to fit
         2. As a baseline for evaluating other, more complex, regression models
   2. **Nonlinear Regression**
      1. How It Works
         1. Nonlinear regression is a statistical modeling technique that helps describe nonlinear relationships in experimental data. Nonlinear regression models are generally assumed to be parametric, where the model is described as a nonlinear equation.
         2. “Nonlinear” refers to a fit function that is a nonlinear function of the parameters. For example, if the fitting parameters
            are b0, b1, and b2: the equation y = b0+b1x+b2x2 is a linear function of the fitting parameters, whereas y = (b0xb1)/(x+b2) is a nonlinear function of the fitting parameters.
      2. Best Used...
         1. When data has strong nonlinear trends and cannot be easily transformed into a linear space
         2. For fitting custom models to data
   3. **Gaussian Process Regression Model**
      1. How it Works
         1. Gaussian process regression (GPR) models are nonparametric models that are used for predicting the value of a continuous response variable. They are widely used in the field of spatial analysis for interpolation in the presence of uncertainty. GPR is also referred to as Kriging.
      2. Best Used...
         1. For interpolating spatial data, such as hydrogeological data for the distribution of ground water
         2. As a surrogate model to facilitate optimization of complex designs such as automotive engines
   4. **SVM Regression**
      1. How It Works
         1. SVM regression algorithms work like SVM classification algorithms, but are modified to be able to predict a continuous response. Instead of finding a hyperplane that separates data, SVM regression algorithms find a model that deviates from the measured data by a value no greater than a small amount, with parameter values that are as small as possible (to minimize sensitivity to error).
      2. Best Used...
         1. For high-dimensional data (where there will be a large number of predictor variables)
   5. **Generalized Linear Model**
      1. How it Works
         1. A generalized linear model is a special case of nonlinear models that uses linear methods. It involves fitting a linear combination of the inputs to a nonlinear function (the link function) of the outputs.
      2. Best Used...
         1. When the response variables have nonnormal distributions, such as a response variable that is always expected to be positive
   6. **Regression Tree**
      1. How It Works
         1. Decision trees for regression are similar to decision trees for classification, but they are modified to be able to predict continuous responses.
      2. Best Used...
         1. When predictors are categorical (discrete) or behave nonlinearly
8. **Common Regression Algorithms Example** 
   1. Example: Forecasting Energy Load
      1. Utility analysts at a large gas and electricity company developed models that predict energy demand for the following day. The models enable power grid operators to optimize resources and schedule power plant generation. Each model accesses a central database for historical power consumption and price data, weather forecasts, and parameters for each power plant, including maximum power out, efficiency, costs, and all the operation constraints that influence the plant dispatch.
      2. Analysts looked for a model that provided a low mean absolute percent error (MAPE) to the testing data set. After trying several different types of regression models, it was determined that neural networks provided the lowest MAPE due to their ability to capture the nonlinear behavior of the system.
9. **Improving Models**
   1. Improving a model means increasing its accuracy and predictive power and preventing overfitting (when the model cannot distinguish between data and noise). Model improvement involves feature engineering (feature selection and transformation) and hyperparameter tuning.
      1. **Feature Selection**:
         1. Identifying the most relevant features, or variables, that provide the best predictive power in modeling your data. This could mean adding variables to the model or removing variables that do not improve model performance.
         2. Feature selection is one of the most important tasks in machine learning. It’s especially useful when you’re dealing with high- dimensional data or when your dataset contains a large number of features and a limited number of observations. Reducing features also saves storage and computation time and makes your results easier to understand.
         3. Common feature selection techniques include:
            1. Stepwise regression: 
               1. Sequentially adding or removing features until there is no improvement in prediction accuracy.
            2. Sequential feature selection: 
               1. Iteratively adding or removing predictor variables and evaluating the effect of each change on the performance of the model.
            3. Regularization: 
               1. Using shrinkage estimators to remove redundant features by reducing their weights (coefficients) to zero.
            4. Neighborhood component analysis (NCA): 
               1. Finding the weight each feature has in predicting the output, so that features with lower weights can be discarded.
      2. **Feature Transformation**:
         1. Turning existing features into new features using techniques such as principal component analysis, nonnegative matrix factorization, and factor analysis.
         2. Feature transformation is a form of dimensionality reduction. As we saw in section 3, the three most commonly used dimensionality reduction techniques are:
            1. Principal component analysis (PCA): 
               1. Performs a linear transformation on the data so that most of the variance or information in your high-dimensional dataset is captured by the first few principal components. The first principal component will capture the most variance, followed by the second principal component, and so on.
            2. Nonnegative matrix factorization: 
               1. Used when model terms must represent nonnegative quantities, such as physical quantities.
            3. Factor analysis: 
               1. Identifies underlying correlations between variables in your dataset to provide a representation in terms of a smaller number of unobserved latent factors, or common factors.
      3. Hyperparameter tuning:
         1. The process of identifying the set of parameters that provides the best model. Hyperparameters control how a machine learning algorithm fits the model to the data.
         2. Like many machine learning tasks, parameter tuning is an iterative process. You begin by setting parameters based on a “best guess” of the outcome. Your goal is to find the “best possible” values— those that yield the best model. As you adjust parameters and model performance begins to improve, you see which parameter settings are effective and which still require tuning.
         3. Three common parameter tuning methods are:
            - Bayesian optimization
            - Grid search
            - Gradient-based optimization



