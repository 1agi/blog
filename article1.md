`Coursera/Deeplearning.AI`

Machine learning - Field of study that gives computers the ability to learn without being explicitly programmed. - Arthur Samuel (1959) [created checkers playing program]


### Types of machine learning algorithms
- Supervised learning - Rapid advancements, used in most real-world applications
- Unsupervised learning
	- Recommender systems
	- Reinforcement learning

## Supervised Learning
Learns from being given the **right answers**.

x -> y
(input) -> (output)

| Input(X)          | Output(Y)              | Application         |
| ----------------- | ---------------------- | ------------------- |
| email             | spam(0/1)              | spam filtering      |
| audio             | text transcripts       | speech recognition  |
| English           | Spanish                | machine translation |
| ad,user info      | click?(0/1)            | online advertising  |
| image, radar info | position of other cars | self-driving car    |
| image of phone    | defect?(0/1)           | visual inspection   |

#### **Regression: Housing price prediction**

fitting a given function to fit the data points in the dataset.
- straight line
- curve line
- any other function f(x)

Regression is predicting a number from infinitely many possible outputs.

#### **Classification**
Predicting small/limited/finite number of outputs/categories/classes from a small set of possible outputs.

**two or more inputs** can also be used to predict the outputs. [a boundary line is formed for classification]

## Unsupervised Learning
Finding something interesting in **unlabeled** data.
Data only comes with inputs x, but not output labels y.
Algorithm has to find **structure** in the data.

#### **Clustering**
group similar data points together.
can be divided in groups/clusters.
**clustering** is a form of unsup. learning algorithm.
this happens without supervisions.
clustering algorithm groups different datapoints into different categories, without giving the right answers.
they have to automatically group the entire dataset.

#### **Anomaly detection**
find unusual data points

#### **Dimensionality Reduction**
compress data using fewer numbers

## Linear Regression with One Variable

### Linear Regression Model
fitting a straight line to the data.
most widely used ml algo

ex- **House pricing prediction**
Regression model predicts numbers (infinitely many possible outputs).
Classification model predicts categories (small number of possible outputs).

- data tables can correspond to the data plots

#### **Terminology**
- Training set- data used to train the model
- x = "input" variable, feature
- y = "output" variable, "target" variable
- m = number of training examples
- (x,y) = single training example
- $(x^i, y^i)$ = ith training example; i is the index not the exponential



The training set has both input features and output targets.
<img src="https://i.imgur.com/Y6k15O3.jpeg" alt="fig1.0" width="500"/>

#### **How to represent f?**
(f is a straight line for now)

##### $f_{(w,b)} (x)$ = $wx + b$

f is a function that takes x as input and depending on the values of w and b, outputs some value of the prediction $\hat{y}$.

- When there is only a single feature x, Linear regression with one variable, also called Univariate linear regression.

- plotting the training set on a graph
<img src="https://i.imgur.com/eOXRHQl.jpeg" alt="fig1.0" width="500"/>
