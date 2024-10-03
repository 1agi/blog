
<br>

## Supervised Machine Learning


### Regression and Classification


What is Machine Learning?
Field of study that gives computers the ability to learn without being explicitly programmed. - Arthur Samuel (1959) [checkers players program]

##### ML Algorithms
- Supervised Learning (Used in most in-real world application)
- Unsupervised Learning
- Recommendation Systems
- Reinforcement Learning

"Practical advice for applying learning algo"

##### Supervised Learning

x     ---->   y    
input ---->   output      (mappings)

learns from being given the 'right answers' (y:output labels)

| Input(x)          | Output(y)              | application         |
| ----------------- | ---------------------- | ------------------- |
| email             | spam?(0/1)             | spam filtering      |
| audio             | text transcripts       | speech recognition  |
| english           | spanish                | machine translation |
| ad,user info      | click?(0/1)            | online advertising  |
| image, radar info | position of other cars | self-driving cars   |
| image of phoone   | defect?(0/1)           | visual              |

###### Regression
Regression is predicting a number from infinitely many possible outputs.

![fig1.0](https://imgur.com/LW5rI30)

- **Classification (Class/Category)**
Classification predict categories from a small number of possible outputs.

![png](ss2.png)

- multiple types of outputs plotted on a straight line.

![png](ss3.png)

- two or more inputs

![png](ss4.png)

##### Unsupervised Learning

Supervised Learning: learns from data 'labelled' with the 'right answers'
Unsupervised Learning: find something interesting in 'unlabelled data' (Clustering)

![png](ss5.png)


- **Clustering Algorithm**
is used by Google News, DNA micro-array, grouping customers

In Unsupervised Learning;
Data only comes with input 'x', but not output labels 'y'
Algorithm has to find structure in the data.
- Clustering: group similar data points together
- Dimensionality reduction: compress data using fewer numbers
- Anomaly detection: find unusual data points

##### Regression Model
###### Linear Regression

Supervised learning model data has right answers,
Classification model predicts categories - small number of possible outputs
`Regression model` predicts numbers from infinitely many possible outputs

![png](ss6.png)

- Data Table represents relation between inputs and outputs well

- Terminology
	- Training set: data used to train the model
- Notation
	- x = input variable / feature
	- y = output variable / target
	- m = number of training examples
	- (x,y) = single training example
	- ($x^{(i)}$, $y^{(i)}$) = $i^{th}$ training example


[ training set ] features,targets --> [ learning algorithm ] --> $f$

x(feature) --> [ f ] hypothesis/function --> $y^{hat}$ (prediction) estimated y(y is target)

**How to represent f?**
(f being a straight line as a foundation)

$f_{w,b}(x) = wx+b$ (y=wx+b)
$f_(x) = wx+b$  { straight line function/ linear function }

![png](ss7.png)

Linear Regression with one variable(single feature x) is also called Univariate linear regression
- [[Linear Regression w one variable Lab]]

##### Cost Function
Model:   $$f_{w,b}(x) = wx + b $$
Parameters/coefficients/weights: $w,b$

![png](ss8.png)

**Cost function**:   $$J(w,b) = \frac{1}{2m} \sum\limits_{i = 1}^{m} y^{cap(i)} - y^{(i)})^2 \tag{1}$$
m = number of training examples
$y^{cap(i)} - y^{(i)}$ = error
$(y^{cap(i)} - y^{(i)})^2$ = squared error cost function; most commonly used for Linear Regression

The prediction ($y^{cap}$) = output of the model $f_{w,b}(x^{(i)})$

![png](ss9.png)

so, 
  $$J(w,b) = \frac{1}{2m} \sum\limits_{i = 1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 $$

**what does $J(w,b)$ compute?**
model: $$f_{w,b}(x) = wx + b $$parameters: $w,b$
cost function:   $$J(w,b) = \frac{1}{2m} \sum\limits_{i = 1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 $$
goal: $minimize_{w,b}$ $J(w,b)$ 

![png](ss10.png)
`Explanation `
You want to fit a straight line to the training data, so you have model $f_(x)=w(x)+b$. Depending on different values chosen for w and b, you get different straight lines, and you want to find the values of w and b, so the straight lines fit the training data, you have a cost function J, and what J does is, it measures the difference between the models predictions and the actual true values (y). 
What you see later is the that, it is that LR would try find values w and bm the makes J of w and b as small as possible.


**simplified version of LR Model**: b = 0
so, $f_(x)=w(x)$ 
cost function:   $$J(w,b) = \frac{1}{2m} \sum\limits_{i = 1}^{m} (f_{w}(x^{(i)}) - y^{(i)})^2 $$
goal: $minimize_{w}$ $J(w)$

plotting it we see, when b is set to 0, then f defines a line that passes from the origin, because when x=0, f(x)=0.
using this simplified model, lets see how cost function changes as we chose different values for the parameter w.

**Visualizing graphs of the model f(x) and cost function J**

- for fixed w, the fw is only a function of x(input); the estimated value of y depends on the input x
- J is function of w, where w controls the slope of the line defined by fw; so the cost defined by J depends on parameter w

- w=1; J(1)=0
line f is a slope of value 1
![png](ss11.png)

- w=0.5, J(0.5)=0.58
line f is an upward slope of value 0.5
![png](ss12.png)

- w=0, J(0)=2.3
line f(fx=0) is a horizontal line on x-axis 
![png](ss13.png)

- w=-0.5, J(-0.5)=
line f is a downward slope of value -0.5
![png](ss14.png)

by continuing to compute the cost function J for different values of w and plotting we can trace-out what J(w) looks like.
![png](ss15.png)
`Explanation`
each value of parameter w corresponds to a different straight line fits f(x) on left-graph, and for the given training set that choice for the value of w corresponds to single point on right-graph, because for each value of w we can calculate the cost J(w).

so, **how to chose w that results function f(line f) fitting the data well?**
choosing the value of w that causes J(w) to be as small as possible seems like a good bet.
J is a cost function that measures how big the squared errors are, so choosing w that minimizes these squared errors(make them as small as possible) will give us a good model.

**goal of linear regression:** $minimize_{w}$ $J(w)$
**general case:** $minimize_{w,b}$ $J(w,b)$

##### **Cost function with b**
model: $$f_{w,b}(x) = wx + b $$parameters: $w,b$
cost function:   $$J(w,b) = \frac{1}{2m} \sum\limits_{i = 1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2 $$
goal: $minimize_{w,b}$ $J(w,b)$ 

##### graph of **$f_{(w,b)}$**
![png](ss16.png)
##### graph of **$J_{(w,b)}$**
**$J$** function of w,b : soup bowl (3-dimensional)
![png](ss17.png)

plotting J using a contour plot
**slice the 3D plot of J(w,b) horizontally to get the contour plot**.
now each horizontal slice is shown as one of the ovals in the contour plot.
![png](ss18.png)
the three points shown by(c=green,orange,blue) are at the same height, i.e., same value of J.
All three points have the same value of the cost function J, even though they have different values for w and b. and they also correspond to different function f.
Therefore each horizontal slice ends up being shown as one of these ellipses/ovals.

**The bottom of the bow where cost function J is minimum is the central oval of contour**.
Contour plots are a convenient way to visualize the 3D cost function J in 2D.
 
###### plotting graphs using different values of w and b

- w=-0.15 b=800
![png](ss19.png)

- w=0 b=360
![png](ss20.png)

- w=-0.15 b=500
![png](ss21.png)

- w=0.13 b=71
![png](ss22.png)
If you measure the vertical distance of the data points, to predict the values on the straight lines, you get the error for each data point. The sum of squared errors for all these data points is pretty close to the minimum possible sum of squared errors of among all possible straight line fits.

- [[Cost Function Lab]]

##### Gradient Descent
Have some function $J_(w,b)$ ; Want $min_{w,b}$ $J(w,b)$ 
for linear regression, or

it can minimize any function
$min_{(w1,w2,...wn,b)}$ $J(w1,w2,...wn,b)$ that gives smallest possible value of J

outline:
start with some w,b (set w=0, b=0)
keep changing w,b to reduce J(w,b)
until we settle at or near a minimum
there may be more than 1 possible minimum values.

for linear regression with squared error cost function(convex function), we always end up wit a bow/hammock shaped graph and have a single global minimum. 
Unlike the graph below.


![png](s23.png)

**Gradient Descent Algorithm**
$$\begin{align*}  w &= w -  \alpha \frac{\partial J(w,b)}{\partial w}\newline 
 b &= b -  \alpha \frac{\partial J(w,b)}{\partial b}
\end{align*}$$

$\alpha$ = learning rate (**size of step**); 0<$\alpha$<1
### $\frac{\partial J(w,b)}{\partial w}$
is the derivative of cost function J (**which direction you want to take the step**)

Repeat until convergence, simultaneously update w and b.

- correct way: simultaneous update
temp_w = w - $\alpha$ dJ_dw(w,b)
temp_b = b - $\alpha$ dJ_db((w,b)

w = temp_w
b = temp_b

###### Gradient Descent Intuition
using simple implementation with b=0

$J(w)$
$$\begin{align*}w &= w -  \alpha \frac{\partial J(w)}{\partial w} \end{align*}$$
goal: $min_{w}$ $J(w)$

- choosing some value for w and plotting
![png](ss24.png)

#### **Learning Rate** $(\alpha)$
![png](ss25.png)

- what if it already at local minima?
![png](ss26.png)
if your parameters are reaching the local minimum then further gradient descent steps do nothing, keeping the value at local minimum,

- Gradient Descent can reach local minimum with fixed learning rate
![png](ss27.png)

##### Training Linear Regression
**Gradient Descent for Linear Regression**
- Linear Regression Model: $$f_{w,b}(x) = wx + b $$
- Cost Function (squared error): $$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 $$
- Gradient Descent Algorithm: $$\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline
\;  w &= w -  \alpha \frac{\partial J(w,b)}{\partial w} \newline 
 b &= b -  \alpha \frac{\partial J(w,b)}{\partial b}  \newline \rbrace
\end{align*}$$
where, parameters $w$, $b$ are updated simultaneously.  


> calculating the partial derivatives dJ_dw and dJ_db we get

**The gradient is defined as:**
$$
\begin{align}
\frac{\partial J(w,b)}{\partial w}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)} \\
  \frac{\partial J(w,b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) \\
\end{align}$$

#### **Gradient descent algorithm:** 

$$\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline
\;  w &= w -  \alpha\frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)}; \newline 
 b &= b -  \alpha \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})  \newline \rbrace
\end{align*}$$
- Local minimum multiple(golf park with hills), have multiple local minimum with different values of w and b, you can end up at different local minimums
- In squared error cost function with linear regression, the cost function will never have multiple local minimum, it has a single global minimum because of its bowl shape. The cost function is a "convex function". therefore, as long as the learning rate is chosen properly, it will always converge to the global minimum.
- Batch gradient descent: "Batch": each step of gradient descent uses all the training examples.

**Running Gradient descent algorithm**
![png](ss23.1.png)

- [[Gradient Descent for Linear Regression Lab]]

---

### Linear Regression with multiple features

![png](ss2.1.png)

### $x^{(i)}{j}=x^{(m)}_{n}$
$x$ is a matrix of dimension [m x n]

m/i = no. of rows/ith row/no. of examples
n/j = no. of columns/jth column/no. of features

- $x_{1},x_{2},x_{3},x_{4}$ denote the features, to represent this list of features 
- n = **number of features**
- ##### $x_{j}$= $j^{th}$ feature 
- ##### $x\vec{}\ ^{(i)}$ = 
**features of ith training example** (x with arrow denotes it is a vector; a list of numbers/parameters of the model)

here, $x\vec{}\ ^{(2)}$ will be a list of 4 numbers [1416,3,2,40]
or, **a vector** that includes all the features of ith training example. (which is a row vector)

- ### $x\vec{}\ ^{(i)}_{j}$ = 
**value of feature j in ith training example**
### $x^{(i)}_{j}$
### $x\vec{}\ ^{(2)}_{3}$ = 2

##### Model: previously, 
#### $f_{w,b}$ = $wx+b$

Now, for multiple features,
#### $f_{w,b}$ = $w_{1}x_{1}+w_{2}x_{2}+w_{3}x_{3}+w_{4}x_{4}+b$

![png](ss2.2.png)

for each additional (size,bedroom,floor,years) the price may increase or decrease, with base price(b).

fw,b(x) is a model with n features.
$w\vec{}$  here is a vector, a list of numbers/parameters of the model
$x\vec{}$ is also a vector
![png](ss2.3.png)
it's more succinct and compact to use the dot product 
#### $w\vec{}\cdot{} x\vec{}$  
rather than multiplying individually or using a for loop.

The linear regression with multiple input features is called Multiple Linear Regression.

#### Vectorization
Vectorization makes the code of learning algorithm both shorter and makes it run much faster and more efficiently. Learning how to write vectrozied code allows you to take advantage of modern numerical algebra libraries as well as GPU hardware.

- example of vectorization and its benefits
![png](ss2.4.png)
behind the scenes the numpy dot function is able to use parallel hardware in computer to compute.
which makes it much more efficient than the for loop or the sequential calculation.

- what is it doing behind the scenes
![png](ss2.5.png)

- how it helps in linear regression with multiple feature (ex. with 16 features/parameters)
![png](ss2.6.png)
vectorization makes a huge difference when there are a large number or features and huge training sets, the difference is quite substantial.

- [[Vectorization Lab]]

##### Gradient Descent for Multiple Linear Regression with Vectorization

Using previous models and parameteres, we turn them into a Vector notaion
![png](ss2.7.png)

###### Gradient Descent with multiple features, 
- w and x are now vectors, 
### $x^{(i)}$ is $x_{1}^{(i)}$ 
- since there can be multiple features j = 1 to n
j(w1,w2,...wn)
so we have $w_{n}$ and b, implementing which we get gradient descent for multiple regression.

![png](ss2.8.png)

- Normal equation can only be used for linear regression, solves for w and b without iterations.
![png](ss2.9.png)

- [[Multiple Linear Regression Lab]]

#### Feature Scaling
If the data set has features with significantly different scales, one should apply feature scaling to speed gradient descent.

The parameters values w are (w0,w1,..wn) and features x(x0,x1,...xn), since the magnitude order of features can vary greatly with each other, so it is better to scale the feature sizes to have more comparable values for plotting and calculations.
![pmg](ss3.1.png)
- Feature Size and parameter size
![png](ss3.2.png)
- Feature size and gradient descent

The contour plot with rescaled features are more asymmetric as compared to actual values
![pmg](ss3.3.png)
#### Normalization
- divide by max
![png](ss3.4.png)
- Mean Normalization
![png](ss3.5.png)
- Z-score normalization
uses standard deviation
![png](ss3.6.png)
- when to do feature scaling
![png](ss3.7.png)
- checking gradient descent for convergence
It is always a good idea to plot the learning curve (Cost vs Iterations) to make a general observation whether the gradient descent is working as intended.
![pmg](ss3.8.png)
- choosing appropriate learning rate
It is either 
	- learning rate alpha is too large
	- there is a bug in the code
![pmg](ss3.9.png)
- Try different values for $\alpha$ and see which is just right
start with 0.001 and then 3X the values in each step
![png](ss3.10.png)
- [[Feature Scaling and Learning Rate]]
#### Feature Engineering and Polynomial Regression

As described before, if the data set has features with significantly different scales, one should apply feature scaling to speed gradient descent.
![png](ss3.11.png)
- polynomial regression
In the example, there is $x$, $x^2$ and $x^3$ which will naturally have very different scales. 
Applying Z-score normalization(Feature Scalling) to our example will allows this to converge much faster.

![png](ss3.12.png)

With feature engineering, even quite complex functions can be modeled:
![png](ss3.13.png)

- [[Feature Engineering and Polynomial Regression]]

- [[Linear Regression using Scikit-Learn]]

---
