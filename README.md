# Sentiment Analysis using Amazon Customer Review Data

## Problem Statement

This project focused on analyzing customer reviews and ratings from Amazon dataset. The project aimed to predict the sentiment of customers depending on their reviews.  The project started with finding a suitable dataset, creating a business problem and requirements document, data cleaning, exploratory data analysis, model training and testing, and finally, deploying the model using Flask and AWS EC2.
We had a dataset of more that 10,000 rows form Amazon customers based in UK. We wrote all our code in Python Pandas to clean our data and then train our model using Logistic Regression model. After that we built a flask app to deploy our ML model on AWS EC2 instance using its cloud UNIX environment. Hence, our application is running perfectly on the web with 88% accuracy and predicting the results most of the time.



### Steps followed 

- Step 1 : Importing Python libraries is an essential part of any project that involves writing code in Python. 

![image](https://github.com/user-attachments/assets/ece5ae6f-8a38-4efc-9930-5e987b780f91)

A library is a collection of pre-written code that we can use to save time and avoid having to write the same code yourself. To import a Python library, we use the import keyword, followed by the name of the library. Once a library is imported, we can use the functions, classes, and other objects defined in the library in your code.

- Step 2 : Reading json data file- Reading JSON data files in Python is a common task in data analysis and web development projects.

![image](https://github.com/user-attachments/assets/934162c4-d7ed-4536-9071-8cc2e411d704) 

JSON (JavaScript Object Notation) is a lightweight data interchange format that is easy for humans to read and write and for machines to parse and generate.

- Step 3 : Loading CSV data files into SQLite is a common task in data analysis and database management projects. 

![image](https://github.com/user-attachments/assets/4665ff13-0461-4e95-872b-dd1c5e5167a4)

SQLite is a lightweight database management system that is widely used in embedded systems and mobile applications. It provides a simple and efficient way to store and retrieve data using SQL (Structured Query Language).

- Step 4 : Cleaning, filtering, and pre-processing data is an essential step in data analysis and machine learning projects. It involves identifying and handling missing or incorrect data, removing irrelevant or duplicate information, and transforming the data into a format suitable for analysis or modeling. Python provides a rich set of libraries and functions for data cleaning, filtering, and pre-processing.We filter the data to focus on specific subsets using functions in pandas. These functions allow you to filter rows and columns based on specific criteria or conditions. After filtering the data, the next step is to pre-process the data by transforming it into a format suitable for analysis or modeling. This can involve scaling or normalizing the data using functions, encoding categorical variables using functions in pandas, or performing feature extraction.

- Step 5 : Data Visualizations: 
        Libraries used – 

SNS: Seaborn is a Python data visualization library based on Matplotlib that provides a high-level interface for creating informative and attractive statistical graphics. 

Matplotlib: Matplotlib is a popular data visualization library in Python that provides a wide range of tools for creating static, animated, and interactive plots. It allows users to create a variety of visualizations such as line plots, scatter plots, histograms, bar charts, and heatmaps, among others.

![image](https://github.com/user-attachments/assets/71759fd6-314b-4d5a-998b-6342d85a32d8)

The graph illustrates the distribution of customer reviews based on ratings.


![image](https://github.com/user-attachments/assets/9eefafe4-2941-421c-b86a-fde778d862eb)

The graph shows the distribution of customer reviews to the sentiment type.
Based on the graph, there are more positive reviews from customers than negative ones.

![image](https://github.com/user-attachments/assets/585ae5b4-a054-44a4-b5c0-014d8314aabd)

A word cloud was generated using the word cloud library to display the most frequent words used by customers in their reviews.
The word cloud was created regardless of the sentiment expressed in the dataset.

![image](https://github.com/user-attachments/assets/912583be-5a83-4956-8ecc-a447c5709d67)

The word cloud library was utilized to generate a plot that displays the most frequent negative words used by customers in their reviews within the dataset.

![image](https://github.com/user-attachments/assets/12aee88c-6fd4-434b-a494-2565929bb8bc)

The Word cloud library was utilized to generate a plot that displays the most frequent positive words used by customers in their reviews within the dataset.

- Step 6 : Machine Learning Models : A machine learning model is a software application that can identify patterns or make decisions based on new data sets that it has not previously encountered. Machine learning methods can be categorized into three main types: supervised learning, unsupervised learning, and reinforcement learning.

- Step 7 :Logistic Regression is a supervised learning technique that is employed to ascertain whether a given input belongs to a particular group or not.Naive Bayes is also a supervised learning classification algorithm that assumes that variables are independent and calculates probabilities to determine the class of an object based on its features. Multinomial Naive Bayes is well-suited for problems where the frequency of each feature's occurrence is significant, while Bernoulli Naive Bayes is better suited for problems where the presence or absence of each feature carries more weight.

![image](https://github.com/user-attachments/assets/c5704e23-2009-458b-b45b-12149ca6305b)

Process: To implement a machine learning model on train and test data, the model must be trained on the training dataset and assessed for its effectiveness on the test dataset.

- Step 8 : Model comparison with ROC-The ROC (Receiver Operating Characteristic) is a graphical method for assessing the performance of a binary classifier. It compares the true positive rate to the false positive rate at different threshold levels. 

![image](https://github.com/user-attachments/assets/49ad428a-1a67-4782-a41f-c1d3bc366feb)

The area under the curve (AUC) is a metric that reflects the classifier's ability to differentiate between positive and negative classes, with a higher AUC indicating better performance. The code segment plots ROC curves and calculates AUC values for various machine learning models to compare their classification performance. Logistic Regression model has highest AUC (0.99) among three models. 


- Step 9 : Function classification report
The `classification report` function presents a summary of the key classification metrics, such as precision, recall, f1-score, and support, for each class in the predicted and actual target values. It also includes weighted average metrics for the entire classification task, offering an extensive report of a model's classification performance.

![image](https://github.com/user-attachments/assets/7a1fda05-1d77-4a5f-9571-97c6dd934d47)
![image](https://github.com/user-attachments/assets/a1d83c11-12f1-4636-a3f7-2187a387933f)

- Step 10 : Function plot_confusion_matrix()

![image](https://github.com/user-attachments/assets/f8d79fd3-83dd-4da4-8bef-2e0022bab1d2)

The function plot_confusion_matrix() creates a heatmap using matplotlib to display the input confusion matrix. The confusion matrix is calculated for the Logistic Regression model using the confusion_matrix() 
function. Two types of confusion matrices are plotted: the non-normalized matrix and the normalized matrix. The non-normalized confusion matrix shows the number of actual and predicted values, whereas the normalized confusion matrix shows the proportion of accurate predictions for each class. The normalized matrix is particularly useful when there is an imbalance in the number of instances in each class since it enables a more equitable comparison of model performance across different classes. The function sets the title and axis labels for the plot and applies a color map to visualize the matrix. Finally, the plots are displayed using the plt.show() function.



- Step 11 : Conclusion:
In conclusion, the Amazon Customer Reviews and Rating Analysis project successfully analyzed customer sentiment on Amazon based on their reviews, providing insights into customer satisfaction, and identifying areas that require improvement. The project successfully met all the milestones, overcame the identified risks and problems, and deployed the model using Flask and AWS EC2. The insights generated from the project can be used to improve customer satisfaction and make data-driven decisions. However, the stakeholders need to continuously monitor model performance and data quality to ensure the insights generated remain relevant and accurate over time.
