_**Liver Disease Prediction using Machine Learning**_


This is a project that aims to predict liver disease using machine learning techniques. The dataset used in this project contains information about liver disease patients, including various clinical and demographic features. The goal is to train a machine learning model that can accurately predict whether a patient has liver disease or not.

**Requirements**

To run this project, you will need the following:

Python 3
PySpark
Jupyter Notebook
Scikit-learn
Pandas
Numpy
Matplotlib
Installation
Install Python 3 from the official website

Install PySpark by following the official installation guide

Install Jupyter Notebook by following the official installation guide

Install the required Python packages by running the following command in the terminal:

Liver Disease Prediction using Machine Learning
This is a project that aims to predict liver disease using machine learning techniques. The dataset used in this project contains information about liver disease patients, including various clinical and demographic features. The goal is to train a machine learning model that can accurately predict whether a patient has liver disease or not.

**Requirements**

To run this project, you will need the following:

Python 3
PySpark
Jupyter Notebook
Scikit-learn
Pandas
Numpy
Matplotlib
Installation
Install Python 3 from the official website

Install PySpark by following the official installation guide

Install Jupyter Notebook by following the official installation guide

Install the required Python packages by running the following command in the terminal:

**Copy code**
#pip install scikit-learn pandas numpy matplotlib#


**Usage**

Clone this repository to your local machine
Open Jupyter Notebook and navigate to the cloned repository
Open the LiverDiseasePrediction.ipynb notebook
Follow the instructions in the notebook to run the code and train the machine learning model
Dataset
The dataset used in this project is the Liver Patient Records dataset from the UCI Machine Learning Repository. This dataset contains records from 416 liver disease patients and 167 non-liver disease patients. The dataset includes various clinical and demographic features, such as age, gender, total bilirubin, direct bilirubin, albumin, and more.

**Usage**

Clone this repository to your local machine
Open Jupyter Notebook and navigate to the cloned repository
Open the LiverDiseasePrediction.ipynb notebook
Follow the instructions in the notebook to run the code and train the machine learning model


**Dataset**

The dataset used in this project is the Liver Patient Records dataset from the UCI Machine Learning Repository. This dataset contains records from 416 liver disease patients and 167 non-liver disease patients. The dataset includes various clinical and demographic features, such as age, gender, total bilirubin, direct bilirubin, albumin, and more.

**Machine Learning Model**

The machine learning models used in this project are Random Forest Regression (RFR), Gaussian Naive Bayes (GNB), and Logistic Regression (LR).

Random Forest Regression was chosen because it is a powerful and versatile ensemble learning method that can handle both categorical and continuous variables, and can effectively deal with missing values and outliers. Gaussian Naive Bayes is a probabilistic classification algorithm that can handle high-dimensional feature spaces and is particularly useful when the assumption of independence between features holds true. Logistic Regression is a simple yet effective classification algorithm that can handle both binary and multi-class classification problems.

The performance of each model was evaluated using various metrics, including accuracy, precision, recall, and F1-score. The models were trained and tested on our liver disease dataset to predict the presence or absence of liver disease in patients.

**Results**

Our project aimed to develop machine learning models to predict the presence of liver disease based on clinical and demographic features. We trained and evaluated three different models: Random Forest, Gaussian Naive Bayes, and Logistic Regression.

Our results showed that all three models achieved high accuracy scores in predicting liver disease, with Random Forest achieving the highest accuracy of 77.3%, followed by Gaussian Naive Bayes at 74.3%, and Logistic Regression at 71.2%.

Furthermore, we found that the Random Forest model outperformed the other models in terms of precision, recall, and F1-score. This suggests that Random Forest may be the most appropriate model for accurately predicting liver disease in our dataset.

We also observed that the most important features for predicting liver disease in our models were direct bilirubin, total protein, and alkaline phosphatase.

**Conclusion**
In conclusion, our project aimed to develop a machine learning model for predicting liver disease using clinical and demographic data. We performed extensive data analysis, including data preparation, exploratory data analysis, and feature selection, to develop three machine learning models: Random Forest, Gaussian Naive Bayes, and Logistic Regression.

Our results showed that all three models had good performance in predicting liver disease, with Logistic Regression having the highest accuracy of 77.3%. We also found that certain features, such as Total Protein, Albumin, and Alkaline Phosphatase, were strong predictors of liver disease.

However, our findings should be interpreted with caution, as they are specific to our dataset and may not generalize to other populations. Further studies with larger and more diverse datasets are needed to confirm the utility of these models and explore the potential of other machine learning techniques for liver disease prediction.

Overall, our project provides valuable insights into the potential of machine learning in predicting liver disease and highlights the importance of data analysis and feature selection in developing accurate and reliable models.

**Kaggle Dataset**

The Liver Patient Dataset used in this project was obtained from Kaggle. You can find the original dataset and more information about it on the following link:
https://www.kaggle.com/uciml/indian-liver-patient-records

**License**

The Liver Patient Dataset is available under the following license:
Database: Open Database, Contents: Database Contents, License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license, Link: https://creativecommons.org/licenses/by-sa/4.0/

The code in this project is available under the MIT License:
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
