# _**Liver Disease Prediction using Machine Learning**_


The application of machine learning techniques in liver disease prediction has shown great promise in recent years. However, the development of accurate and reliable machine learning models requires comprehensive data about the disease and its progression, as well as high-quality EDA and feature classification. By reviewing the current state-of-the-art machine-learning techniques for liver disease prediction, this study aims to provide valuable insights that can inform the development of more effective and accurate machine-learning models for predicting liver disease and its progression.

# **Requirements**

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

# **Kaggle Dataset**

The Liver Patient Dataset used in this project was obtained from Kaggle. You can find the original dataset and more information about it on the following link:
https://www.kaggle.com/datasets/abhi8923shriv/liver-disease-patient-dataset

# **Requirements**

To run this project, you will need the following:

* Python 3
* PySpark
* Jupyter Notebook
* Scikit-learn
* Pandas
* Numpy
* Matplotlib
Installation
* Install Python 3 from the official website

* Install PySpark by following the official installation guide

* Install Jupyter Notebook by following the official installation guide

* Install the required Python packages by running the following command in the terminal:

# **Copy code**
* pip install scikit-learn pandas numpy matplotlib PySpark 


# **Usage**

To train the machine learning model for our liver disease prediction project, follow these steps:

1. Clone the repository to your local machine.
2. Open Jupyter Notebook and navigate to the cloned repository.
3. Open the LiverDiseasePrediction.ipynb notebook.
4. Follow the instructions in the notebook to run the code and train the machine learning model.

Our dataset includes records from both liver disease patients and non-liver disease patients. Specifically, it consists of 20,000 training data and ~1,000 test data points, with 10 variables such as age, gender, total bilirubin, direct bilirubin, albumin, and more. This dataset was used to predict liver disease using various machine learning techniques.


# **Dataset**

The Liver Patient Dataset contains 20,000 training data and approximately 1,000 test data, with 10 variables including age, gender, total bilirubin, direct bilirubin, total proteins, albumin, A/G ratio, SGPT, SGOT, and alkphos. The dataset also includes a selector field used to split the data into two sets: labeled by the experts as 1 for Liver Patient and 2 for Mon Liver Patient. The dataset was obtained from an unknown source during an unknown time period. We would like to thank the Kaggle dataset community for inspiration. The dataset inspires us to predict liver patients, and we hope to contribute to the largest data science community with our findings.

# **Machine Learning Model**

Machine learning techniques such as Random Forest Regression (RFR), Gaussian Naive Bayes (GNB), and logistic regression (LR) have been used extensively in predicting liver disease. These supervised learning methods are capable of handling complex and high-dimensional data, which is often encountered in medical data with multiple features. In this literature review \cite{RFRLiverDiseasePrediction}, we will explore various machine learning techniques applied in different domains that are relevant to liver disease prediction.

# **Results**

In this paper, our goal was to predict the presence of liver disease with high accuracy. To achieve this, we examined the given features and found significant correlations among certain blood test features. We attempted to classify each feature using normal ranges found online to understand their relationship with the target value better. However, it is important to note that our findings are limited to our specific dataset and may not generalize to other populations or datasets. Therefore, further studies are needed to validate our findings and explore the potential of other machine-learning techniques for liver disease prediction. Furthermore, it is important to exercise caution when interpreting the results of any machine learning model, taking into account potential factors such as bias and variability in the data. We recommend future research be conducted in collaboration with medical professionals to ensure accurate distribution and interpretation of results.

# **Conclusion**

In this study, we aimed to predict liver disease and its progression using machine learning models. Our analysis revealed that Random Forest had the highest performance in predicting liver disease.   

The importance of feature classification in machine learning was also highlighted in our study. We found that Total Bilirubin, Direct Bilirubin, Albumin, A/G Ratio, SGPT, SGOT, and Alkphos were important predictors for liver disease prediction. Interestingly, we did not find a significant impact of age and gender on liver disease prediction.

Our work also emphasized the significance of correlation evaluation between the features, which helped us get an accurate model. Ends with the importance of using not only one evaluator to determine between the models (AUC) but also understanding the particular field of the problem we are aiming to solve and another potential evaluator, such as recall which we also aimed to maximize.

Overall, our study demonstrated the potential of machine learning models for predicting liver disease and provided insights into the important features of this task. Our work also highlighted the need for further research in this area and the importance of evaluating model performance in healthcare applications.


# **License**

The Liver Patient Dataset is available under the following license:
Database: Open Database, Contents: Database Contents, License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license, Link: https://creativecommons.org/licenses/by-sa/4.0/

The code in this project is available under the MIT License:
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
