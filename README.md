# Data Science with R and R Studio

## Prerequisites
* Download and install R. You can download it [here](https://cran.r-project.org/mirrors.html).
* Install 'R Studio Desktop'. You can download it [here](https://www.rstudio.com/products/rstudio/download/).

## Data Mining Cup 1 - Car Insurance Conversion Prediction

### Description
This is a dataset from one bank in the United States. Besides usual services, this bank also provides car insurance services. The bank organizes regular campaigns to attract new clients. The bank has potential customers’ data, and bank’s employees call them for advertising available car insurance options. We are provided with general information about clients (age, job, etc.) as well as more specific information about the current insurance sell campaign (communication, last contact day) and previous campaigns (attributes like previous attempts, outcome).
You have data about 4000 customers who were contacted during the last campaign and for whom the results of campaign (did the customer buy insurance or not) are known.

### Classification Task
The task is to predict for 1000 customers who were contacted during the current campaign, whether they will buy car insurance or not.

### Used Algorithm
For the first task, we used the [Decision Tree model](https://en.wikipedia.org/wiki/Decision_tree). The accuracy of the used model is around **82%**.

## Data Mining Cup 2 - Car Engine Parts Failure Rates

### Description
The topic of the second data mining cup is predictive maintenance, which is currently a very popular application area of predictive analytics.
The dataset is about predicting defects of a specific engine part in cars. There are at least two use cases, where a sufficient prediction model is of high value for both the car manufacture and the car driver: During a car inspection, the model can be used to indicate that the engine part should be replaced to prevent a future breakdown. While driving, the driver could be warned about an imminent engine defect and the chances of an accident are decreased when the driver brings the car to inspection.
Originally, the dataset is from a project between a leading German car manufacturer and Alexander Thamm Data Science Services GmbH. We highly appreciate that both have agreed to sponsor the dataset for academic purposes. The project is already finished and at the end of the data mining cup Alexander Thamm’s data scientists will share insight of their solution. Furthermore, the three best teams will be awarded with an additional prize on top of the exam bonus points.

### Classification Task
Within the scope of this dataset, a defect is defined as a failure of a specific engine component. Your task is to determine whether a system readout, including several different information about the car and engine, can be used to predict if the engine part will be defective.

### Used Algorithm
For the second task, we used the [Generalized Linear Model (GLM)](https://en.wikipedia.org/wiki/Generalized_linear_model). The accuracy of the used model is around **79%**.

## Improving the Results
There are numerous ways to improve the accuracy of the predictions.

- Applying **ensemble methods** like bagging, boosting and stacking would push the accuracy close to 85%.
- **Feature engineering** would push results between 85% to 90% accuracy.
- **Neural Networks**, also known as **Deep Learning**, has surpassed all other algorithms in terms of accuracy. Accuracy beyond 90% is possible with tools like e.g. TensorFlow.