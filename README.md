# disaster_response_project
Disaster Response Project

This Disaster Response Project utilizes text data provided by Figure Eight to build a model that classifies disaster messages. 

The aim of the project is to have the machine learning model be able to classify unseen texts correctly. In the event of a disaster, many text messages are sent either direct or via social media. During that time, it will be very helpful that the important messages are filtered and pulled out for disaster response professionals so that they can take action accordingly. 

The data files disaster_categories.csv and disaster_messages.csv are cleaned for machine learning using the python script process_data.py in the data folder. The process_data.py will take the disaster_categories.csv and disaster_messages.csv as inputs and then output a cleaned dataset. The cleaned dataset is then saved into an sqlite database in the same folder. 

The dataset will then be used to train a machine learning classifier model using the python script train_classifier.py in the models folder. The train_classifier.py will take the cleaned dataset as input and train the machine learning model. Once the model training is finished, the model is exported out as a pickle file. 

The trained model shall then be used to predict and classify unseen texts input. Inside the app folder, there is a python script run.py and templates folder. The templates folder contains html scripts to display a simple website dashboard in the Flask web app where you can enter a text sentence and have the machine learning model attempt to classify the sentence. Running the run.py file will initiate the dashboard. 



Libraries used:

Numpy

Pandas

nltk

AdaBoostClassifier

Version:

Python version = 3.6.3

Numpy = 1.12.1

Pandas = 0.23.3
