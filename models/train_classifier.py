import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import pickle
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier

from sqlalchemy import create_engine

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM MessageCategories", con = engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    
    return X, Y, Y.columns


def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()

    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens


def build_model():
    # Create pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('multiclf', MultiOutputClassifier(AdaBoostClassifier()))])
    
    # Perform GridSearchCV to locate best parameter
    # Specify parameters for grid search
    parameters = {
        'multiclf__estimator__n_estimators': [100, 300],
        'multiclf__estimator__learning_rate':[1.0, 0.5]
    }
    
    # Initialize GridSearchCV to run with specified parameters
    cv = GridSearchCV(pipeline, param_grid = parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Input
    model: build_model()
    X_test: Feature test set
    Y_test: Target test set
    category_names: List of column names
    '''
    print('Category Names', category_names)
    # Predict
    y_pred = model.predict(X_test)

    # Iterate through columns and calling classification_report on each column
    for count, column in enumerate (category_names):
        class_report = classification_report(Y_test.iloc[:,count].values, y_pred[:,count])
        accuracy = (Y_test.iloc[:,count] == y_pred[:,count]).mean()
        print("\nColumn:\n", column)
        print("Classification Report:\n", class_report)
        print("Accuracy:", accuracy)

def save_model(model, model_filepath):
    '''
    Input
    model: Trained model
    model_filepath: Filename of the pickle file
    
    Output
    Pickle file
    '''
    filename = model_filepath
    outfile = open(filename,'wb')
    pickle.dump(model,outfile)
    outfile.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format('sqlite:///' + database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()