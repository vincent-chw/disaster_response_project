import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Input
    messages_filepath: string, filetype: DataFrame
    categories_filepath: string, filetype: DataFrame
    
    Output
    DataFrame
    '''
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets
    df = pd.merge(messages,categories, on = ["id"])
    
    return df


def clean_data(df):
    '''
    Input
    df: DataFrame
    
    Output
    DataFrame
    '''
    
    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)
    
    # Select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # Create new column names for categories
    category_colnames = [x[:-2] for x in row]
    categories.columns = category_colnames
    
    # Convert category values to numeric 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.split('-').str.get(1)
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
        # change number 2 present in "related" to 1
        categories[column].replace({2: 1}, inplace=True)
    
    # Replace categories column in df with new category columns
    # Drop the original categories column from df
    df.drop(['categories'], axis=1, inplace=True)
    
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filepath):
    '''
    Input
    df: DataFrame
    database_filepath: filepath for database (sqlite:///DatabaseName.db)
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df.to_sql('MessageCategories', con = engine, index=False, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()