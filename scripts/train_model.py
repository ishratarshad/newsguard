#Importing the required module to do the IDF calculation 
#importing pandas for the csv document we need to process.   

from sklearn.feature_extraction.text import TfidfVectorizer 
import pandas as pd 

cleaned_data_set = pd.read_csv("C:\\Users\\rayat\\OneDrive\\Desktop\\MyProjects\\NewsGuardAi\\newsguard\\datasets\\cleaned_data_fixed.csv")   

tfidf = TfidfVectorizer() 

result = tfidf.fit_transform(cleaned_data_set['cleaned_text'])  

#get the idf values from the fitted vectorizer 
idf_values = tfidf.idf_  #here we are basically just accessing the idf values directly where the idf_ attribute stores the IDF values 

print("\nIDF values:") 
for word, idf in zip(tfidf.get_feature_names_out(), idf_values):
    print(f"{word}: {idf}") 
    
    