# Data Engineering Project Proposal

Question: 
Can cloud computing and and big data pipelines be used to create a meaningful app/interface for visualizing and regressing housing data? 

Data Description: 
My primary data set will be Zillow webscraping done with Beautiful Soup (if I can overlay an API like walkscore or census data, I will try to add this as well). I aim to create a storage and update process for which houses or areas that have been previously searched can be updated or calculated quickly. This will involve 1) creating a dynamic and resilient webscrapping system for zillow 2) using cloud computing/efficient data storage and 3) building an interactive app (streamlit or flask) that can provide basic visualizations and statistics about relevant housing data. 

Tools/MVP: 
I will be using Beautiful Soup for my webscraping. I will be doing the majority of my analysis in pandas, and will utlize seaborn and Scikit-learn for analysis. I will be using google cloud and a remote computer for comutation. I will use either Flask or Streamlit as my tool to build this in a web app form. An MVP for this project would be a data pipeline that can run through the scrape, storage, calculate, and visualize components of this app. 
