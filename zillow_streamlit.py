

import streamlit as st 
import pandas as pd
from IPython.core.display import display, HTML
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import os
from bs4 import BeautifulSoup
import time
import sys
import numpy as np
import pandas as pd
import regex as re
import requests
import seaborn as sns
import lxml
from lxml.html.soupparser import fromstring
import numbers
from sklearn.linear_model import LinearRegression
import prettify
import numbers
import htmltext
from sqlalchemy import create_engine
import json
import pickle
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from re import search
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (cross_val_score, train_test_split, 
                                     KFold, GridSearchCV)
import warnings
from datetime import datetime, timedelta, date
import datetime
from scipy import stats
warnings.filterwarnings('ignore')


def main():
    st.title("Zillow Comparative Analysis")
    st.sidebar.title("Analysis Details")
    st.sidebar.markdown("Ordinary least squares regression is based on square footage, number of beds, number of baths, tax assessment, year built, lot size, geographical coordinates, and days on zillow.  ")
    st.sidebar.markdown("Sales data is updated/re-scraped weekly")
    st.sidebar.markdown("Most similar homes based on prediction value from OLS regression. ")
    

#st.set_page_config(layout='wide')

if __name__ == "__main__":
    main()


input_url = st.text_input("Input a link:", value = 'https://www.zillow.com/homedetails/446-Weaver-St-Larchmont-NY-10538/33050705_zpid/')

#input_url = 'https://www.zillow.com/homedetails/42-Brookby-Rd-Scarsdale-NY-10583/33096166_zpid/'



req_headers = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'accept-encoding': 'gzip, deflate, br',
    'accept-language': 'en-US,en;q=0.8',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'
}


def get_html(url):

    # make a request to Zillow
    r = requests.get(url, headers=req_headers)
    content = r.content
    
    return content

def convert_to_soup(html):
    """Takes HTML object and converts to soup using BeautifulSoup"""
    soup = BeautifulSoup(html, 'html.parser')
    
    return soup



def beds(soup):
    try:
        details = soup.find('meta', attrs={'name': 'description'})
        beds=re.findall(r"(\d+) bed", str(details))
        return int(beds[0])
    except:
        return np.nan 
    
def sqft(soup):
    try:
        details = soup.find('meta', attrs={'name': 'description'})
        unclean_deets=(re.findall(r'\d{1,2}[,.]\d{1,3} sq', str(details)))
        sqft1=re.findall(r'\d{1,2}[,.]\d{1,3}', str(unclean_deets))
        sqft=int(sqft1[0].replace(',',''))
        return sqft
    except:
        return np.nan 
def baths(soup):
    try:
        details = soup.find('meta', attrs={'name': 'description'})
        unclean_deets=(re.findall(r'\d{1,2}[,.]\d{1,3} bath', str(details)))
        bath1=re.findall(r'\d{1,2}[,.]\d{1,3}', str(unclean_deets))
        bath=float(bath1[0].replace(',',''))
        return bath
    except:
        return np.nan 
    
    
def zillow_scrape(url):
    # get html for the url and convert to soup
    html = get_html(url)
    soup = convert_to_soup(html)
    script = soup.select("script#hdpApolloPreloadedData")[0]
    script = script.contents[0]
    script_json = json.loads(script)
    features_string = script_json['apiCache']
    api_features_dict = json.loads(features_string)
    api_key_2 = list(api_features_dict.keys())[1]
    #print(api_features_dict)
    dict2_feat={
        'price':api_features_dict[api_key_2]['property']['price'],
        'city': api_features_dict[api_key_2]['property']['city'],
        'street_address': api_features_dict[api_key_2]['property']['streetAddress'],
        'state': api_features_dict[api_key_2]['property']['state'],
        'zipcode': api_features_dict[api_key_2]['property']['zipcode'], 
        'full_address': api_features_dict[api_key_2]['property']['streetAddress']+", "+ api_features_dict[api_key_2]['property']['city']+", "+api_features_dict[api_key_2]['property']['state'],
        'latitude': api_features_dict[api_key_2]['property']['latitude'],
        'longitude': api_features_dict[api_key_2]['property']['longitude'],
        'year_built': api_features_dict[api_key_2]['property']['yearBuilt'],
        'lot_size': api_features_dict[api_key_2]['property']['lotSize'],
        'days_on_zillow': api_features_dict[api_key_2]['property']['daysOnZillow'],
        'zestimate': api_features_dict[api_key_2]['property']['zestimate'],
        'tax_assessment': api_features_dict[api_key_2]['property']['taxAssessedValue'],
        "beds"  : beds(soup),
        "baths" : baths(soup),
        "sqft" : sqft(soup),
        "links" : url
    }
    return dict2_feat


try:
    input_house=zillow_scrape((input_url))
except: 
    st.write("Incorrect link or no link input")


    
    
with open('/Users/jenniferhilibrand/Metis/Data_Engineering/Zillow_App/zillow_timestamps.pickle', 'rb') as to_read: 
    timestamp_dict = pickle.load(to_read)
d0= date(2020, 8, 18)
today = datetime.date.today()
try:
    time_between = today - timestamp_dict[input_house['city']]
except: 
    time_between = today - d0

if time_between.days > 30:
    ###SCRAPE PROCESS####
    timestamp_dict[input_house['city']]=today
    town_url_link=input_house['city']+'-'+input_house['state']

    with requests.Session() as s:
        url2 = 'https://www.zillow.com/'+town_url_link+'/sold/'
        url1=s.get(url2, headers=req_headers)
    html1=BeautifulSoup(url1.content,"html.parser")
    li_list=[i.text for i in html1.find_all('li')]

    test=[j for j in li_list if "Page" in j]
    pages=test[0]
    pages2=re.search(r'\d+$', pages)
    page_num_sold=int(pages2.group())if pages2 else None

    num=range(1,page_num_sold+1)
    #num=range(1,3)
    url_list=[]
    for i in num:
        try:
            with requests.Session() as s:
                url = 'https://www.zillow.com/'+town_url_link+'/sold/'+str(i)+"_p/"
                url_list.append(s.get(url, headers=req_headers))
        except:
            pass
    soup_list_sold=[]
    for i in url_list:
        soup_list_sold.append(BeautifulSoup(i.content,"html.parser"))

    listings=[]
    for soup in soup_list_sold:
        for i in soup:
            listings.extend(soup.find_all("a", class_="list-card-link"))
    home_links=[]
    for i in listings: 
        home_links.append(i['href'])

    #unique_links=home_links[::2]
    unique_links=list(dict.fromkeys(home_links))
    #pickle_in = pd.read_sql( "SELECT * FROM zillow ;",engine)
    with open('/Users/jenniferhilibrand/Metis/Data_Engineering/Zillow_App/df_zillow_data.pickle', 'rb') as to_read: 
        pickle_in = pickle.load(to_read)
                                 
                                     
    boolean_series=pickle_in.links.isin(unique_links)
    relevant_df=pickle_in[boolean_series]
    relevant_df.append(pickle_in[pickle_in.city == input_house['city']])

    for i in unique_links:
        if i in list(relevant_df['links']):
            unique_links.remove(i)

    #######################################
    town_dicts=[]
    for i in unique_links: 
        try:
            town_dicts.append(zillow_scrape((i))) #this takes roughly 1 second a house 
        except IndexError:
            pass
    df=pd.DataFrame(town_dicts)
    data_df=pd.concat([relevant_df,df])
    data_df.drop_duplicates(subset='links',inplace=False)
    ###SCRAPE PROCESS####
    
    with open('/Users/jenniferhilibrand/Metis/Data_Engineering/Zillow_App/zillow_timestamps.pickle', 'wb') as to_save:
        pickle.dump(timestamp_dict ,to_save) 

    with open('/Users/jenniferhilibrand/Metis/Data_Engineering/Zillow_App/df_zillow_data.pickle', 'rb') as to_read: 
        pickle_in = pickle.load(to_read)

    pickle_out=pd.concat([pickle_in,data_df])
    pickle_out_final=pickle_out.drop_duplicates(subset='links',inplace=False)

    with open('/Users/jenniferhilibrand/Metis/Data_Engineering/Zillow_App/df_zillow_data.pickle', 'wb') as to_save:
        pickle.dump(pickle_out_final,to_save)
    
    
else: 
    with open('/Users/jenniferhilibrand/Metis/Data_Engineering/Zillow_App/df_zillow_data.pickle', 'rb') as to_read: 
        pickle_in = pickle.load(to_read)
    data_df=pickle_in[pickle_in.city == input_house['city']]




#pickle_out_final.shape

tax_url = "https://stylinandy-taxee.p.rapidapi.com/v2/state/2020/"+input_house['state']

headers = {
    'authorization': "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUElfS0VZX01BTkFHRVIiLCJodHRwOi8vdGF4ZWUuaW8vdXNlcl9pZCI6IjYwY2I1ODdiYTRkNDk5M2I2YmIzOGFmNCIsImh0dHA6Ly90YXhlZS5pby9zY29wZXMiOlsiYXBpIl0sImlhdCI6MTYyMzk0MDkyOX0.7n0loBZcIC2xQpH4RKjiajmow2DbGPkWdEAHWUcsfkc",
    'x-rapidapi-key': "8951dd6fc2msh87cf2fc0cee41cep1ac2f2jsn7175def81ef2",
    'x-rapidapi-host': "stylinandy-taxee.p.rapidapi.com"
    }

response = requests.request("GET", tax_url, headers=headers)
tax_dict=response.json()
brackets=[]
rates=[]
tax_df=pd.DataFrame

try:
    for i in tax_dict['head_of_household']['income_tax_brackets']:
        brackets.append(i['bracket'])
        rates.append(i['marginal_rate'])
        tax_df=pd.DataFrame(brackets)
        tax_df['rates']=rates
except KeyError: 
        tax_df=("No Income Tax!")


        
df_analysis=data_df.dropna()

#remove outliers, nulls, add the input home (and drop any null columns from the input home)
outlier_mask=(df_analysis['price']<50000)
dropme_outliers=df_analysis[outlier_mask].index
df_analysis.drop(dropme_outliers,inplace=True)     
df_analysis=df_analysis.dropna()
df_total = df_analysis.append(input_house, ignore_index=True).dropna(axis=1) 


#split out test and train data 
X_overall = df_total.drop(['price'],axis=1).select_dtypes(include=np.number)

#X_overall=df_total.loc[:,['latitude','longitude', 'year_built',	'lot_size',	 'tax_assessment', 'beds',	'baths','sqft']].select_dtypes(include=np.number)

Y_overall=df_total['price']
X_train, X_test, Y_train, Y_test = train_test_split(X_overall, Y_overall, test_size=0.2,random_state=44)


X_sample=X_train
Y_sample=Y_train
lm=LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state = 44)
scorelist_lin0=cross_val_score(lm, X_sample, Y_sample, cv=kf, scoring='r2')


lm.fit(X_overall, Y_overall)

total_preds=lm.predict(X_overall)


#total_preds=lm.predict(df_total.loc[:,['latitude','longitude', 'year_built','lot_size',	 'tax_assessment', 'beds',	'baths','sqft']])

df_total['predicted']=total_preds
df_total_sorted = df_total.sort_values(by=['predicted'])

house=df_total_sorted.loc[df_total_sorted['full_address']==input_house['full_address']]



predicted_price = int(house['predicted'])



df_total_sorted['ishouse']=df_total_sorted.full_address.isin((house['full_address']))
idx = df_total_sorted[df_total_sorted['ishouse']==1].index.tolist()
idx=idx[0]
big_slice = df_total_sorted.loc[idx:]
similar_homes1 = big_slice.iloc[1:3]
big_slice2 = df_total_sorted.loc[:idx]
similar_homes2 = big_slice2.iloc[-3:-1]
similar_homes=pd.concat([similar_homes1, similar_homes2])

similar_homes['over_under'] = similar_homes['price']-similar_homes['predicted']


fig = sns.jointplot(x=total_preds,y=df_total['price'], kind='reg', color='#99CCFF')
sns.set(rc={"figure.figsize":(1,1)})

fig.ax_joint.scatter(similar_homes['predicted'], similar_homes['price'], color = '#E8BF87')
fig.ax_joint.scatter(house['predicted'], house['price'], color = 'red')
       
price_spread= int(house['price']-predicted_price)


import locale
locale.setlocale(locale.LC_ALL, '') 



try:
    bed_perc=stats.percentileofscore(df_total['beds'].array, house['beds'].item()) 
    st.write("Beds:",locale.format("%d",int(house['beds']), grouping=True), " -> ", locale.format("%d", bed_perc, grouping=True), "Percentile")
except: 
    st.write("No beds found")

try:
    bath_perc=stats.percentileofscore(df_total['baths'].array, house['baths'].item())
    st.write("Baths:",locale.format("%d",int(house['baths']), grouping=True), " -> ", locale.format("%d", bath_perc, grouping=True), "Percentile")
except: 
    st.write("No baths found")

try:
    lotsize_perc=stats.percentileofscore(df_total['lot_size'].array, house['lot_size'].item())
    st.write("Lot Size:",locale.format("%d",int(house['lot_size']), grouping=True),"sqfeet", " -> ", locale.format("%d", lotsize_perc, grouping=True), "Percentile")
except: 
    st.write("No lot size found")

try: 
    sqft_perc=stats.percentileofscore(df_total['sqft'].array, house['sqft'].item())
    st.write("Square Footage:",locale.format("%d",int(house['sqft']), grouping=True), " -> ", locale.format("%d", sqft_perc, grouping=True), "Percentile")
except: 
    st.write("No square footage found")
#st.write("Beds:", ,", Baths:" , locale.format("%d",int(house['baths']), grouping=True),", SqFt:", locale.format("%d",int(house['sqft']), grouping=True))


st.write("This home is currently priced at: ", locale.format("%d",int(house['price']), grouping=True))

st.write("My predicted price for this home: ", locale.format("%d",int(house['predicted']), grouping=True))

    
try: 
    st.write("The Zestimate for this home is: ",  locale.format("%d",input_house['zestimate'], grouping=True))
except: 
    st.write("No Zestimate!")
             
             
if price_spread < 0: 
    st.write("This house is underpriced by:", locale.format("%d",price_spread , grouping=True))
if price_spread > 0:
    st.write("This house is overpriced by: ", locale.format("%d",price_spread , grouping=True))
    

try:
    st.write("It was assessed at: ", locale.format("%d",int(house['tax_assessment']), grouping=True))
except: 
     st.write("No tax assessment")


st.write("Average home sale price of our comparison group: ", locale.format("%d",int(np.mean(df_analysis['price'])), grouping=True))

st.write("Number of comp houses:", len(df_analysis['price']))

results_table = similar_homes.loc[:,['full_address','predicted','over_under', 'price', 'links']]

st.markdown("Most similar homes:")
st.dataframe(results_table)


st.write("Average Cross Validation R^2 of our Regression", np.mean(scorelist_lin0))

st.markdown("OLS regression and density plot:")
st.pyplot(fig,figsize=(0.5,2))
#st.write(graph2)

try:
    tax_df.columns = ['Income Bracket barrier', 'Income tax % ']
    st.write("State income tax brackets:", tax_df)
except:
    st.write("State income tax brackets:", tax_df)


