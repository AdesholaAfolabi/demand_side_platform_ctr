import pandas as pd
import numpy as np

dictionary = {'Bid':0,'Click':1}
joined_columns = ['os_name', 'os_version', 'browser_name', 'browser_version', 'app_name', 'site_name', 'site_link', 'captured_time', 'data_type']
top_attr = ['os_name_version', 'browser_name_version', 'inventory']
numerical_columms = ['device_screen_height', 'device_screen_width', 'device_screen_pixel_ratio']
columns_with_nan = ['os_name', 'browser_name', 'location_region', 'location_state', 'app_name', 'site_name', 'site_link', 'carrier', 'inventory', 'os_vendor', 'os_version', 'browser_version', 'os_name_version', 'browser_name_version']
#The idea here is to add the app_name and site_name together to have a singular column inventory
def inventory(data,column1= 'app_name', column2 = 'site_name'):
    data['inventory'] = data[column1].replace(np.nan,'')+ data[column2].replace(np.nan,'')
    data['inventory'] = data['inventory'].replace('',np.nan)
#The split function gets the site_name from site link and helps reduce the Nan values 
def split(data,column1,column2):
    data[column2] = data[column1].str.split('/', expand = True)[2]
#A lot of columns were missing for region. The states column had less missing value. Since each state belong to a geographical region, it made sense to map them accordingly
def replace_region(data):
    region = data['location_state']
    if region in ['Benue','Kogi','Kwara','Nasarawa','Niger','Plateau','Abuja','Makurdi','Lokoja','Asokoro']:
        return 'North_Central'
    elif region in ['Adamawa','Bauchi','Borno','Gombe','Taraba','Yobe','Jos','Minna','Maiduguri','Yola']:
        return 'North_East'
    elif region in ['Jigawa','Kaduna','Kano','Katsina','Kebbi','Sokoto','Zamfara','Zaria','Dutse']:
        return 'North_West'
    elif region in ['Abia','Anambra','Ebonyi','Enugu','Imo','Abakaliki','Owerri','Umuahia']:
        return 'South_East'
    elif region in ['Akwa Ibom','Cross River','Bayelsa','Rivers','Delta','Edo','Benin City','Port Harcourt','Asaba',
                   'Warri','Nsukka','Calabar','Uyo','Yenagoa','Eket','Sagbama','Bonny','Effurun']:
        return 'South_South'
    elif region in ['Ekiti','Lagos','Ogun','Ondo','Osun','Oyo','Ikeja','Ikire','Badagri','Ikorodu','Ibadan',
                   'Suleja','Ilorin','Abeokuta','Osogbo','Akure','Ede','Ikotun','Lekki','Ikoyi','Ota','Ojota',
                   'Sagamu','Ogudu','Mowe','Agege','Omu-Aran','Aponri']:
        return 'South_West'
#This function takes care of missing values in the os_vendor column. Values from os_name can be used to fix this.
def replace_vendor(os_name,os_vendor):
    if 'Macintosh' in str(os_name):
        return 'Apple'
    elif 'X11' in str(os_name):
        return 'Microsoft'
    else:
        return os_vendor
#This cleans the time column and separates each value into day, hour and minute 
def clean_time(value):
    value = value.replace('T', ' ')
    value = value[:19]
    return value
def process_time(df):
    df['captured_time'] = df['captured_time'].apply(lambda x:clean_time(x))
    df['captured_time'] = pd.to_datetime(df['captured_time'])
    df['day'] = df['captured_time'].dt.day.astype('uint8')
    df['hour'] = df['captured_time'].dt.hour.astype('uint8')
    df['minute'] = df['captured_time'].dt.minute.astype('uint8')
#This functions clear Nans and replaces them with others for categorical variables and replaces numerical variables with the mean
def clear_nan(data):
    for item in columns_with_nan:
        data[item] = data[item].fillna("Others")
def clear_nan_numeric(data):
    for item in numerical_columms:
        data[item] = data[item].fillna(data[item].mean())
#This function picks top 50 values in a column and replaces every other value with others    
def pick_top_attr(data):
    for item in top_attr:
        top_val = list(data[item].value_counts()[:50].index)
        data[item] = data[item].apply(lambda x : x if x in top_val else "other")
#This is a general cleanup function. Converts columns to strings and joins columns together based on relevance.
def column_cleanup(data):
    data['y'] = data['data_type'].map(dictionary)
    data['os_version'] = data.os_version.astype(str)
    data['browser_version'] = data.browser_version.astype(str)
    data['location_region'] = data.apply(lambda x: replace_region(x), axis=1)
    data['os_vendor'] = data.apply(lambda x: replace_vendor(x['os_name'], x['os_vendor']), axis=1)
    data['os_name_version'] = data['os_name'] + data['os_version']
    data['browser_name_version'] = data['browser_name'] + data['browser_version']

def preprocessing_file(data):    
    process_time(data)
    split(data, 'site_link', 'site_name')
    inventory(data)
    column_cleanup(data)
    clear_nan_numeric(data)
    clear_nan(data) 
    #pick_top_attr(data)
    training_data = data.drop(joined_columns, axis=1)
    return training_data
