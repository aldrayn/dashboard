import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, pointbiserialr

def convert_to_original_scale(data_) :
    
    data = data_.copy()
    
    data['season'] = data['season'].map({   1 : 'springer', 
                                                2 : 'summer', 
                                                3 : 'fall', 
                                                4 : 'winter'
    })

    data['yr']   = np.where(data['yr'] == 0, '2011', '2012')

    data['mnth'] = data['mnth'].map({       1 : 'Jan',  2 : 'Feb', 
                                            3 : 'Mar',  4 : 'Apr',
                                            5 : 'May',  6 : 'Jun', 
                                            7 : 'Jul',  8 : 'Aug', 
                                            9 : 'Sep', 10 : 'Oct', 
                                            11: 'Nov', 12 : 'Dec'})

    data['holiday'] = np.where(data['holiday'] == 0, 'No', 'Yes')
    data['weekday'] = data['weekday'].map({ 
                                                1 : 'Mon', 
                                                2 : 'Tue', 
                                                3 : 'Wed', 
                                                4 : 'Thr', 
                                                5 : 'Fri',
                                                6 : 'Sat',
                                                0 : 'Sun'
    })

    data['workingday'] = data['workingday'].map({0 : 'No', 1 : 'Yes'})
    data['weathersit'] = data['weathersit'].map({1 : 'Normal',   2 : 'Moderate', 3 : 'Caution', 4 : 'Extreme'})
    
    data['temp']  = round(41  * data['temp'])
    data['atemp'] = round(50  * data['atemp'])
    data['hum']   = round(100 * data['hum'])

    data['windspeed'] = round(67  * data['windspeed'])
    data['date']  = [d[-2:] for d in data['dteday']]
    
    return data

def pearson(data_) :
    
    data = convert_to_original_scale(data_)
    
    return data.select_dtypes('number').corr()

def crammersv(data_, col1, col2) :
    multi_multi = pd.DataFrame()

    for a in col1 :

        for b in col2 :    
            contingen_table = pd.crosstab(index = data_[a], columns = data_[b])

            chi2, p, dof, expected_freq = chi2_contingency(contingen_table)

            n = np.sum(np.sum(contingen_table))

            v = np.sqrt(chi2/(n * (min(contingen_table.shape) - 1)))

            multi_multi.loc[a, b] = v

    return multi_multi

def phi_coefficient(data_, biner_col) :
    bin_bin = pd.DataFrame()

    for a in biner_col :
        for b in biner_col :
            contingency_table = pd.crosstab(data_[a], data_[b])

            chi2, p, dof, expected = chi2_contingency(contingency_table)

            phi_coefficient = np.sqrt(chi2 / data_.shape[0])

            bin_bin.loc[a, b] = phi_coefficient
    
    return bin_bin

def pointbiserial(data_) :
    num_bin = pd.DataFrame()
    
    data = convert_to_original_scale(data_)

    numeric_col = data.select_dtypes('number').columns
    biner_col   = ['yr', 'holiday', 'workingday']

    for num in numeric_col :
        for bin in biner_col :
            r_pb, p_value = pointbiserialr(data_[bin], data_[num])

            num_bin.loc[num, bin] = r_pb

    return num_bin
