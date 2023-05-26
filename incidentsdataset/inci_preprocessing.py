import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from scipy.sparse import save_npz

def load_incident_dataset():
    data = pd.read_csv('incidentsdataset/incidents.csv.gz', compression='gzip', na_values='?', parse_dates=[9,11,13,34,35], 
                       dayfirst=True, infer_datetime_format=True)
    data.drop_duplicates('number', keep='first', inplace=True)
    print("Add Target Columns...")
    data['time_to_close'] = data.apply(lambda row: (row.closed_at - row.opened_at).total_seconds() / (24*60*60), axis=1)
    data['time_to_resolve'] = data.apply(lambda row: (row.resolved_at - row.opened_at).total_seconds() / (24 * 60 * 60), axis=1)

    data['weekday'] = data['opened_at'].map(lambda x: x.dayofweek)
    data['late'] = data['opened_at'].map(lambda x: x.hour > 14)
    #data['date'] = data['date_time'].map(lambda x: x.strftime('%Y-%m-%d'))
    
    data.loc[data['assignment_group'].isna(), 'assignment_group'] = "group_na"
    data = data[data['time_to_resolve'] > 0.0]
    
    data['resolve_transformed'] = np.log1p(data['time_to_resolve'])
    
    f, (ax0, ax1) = plt.subplots(1, 2)

    ax0.hist(data['time_to_resolve'], bins=100)
    ax0.set_ylabel('Probability')
    ax0.set_xlabel('Target')
    ax0.set_title('Target distribution')

    ax1.hist(data['resolve_transformed'], bins=100)
    ax1.set_ylabel('Probability')
    ax1.set_xlabel('Target')
    ax1.set_title('Transformed target distribution')

    f.suptitle("Time to Resolve", y=0.035)
    f.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    
    plt.show()
    data.to_csv('incidentsdataset/test_prep.csv')
    
    ct = ColumnTransformer(transformers = [
        #('num', StandardScaler(), ['temp','rain_1h','snow_1h']),
        ('cat', OneHotEncoder(), ['impact', 'urgency', 'priority', 'notify', 'weekday', 'assignment_group', 'contact_type']),
        ('bool', 'passthrough', ['knowledge', 'u_priority_confirmation', 'made_sla', 'late']),
        ('minmax', MinMaxScaler(), ['reassignment_count', 'sys_mod_count']),
        ('target', 'passthrough', ['time_to_resolve'])
        #('target', FunctionTransformer(np.log1p), ['time_to_resolve'])
        ], sparse_threshold = 1.0)
    
    p = ct.fit_transform(data[data['time_to_resolve'] > 0.0])
    
    print(p)

    return p

p = load_incident_dataset()

save_npz('incidentsdataset/npz_incidents.csv', p)