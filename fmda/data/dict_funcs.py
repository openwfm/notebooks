def set_time_params(dict, h2, hours=None):
    # Given FMDA dictionary and time params, 
    # You CAN use this function just to change h2 (training period hours)
    # NOTE: this only performs a shallow copy and once you filter you can't filter original data to make larger dataset
    # Inputs: 
    # dict: fmda dictionary
    # h2: (int) length of training period
    # hours: (int) total number of hours (train + test)

    return_dict = dict.copy() 
    
    if not(hours is None):
        for case in return_dict:
            if len(return_dict[case]['fm']) < hours:
                raise ValueError(f"Number of hours ({hours}) is greater than length of fm data ({len(return_dict[case]['fm'])})")
            return_dict[case]['hours']=hours
            # Filter relevant fields (compatible with synthetic data)
            return_dict[case]['rain']=return_dict[case]['rain'][0:hours]
            return_dict[case]['Ed']=return_dict[case]['Ed'][0:hours]
            return_dict[case]['Ew']=return_dict[case]['Ew'][0:hours]
            return_dict[case]['fm']=return_dict[case]['fm'][0:hours]
            
            # Filter fields not in synthetic data
            if not('synthetic' not in return_dict[case]['title'].lower() or 'synthetic' not in return_dict[case]['descr'].lower()):
                return_dict[case]['solar']=return_dict[case]['solar'][0:hours]
                return_dict[case]['temp']=return_dict[case]['temp'][0:hours]
                return_dict[case]['rh']=return_dict[case]['rh'][0:hours]
                return_dict[case]['wind_speed']=return_dict[case]['wind_speed'][0:hours]
            
            
    
    # Loop through cases and change training period
    for case in return_dict:
        return_dict[case]['h2']=h2
        
    return return_dict