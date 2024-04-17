import numpy as np

param_sets={
  '0': {'id':0,
        'purpose':'reproducibility',
        'batch_size':np.inf,
        'training':None,
        'cases':['case11'],
        'scale':0,
        'rain_do':False,
#        'verbose':False,
        'verbose':1,
        'timesteps':5,
        'activation':['linear','linear'],
        'centering':[0.0,0.0],
        'hidden_units':6,
        'dense_units':1,
        'dense_layers':1,
        'DeltaE':[0,-1],    # -1.0 is to correct E bias but put at the end
        'synthetic':False,  # run also synthetic cases
        'T1': 0.1,          # 1/fuel class (10)
        'fm_raise_vs_rain': 2.0,         # fm increase per mm rain                              
        'epochs':5000,
        'verbose_fit':0,
        'verbose_weights':False,
        'note':'check 5 should give zero error'
        },
    
   '1':{'id':1,
        'purpose':'test 1',
        'batch_size':32,
        'training':5,
        'cases':'all',
        'scale':1,        # every feature in [0, scale]
        'rain_do':True,
        'verbose':False,
        'timesteps':5,
        'activation':['linear','linear'],
        'hidden_units':20,  
        'dense_units':1,    # do not change
        'dense_layers':1,   # do not change
        'centering':[0.0,0.0],  # should be activation at 0
        'DeltaE':[0,-1],    # bias correction
        'synthetic':False,  # run also synthetic cases
        'T1': 0.1,          # 1/fuel class (10)
        'fm_raise_vs_rain': 0.2,         # fm increase per mm rain 
        'train_frac':0.5,  # time fraction to spend on training
        'epochs':200,
        'verbose_fit':0,
        'verbose_weights':False,
        },
    
   '2':{'id':2,
        'purpose':'test 2',
        'batch_size':32,
        'training':5,
        'cases':'all',
        'scale':0.8,        # every feature in [0, scale]
        'rain_do':True,
        'verbose':False,
        'timesteps':5,
        'activation':['tanh','tanh'],
        'hidden_units':20,  
        'dense_units':1,    # do not change
        'dense_layers':1,   # do not change
        'DeltaE':[0,-1],    # bias correction
        'centering':[0.0,0.0],  # should be activation at 0
        'synthetic':False,  # run also synthetic cases
        'T1': 0.1,          # 1/fuel class (10)
        'fm_raise_vs_rain': 0.2,         # fm increase per mm rain 
        'train_frac':0.5,  # time fraction to spend on training
        'epochs':200,
        'verbose_fit':0,
        'verbose_weights':False,
       },
    
   '3':{'id':3,
        'purpose':'test 3',
        'batch_size':32,
        'training':5,
        'cases':'all',
        'scale':0.8,        # every feature in [0, scale]
        'rain_do':True,
        'verbose':False,
        'timesteps':5,
        'activation':['sigmoid','sigmoid'],
        'hidden_units':20,  
        'dense_units':1,    # do not change
        'dense_layers':1,   # do not change
        'DeltaE':[0,0],    # bias correction
        'centering':[0.5,0.5],  # should be activation at 0
        'synthetic':False,  # run also synthetic cases
        'T1': 0.1,          # 1/fuel class (10)
        'fm_raise_vs_rain': 0.2,         # fm increase per mm rain 
        'train_frac':0.5,  # time fraction to spend on training
        'epochs':200,
        'verbose_fit':0,
        'verbose_weights':False
       }
}
   
   
    