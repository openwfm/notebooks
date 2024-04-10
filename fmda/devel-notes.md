the network training and prediction are in run_case(case_data,params) called from fmda_rnn_rain.ipynb

moisture_rnn.py: 

    def run_case(case_data,params, check_data=False)
        calls run_rnn(case_data,params,...) 
            calls train_rnn (rnn_dat,params,hours,fit)
                runs over one stretch <============ generalize to multiple atretches
            calls rnn_predict(model, params, rnn_dat)