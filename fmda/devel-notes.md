the network training and prediction are in run_case(case_data,params) called from fmda_rnn_rain.ipynb

fmda_rnn_rain.ipynb:

    run_case
    
moisture_rnn.py:

    def run_case(case_data,params, check_data=False)
        run_rnn(case_data,params,...) + plots, stats
            
        def run_rnn
            create_rnn_data_1() to make  arrays X features and Y target
            create_rnn_data_2() to create x_train and y_train
            train_rnn (rnn_dat,params,hours,fit)
            rnn_predict(model, params, rnn_dat)
            

fmda_pkl_hrrr.ipynb:

    call run_pkl_case

moisture_rnn_pkl.py:

        def run_rnn_pkl
            pkl2train already made arrays X features and Y target
            create_rnn_data_2() to create x_train and y_train
            call rnn_predict
            
test-pkl2train.ipynb:

    pkl2train make  arrays X features and Y target
