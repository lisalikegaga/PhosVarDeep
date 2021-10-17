
PhosVarDeep
=========
PhosVarDeep: a deep-learning model for phospho-variant prediction using sequence information
Developer: 
Liu Xia from Health Informatics Lab, School of Information Science and Technology, University of Science and Technology of China

Requirement
=========
    keras==2.0.0
    numpy>=1.8.0
    backend==tensorflow

Related data information needs to first load
=========
testdata.csv

The input file is an csv file, which includes proteinName, postion, reference sequences，mutation and labels.

ori_var_input.py helps to get variant sequences, and get testdata_ori.csv, testdata_var.csv, which separately includes reference sequences and corresponding variant sequences.




Predict for your test data
=========
If you want to use the model to predict your test data, you must prepare the test data (testdata_ori.csv, testdata_var.csv) as an csv file, the first column is protein Name, the second col: position, the third col: sequences 

The you can run the predict.py 


You can change the corresponding parameters in main function prdict.py to choose to use the model to predict for S/T or Y sites

Train with your own data
=====
If you want to train your own network, your input file are four csv files (positive reference sequences and corresponding variant sequences; negative reference sequences and corresponding variant sequences), while separately contains 4 columns: label, protein Name, position, sequence



You can change the corresponding parameters in main function train.py to choose to use the model to predict for S/T or Y sites

Contact
=========
Please feel free to contact us if you need any help: lxlovetf@mail.ustc.edu.cn

