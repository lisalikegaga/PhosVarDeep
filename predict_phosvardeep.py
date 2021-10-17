import os
import numpy as np
from keras.models import Model

def predict_for_phosvardeep(test_file_name1, test_file_name2, sites):
    '''
        :param test_file_name1: input of your positive reference sequence file
                                it must be a .csv file and theinput format  is proteinName, postion,sites, shortseq
               test_file_name2: input of your positive variant sequence file
        :param sites: the sites predict: site = 'S','T' OR 'Y'
        :return:
         a file with the score
        '''
    #data processing
    win1 = 51
    win2 = 33
    win3 = 15
    from methods.dataprocess import getMatrixInput
    #test-ori
    [X_test1_1,y_test] = getMatrixLabel(test_file_name1, sites, win1)
    [X_test2_1,_] = getMatrixLabel(test_file_name1, sites, win2)
    [X_test3_1,_]  = getMatrixLabel(test_file_name1, sites, win3)
    #test-var
    [X_test1_2, y_test] = getMatrixLabel(test_file_name2, sites, win1)
    [X_test2_2, _] = getMatrixLabel(test_file_name2, sites, win2)
    [X_test3_2, _] = getMatrixLabel(test_file_name2, sites, win3)

    img_dim1 = X_test1_1.shape[1:]
    img_dim2 = X_test2_1.shape[1:]
    img_dim3 = X_test3_1.shape[1:]

    # phosphorlytion feature extraction by pre-trained model PhosFEN
    from methods.phosnet import PhosFEN

    # load model weight
    if sites == ('S', 'T'):
        model_weight = './models/model_general_S,T.h5'
    if sites == 'Y':
        model_weight = './models/model_general_Y.h5'

    base_model = PhosFEN(img_dim1, img_dim2, img_dim3)
    base_model.load_weights(model_weight)

    FeatureNetwork = Model(inputs=base_model.input, outputs=base_model.get_layer('contact_multi_seq').output)

    # test input
    X_test1 = FeatureNetwork.predict([X_test1_1, X_test2_1, X_test3_1])
    X_test2 = FeatureNetwork.predict([X_test1_2, X_test2_2, X_test3_2])

    # phospho-variant aware feature extraction by CNN_module
    input11 = Input(shape=X_test1.shape[1:])
    input22 = Input(shape=X_test2.shape[1:])

    from methods.model_cnn_pre import CNN_module, prediction_mudule
    snn_ori = CNN_module(input11, 'ori')
    snn_var = CNN_module(input22, 'var')

    # intergrating by prediction_module
    fc2 = prediction_module(snn_ori, snn_var)

    model = Model(inputs=[input11, input22], outputs=fc2)


    #load model weight
    if site == ('S','T'):
            model_weight = './models/model_phosvar_S,T.h5'
    if site == 'Y':
            model_weight = './models/model_phosvar_Y.h5'

    model.load_weights(model_weight)
    predictions_t = model.predict([X_test1, X_test2])
    np.savetxt(prefolder + '/phosvardeep scores.txt',predictions_t[:, 1],fmt='%.5f')

if __name__ == '__main__':
    test_file_name1 = 'testdata_ori.csv'  #reference sequence
    test_file_name2 = 'testdata_var.csv'  #varaint sequence
    sites = 'S','T'  # or 'Y'
    predict_for_phosvardeep(test_file_name1, test_file_name2, sites)




