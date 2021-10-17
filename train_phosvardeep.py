import functools
import itertools
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from keras.layers import Dense, Activation, Flatten, Dropout, Reshape
from keras.layers import Conv1D, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.optimizers import Adam, SGD
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import copy

def train_phosvardeep(pos_file_name1, pos_file_name2, nag_file_name1, nag_file_name2, sites):
    '''
    :param pos_file_name1: input of your positive reference sequence file
                            it must be a .csv file and theinput format  is proteinName, postion,sites, shortseq
           pos_file_name2: input of your positive variant sequence file
           pos_file_name1: input of your negative reference sequence file
           pos_file_name2: input of your negative variant sequence file
    :param sites: the sites predict: site = 'S','T' OR 'Y'
    :return:
     a file with the score
    '''

    #data processing
    
    #multi-length
    win1 = 51
    win2 = 33
    win3 = 15
    from methods.dataprocess_train import getMatrixLabel
    #nag-ori
    [X_nag1_1, y_neg] = getMatrixLabel(nag_file_name1, sites, win1)
    [X_nag2_1, _] = getMatrixLabel(nag_file_name1, sites, win2)
    [X_nag3_1, _] = getMatrixLabel(nag_file_name1, sites, win3)
    #nag-var
    [X_nag1_2, _] = getMatrixLabel(nag_file_name2, sites, win1)
    [X_nag2_2, _] = getMatrixLabel(nag_file_name2, sites, win2)
    [X_nag3_2, _] = getMatrixLabel(nag_file_name2, sites, win3)
    #pos-ori
    [X_pos1_1, y_pos] = getMatrixLabel(pos_file_name1, sites, win1)
    [X_pos2_1, _] = getMatrixLabel(pos_file_name1, sites, win2)
    [X_pos3_1, _] = getMatrixLabel(pos_file_name1, sites, win3)
    #pos-var
    [X_pos1_2, _] = getMatrixLabel(pos_file_name2, sites, win1)
    [X_pos2_2, _] = getMatrixLabel(pos_file_name2, sites, win2)
    [X_pos3_2, _] = getMatrixLabel(pos_file_name2, sites, win3)
    


    img_dim1 = X_pos1_1.shape[1:]
    img_dim2 = X_pos2_1.shape[1:]
    img_dim3 = X_pos3_1.shape[1:]

    #phosphorlytion feature extraction by pre-trained model PhosFEN
    from methods.phosnet import PhosFEN

    # load model weight
    if sites == ('S', 'T'):
        model_weight = './models/model_general_S,T.h5'
    if sites == 'Y':
        model_weight = './models/model_general_Y.h5'

    base_model = PhosFEN(img_dim1, img_dim2, img_dim3)
    base_model.load_weights(model_weight)

    FeatureNetwork = Model(inputs=base_model.input, outputs=base_model.get_layer('contact_multi_seq').output)


   #train input
    X_pos1 = FeatureNetwork.predict([X_pos1_1, X_pos2_1, X_pos3_1])
    X_pos2 = FeatureNetwork.predict([X_pos1_2, X_pos2_2, X_pos3_2])
 
    X_neg1 = FeatureNetwork.predict([X_nag1_1, X_nag2_1, X_nag3_1])
    X_neg2 = FeatureNetwork.predict([X_nag1_2, X_nag2_2, X_nag3_2])

    #phospho-variant aware feature extraction by CNN_module
    input11 = Input(shape=X_pos1.shape[1:])
    input22 = Input(shape=X_neg1.shape[1:])


    from methods.model_cnn_pre import CNN_module, prediction_mudule
    snn_ori = CNN_module(input11, 'ori')
    snn_var = CNN_module(input22, 'var')
     
    
    #intergrating by prediction_module
    fc2 = prediction_module(snn_ori, snn_var)
    
    
    model = Model(inputs=[input11, input22], outputs=fc2)

    # Model output
    model.summary()
    # choose optimazation
    if opt_model == 'adam':
        opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    if opt_model == 'sgd':
        opt = SGD(lr=learning_rate, momentum=0.0001, decay=0.00001, nesterov=False)

    # model compile
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

   


    if not os.path.exists(mainfolder):
        os.makedirs(mainfolder)
    plot_model(model, to_file=mainfolder + '/model.png')


    #bootstrapping
    a = int(X_pos1.shape[0] * 0.2)

    val_pos1 = X_pos1[0:a]
    val_neg1 = X_neg1[0:a]
    train_pos1 = X_pos1[a:]
    train_neg1 = X_neg1[a:]
    X_val1 = np.concatenate((val_pos1, val_neg1), axis=0)

    val_pos2 = X_pos2[0:a]
    val_neg2 = X_neg2[0:a]
    train_pos2 = X_pos2[a:]
    train_neg2 = X_neg2[a:]
    X_val2 = np.concatenate((val_pos2, val_neg2), axis=0)


    y_val_pos = y_pos[0:a]
    y_val_neg = y_neg[0:a]
    y_train_pos = y_pos[a:]
    y_train_neg = y_neg[a:]
    y_val = np.concatenate((y_val_pos, y_val_neg), axis=0)

    slength = int(train_pos2.shape[0]*0.8)
    nclass = int(train_neg2.shape[0] / slength)
    y_train_pos = sample(y_train_pos, slength)

    #test_acc = []
    #test_loss = []
    train_loss = []
    train_acc = []
    vali_loss = []
    vali_acc = []
    #test_auc = []

    plt.title('model loss and acc')
    plt.ylabel('loss-acc')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss', 'train_acc', 'val_acc'], loc='upper right')

    for t in range(nclass):

        train_neg_ss1 = train_neg1[(slength * t):(slength * t + slength)]
        train_neg_ss2 = train_neg2[(slength * t):(slength * t + slength)]
        y_train_neg_ss = y_train_neg[(slength * t):(slength * t + slength)]

        random.seed(t)
        train_pos11 = sample(train_pos1, slength)
        random.seed(t)
        train_pos22 = sample(train_pos2, slength)

        X_train1 = np.concatenate((train_pos11, train_neg_ss1), axis=0)
        X_train2 = np.concatenate((train_pos22, train_neg_ss2), axis=0)
        y_train = np.concatenate((y_train_pos, y_train_neg_ss), axis=0)


        for i in range(nb_epoch):
            hist = model.fit([X_train1, X_train2], y_train, batch_size=nb_batch_size,
                             validation_data=([X_val1, X_val2], y_val),
                             epochs=i + 1, initial_epoch=i, shuffle=True, verbose=1)
            train_loss.append(hist.history['loss'])
            train_acc.append(hist.history['acc'])
            vali_loss.append(hist.history['val_loss'])
            vali_acc.append(hist.history['val_acc'])

            if not os.path.exists(modelfolder):
                os.makedirs(modelfolder)
            model.save_weights(modelfolder + '/model_dense' + str(t*nb_epoch + i + 1) + '.h5', overwrite=True)



    if not os.path.exists(mainfolder):
        os.makedirs(mainfolder)
    fig, ax = plt.subplots()
    x = range(0, nb_epoch*nclass)
    plt.plot(x, train_loss, 'r-', label='train loss ')
    ax.legend(loc='upper right', shadow=True)
    plt.plot(x, vali_loss, 'b-', label='validation loss ')
    ax.legend(loc='upper right', shadow=True)
    plt.title('loss of PhosVarDep {:s} epoch-{:d}'.format(ptm_type, nb_epoch*nclass))
    plt.xlabel('epoch')
    plt.savefig(mainfolder + '/loss of epoch {:d}.png'.format(nb_epoch*nclass))

    fig, ax = plt.subplots()
    plt.plot(x, train_acc, 'r-', label='train acc ')
    ax.legend(loc='lower right', shadow=True)
    plt.plot(x, vali_acc, 'b-', label='validation acc ')
    ax.legend(loc='lower right', shadow=True)
    plt.title('accuracy of PhosVarDep {:s} epoch-{:d}'.format(ptm_type, nb_epoch*nclass))
    plt.xlabel('epoch')
    plt.savefig(mainfolder + '/acc of epoch {:d}'.format(nb_epoch*nclass))
    plt.close()

    print('best validation acc is: {:s}, epoch:{:d}'.format(max(vali_acc), vali_acc.index(max(vali_acc)) + 1))




if __name__ == '__main__':
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # '1' TITAN 1080
    nb_batch_size = 512
    nb_epoch = 30
    learning_rate = 0.00003
    opt_model = 'adam'

    set = '1'
    site = 'S', 'T'
    mainfolder = 'PhosVarDeep/{:s}/set{:s}'.format(site, set)
    modelfolder = mainfolder + '/model'
    figfolder = mainfolder + '/figure'

    train_file_name1 = 'train_oridata{:s}{:s}.csv'.format(site, set)
    train_file_name2 = 'train_vardata{:s}{:s}.csv'.format(site,set)
    pos_file_name1 = 'pos_oridata{:s}{:s}.csv'.format(site,set)
    pos_file_name2 = 'pos_vardata{:s}{:s}.csv'.format(site,set)

    train_phosvardeep(pos_file_name1, pos_file_name2, train_file_name1,train_file_name2, site)





