#This one is for the writeup.
import argparse
import ast
import numpy as np
import os
import json
import math
from allennlp.commands.elmo import ElmoEmbedder
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from keras.layers import Input, Flatten, Dense, Activation,Average
from keras.layers import Concatenate,Dropout,Conv1D,MaxPooling1D,BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint

configs = {
  "small": "elmo_2x1024_128_2048cnn_1xhighway_options.json",
  "medium": "elmo_2x2048_256_2048cnn_1xhighway_options.json",
  "original5b": "elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
  "original": "elmo_2x4096_512_2048cnn_2xhighway_options.json"
}

models = {
  "small": "elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5",
  "medium": "elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5",
  "original5b": "elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5",
  "original": "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
}


def line2elmo(line, elmo, batchsize, maxsents, maxtoks):
    #line: string of tokenized text to be embedded
    #elmo: ElmoEmbedder of desired specifications
    fields = line.split("\t")
    title = fields[5]
    tmp = fields[4]
    tmp = tmp.split(" <splt> ")[:maxsents]  #split character is mutable
    sents = [title]
    sents.extend(tmp)
    # now processes the sents in batches
    outs = []
    for batchnr in range(math.ceil(len(sents)/batchsize)):
        fromidx = batchnr * batchsize
        toidx = (batchnr+1) * batchsize
        actualtoidx = min(len(sents), toidx)
        # print("Batch: from=",fromidx,"toidx=",toidx,"actualtoidx=",actualtoidx)
        sentsbatch = sents[fromidx:actualtoidx]
        sentsbatch = [s.split()[:maxtoks] for s in sentsbatch]
        for s in sentsbatch:
            if len(s) == 0:
                s.append("")  # otherwise a shape (3,0,dims) result
        ret = list(elmo.embed_sentences(sentsbatch))
        # the ret is the original representation of three vectors per word
        # first combine per word through concatenation or average, then average
        ret = [np.average(x, axis=1) for x in ret]
        # print("DEBUG tmpembs=", [l.shape for l in tmpembs])
        ret = [np.average(x, axis=0) for x in ret]
        # print("DEBUG finalreps=", [l.shape for l in finalreps])
        outs.extend(ret)

    outs = [a.tolist() for a in outs]
    return (fields[0], fields[1], outs)

def conv1d(max_len, embed_size):
    '''
    CNN without Batch Normalisation.
    :param max_len: maximum sentence numbers, default=200
    :param embed_size: ELMo embeddings dimension, default=1024
    :return: CNN without BN model
    '''
    filter_sizes = [2, 3, 4, 5, 6]
    num_filters = 128
    drop = 0.5
    inputs = Input(shape=(max_len,embed_size), dtype='float32')

    conv_0 = Conv1D(num_filters, kernel_size=(filter_sizes[0]))(inputs)
    act_0 = Activation('relu')(conv_0)
    conv_1 = Conv1D(num_filters, kernel_size=(filter_sizes[1]))(inputs)
    act_1 = Activation('relu')(conv_1)
    conv_2 = Conv1D(num_filters, kernel_size=(filter_sizes[2]))(inputs)
    act_2 = Activation('relu')(conv_2)
    conv_3 = Conv1D(num_filters, kernel_size=(filter_sizes[3]))(inputs)
    act_3 = Activation('relu')(conv_3)
    conv_4 = Conv1D(num_filters, kernel_size=(filter_sizes[4]))(inputs)
    act_4 = Activation('relu')(conv_4)

    maxpool_0 = MaxPooling1D(pool_size=(max_len - filter_sizes[0]))(act_0)
    maxpool_1 = MaxPooling1D(pool_size=(max_len - filter_sizes[1]))(act_1)
    maxpool_2 = MaxPooling1D(pool_size=(max_len - filter_sizes[2]))(act_2)
    maxpool_3 = MaxPooling1D(pool_size=(max_len - filter_sizes[3]))(act_3)
    maxpool_4 = MaxPooling1D(pool_size=(max_len - filter_sizes[4]))(act_4)

    concatenated_tensor = Concatenate()([maxpool_0, maxpool_1, maxpool_2, maxpool_3, maxpool_4])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=1, activation='sigmoid')(dropout)

    model = Model(inputs=inputs, outputs=output)
    #model = multi_gpu_model(model, gpus=gpus)
    model.summary()
    model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer='adam')
    return model

def conv1d_BN(max_len, embed_size):
    '''
    CNN with Batch Normalisation.
    :param max_len: maximum sentence numbers, default=200
    :param embed_size: ELMo embeddings dimension, default=1024
    :return: CNN with BN model
    '''
    filter_sizes = [2, 3, 4, 5, 6]
    num_filters = 128
    inputs = Input(shape=(max_len,embed_size), dtype='float32')
    conv_0 = Conv1D(num_filters, kernel_size=(filter_sizes[0]))(inputs)
    act_0 = Activation('relu')(conv_0)
    bn_0 = BatchNormalization(momentum=0.7)(act_0)

    conv_1 = Conv1D(num_filters, kernel_size=(filter_sizes[1]))(inputs)
    act_1 = Activation('relu')(conv_1)
    bn_1 = BatchNormalization(momentum=0.7)(act_1)

    conv_2 = Conv1D(num_filters, kernel_size=(filter_sizes[2]))(inputs)
    act_2 = Activation('relu')(conv_2)
    bn_2 = BatchNormalization(momentum=0.7)(act_2)

    conv_3 = Conv1D(num_filters, kernel_size=(filter_sizes[3]))(inputs)
    act_3 = Activation('relu')(conv_3)
    bn_3 = BatchNormalization(momentum=0.7)(act_3)

    conv_4 = Conv1D(num_filters, kernel_size=(filter_sizes[4]))(inputs)
    act_4 = Activation('relu')(conv_4)
    bn_4 = BatchNormalization(momentum=0.7)(act_4)

    maxpool_0 = MaxPooling1D(pool_size=(max_len - filter_sizes[0]))(bn_0)
    maxpool_1 = MaxPooling1D(pool_size=(max_len - filter_sizes[1]))(bn_1)
    maxpool_2 = MaxPooling1D(pool_size=(max_len - filter_sizes[2]))(bn_2)
    maxpool_3 = MaxPooling1D(pool_size=(max_len - filter_sizes[3]))(bn_3)
    maxpool_4 = MaxPooling1D(pool_size=(max_len - filter_sizes[4]))(bn_4)

    concatenated_tensor = Concatenate()([maxpool_0, maxpool_1, maxpool_2, maxpool_3, maxpool_4])
    flatten = Flatten()(concatenated_tensor)
    output = Dense(units=1, activation='sigmoid')(flatten)

    model = Model(inputs=inputs, outputs=output)
    #model = multi_gpu_model(model, gpus=gpus)
    model.summary()
    model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer='adam')
    return model

if __name__ == '__main__':
    default_m = "original"

    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str, help="Training file, should be a train.text.tsv file")
    parser.add_argument("-eb", type=int, default=50, help="Embedding Batch Size (50)")
    parser.add_argument("-tb", type=int, default=500, help="Training Batch Size (500)")
    parser.add_argument("-l", type=int, default=1000, help="Log every (1000)")
    parser.add_argument("--maxtoks", type=int, default=200, help="Maximum number of tokens per sentence to use (200)")
    parser.add_argument("--maxsents", type=int, default=200, help="Maximum number of sentences per article to use (200)")
    parser.add_argument("-m", type=str, default=default_m,
                        help="Model (small, medium, original, original5b ({})".format(default_m))
    parser.add_argument("-g", action='store_true', help="Use the GPU (default: don't)")
    parser.add_argument("--concat", action='store_true', help="Concatenate representations instead of averaging")
    args = parser.parse_args()

    outfile = args.outfile
    infile = args.infile
    batchsize_e = args.eb
    batchsize_t = args.tb
    every = args.l
    use_gpu = args.g
    model = os.path.join("elmo", models[args.m])
    config = os.path.join("elmo", configs[args.m])
    concat = args.concat
    maxtoks = args.maxtoks
    maxsents = args.maxsents

    if use_gpu:
        device = 0
    else:
        device = -1
    
    elmo = ElmoEmbedder(options_file=config, weight_file=model, cuda_device=device)
    l_encoder = LabelEncoder()

    seed = 7
    max_len = 200
    embed_size = 1024

    i = 0
    vscores = []

    train_ratio = 0.8

    model = conv1d_BN(max_len, embed_size)
    with open(infile, "rt", encoding="utf8") as inp:
        lines = inp.readlines()
        np.random.shuffle(lines)
        n = len(lines)
        while i < n - batchsize_t:
            nlines = 0
            batch = []
            X = []
            Y = []
            ids = []
            while nlines < batchsize_t:
                ident, label, embedding = line2elmo(lines[i], elmo, batchsize_e, maxsents, maxtoks)
                padded_seq = sequence.pad_sequences([embedding], maxlen=max_len, dtype='float32')[0]
                X.append(padded_seq)
                Y.append(label)
                ids.append(ids)
                nlines += 1
                i += 1
            X = np.array(X)
            Y = np.array(l_encoder.fit_transform(Y))
            print("current batch is : %s " % str(i // batchsize_t))

            val_split = int(len(Y) * train_ratio)
            X_train = X[:val_split]
            Y_train = Y[:val_split]
            X_val = X[val_split:]
            Y_val = Y[val_split:]

            checkpoints = ModelCheckpoint(filepath='./saved_models/BNCNN_vacc{val_acc:.4f}_f%s_e{epoch:02d}.hdf5' % str(i),
                                        verbose=1,monitor='val_acc', save_best_only=True)
            history = model.fit(X_train,Y_train,batch_size=32,verbose=1, epochs=1,
                    validation_data=[X_val,Y_val], callbacks=[checkpoints])
            # use the last validation accuracy from the 30 epochs
            his_val = history.history['val_acc'][-1]
            vscores.append(his_val)
            print('trained on %s examples' % i)
    print("Final score: %.4f%% (+/- %.4f%%)" % (np.mean(vscores), np.std(vscores)))
