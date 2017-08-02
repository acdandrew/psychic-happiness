import keras
import sys
sys.path.append('kernel')
import numpy as np
import os, sys
import shutil
path = "../data/statefarm/"
# Import our class, and instantiate
import vgg16; reload(vgg16)
from vgg16 import Vgg16

def init_model():
    m = Vgg16()
    batch_size = 48
    batches = m.get_batches(path+'train/train/', batch_size=batch_size)
    val_batches = m.get_batches(path+'train/val', batch_size=batch_size)
    m.finetune(batches)
    m.fit(batches, val_batches, nb_epoch=5)
    m.model.save_weights("lastconvo.h5")
    print("We reached end")
    return (m,batches,val_batches)


def predict_full(weights_name):
    m = Vgg16()
    batches = m.get_batches(path+'train/train/')
    m.finetune(batches)
    m.model.load_weights(weights_name)
    names = os.listdir(path + '/test/unknown')
    test_imgs = m.get_batches(path +'test/', shuffle=False)
    
    results = open('state_farm_results.csv', 'w')
    results.write('img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n')
    index = 0
    for imgs,cls in test_imgs:
        pred, cname = m.predict_full(imgs)
        for i in pred:
            results.write(str(names[index]))
            o = np.clip(i,.05,.95)
            for x in np.nditer(o):
                results.write(',' + str(x))
            results.write('\n')
            index = index + 1
    results.close() 
        

def predict_current(m, result_name='state_farm_results.csv'):
    names = os.listdir(path + '/test/unknown')
    test_imgs = m.get_batches(path +'test/', shuffle=False,batch_size = 48)
    
    results = open(result_name, 'w')
    results.write('img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n')
    index = 0
    try:
        for imgs,cls in test_imgs:
            pred, cname = m.predict_full(imgs)
            for i in pred:
                results.write(str(names[index]))
                o = np.clip(i,.05,.95)
                for x in np.nditer(o):
                    results.write(',' + str(x))
                results.write('\n')
                index = index + 1
    finally:
        results.close()

def train_and_predict():
    m,b,c= init_model()
    predict_current(m)
