# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 15:47:06 2022

@author: hbrhu
"""

import os, pickle, random
import tensorflow as tf


def tvtkque(file, dir0='./Params_3Hz', name='_Params_Type1'):
    """
    TraLis, Mag 0, 1, 2, 3, 4, 5, 6, 7;
    ValLis, Mag 0, 1, 2, 3, 4, 5, 6, 7;
    TesLis, Mag 0, 1, 2, 3, 4, 5, 6, 7;
    KumLis, Mag 0, 1, 2, 3, 4, 5, 6, 7;
    """

    with open(file, 'rb') as f:
        dic = pickle.load(f)
        TraLis = [dic[f'Mag{i}']['Tra'] for i in range(len(dic))]
        ValLis = [dic[f'Mag{i}']['Val'] for i in range(len(dic))]
        TesLis = [dic[f'Mag{i}']['Tes'] for i in range(len(dic))]
        KumLis = [dic[f'Mag{i}']['Kum'] for i in range(len(dic))]

    Lis0 = []
    for SLis in TraLis:
        Lis1 = []
        for f in SLis:
            f = str(f)
            Lis1.append(os.sep.join([dir0, f[:4], f + name]))
        Lis0.append(Lis1)
    TraLis = Lis0

    Lis0 = []
    for SLis in ValLis:
        Lis1 = []
        for f in SLis:
            f = str(f)
            Lis1.append(os.sep.join([dir0, f[:4], f + name]))
        Lis0.append(Lis1)
    ValLis = Lis0

    Lis0 = []
    for SLis in TesLis:
        Lis1 = []
        for f in SLis:
            f = str(f)
            Lis1.append(os.sep.join([dir0, f[:4], f + name]))
        Lis0.append(Lis1)
    TesLis = Lis0

    Lis0 = []
    for SLis in KumLis:
        Lis1 = []
        for f in SLis:
            f = str(f)
            Lis1.append(os.sep.join([dir0, f[:4], f + name]))
        Lis0.append(Lis1)
    KumLis = Lis0

    return TraLis, ValLis, TesLis, KumLis


def Dataset(Que, ratio, shuf, btc_sz):
    """
    Dataset Queue 
    """
    # data set
    dataset = tf.data.TFRecordDataset(Que)
    # data repeat
    dataset = dataset.repeat(ratio)
    # dataset shuffle
    dataset = dataset.shuffle(shuf)
    # dataset map
    dataset = dataset.map(decode_fn)
    dataset = dataset.batch(btc_sz)
    return dataset


def DatasetNShuf(Que, ratio, shuf, btc_sz):
    """
    Dataset Queue 
    """
    # data set
    dataset = tf.data.TFRecordDataset(Que)
    # data repeat
    dataset = dataset.repeat(ratio)
    # dataset shuffle
    # dataset = dataset.shuffle(shuf)
    # dataset map
    dataset = dataset.map(decode_fn)
    dataset = dataset.batch(btc_sz)
    return dataset



def tvtkdata(dir_in='./MagTFRecs', btc_sz=1024, shuf=512, rat=1, num=[7, 7, 7]):

    num2, num3, num4 = num[0], num[1], num[2]

    # training
    dir0_0 = f'{dir_in}/train/Mag0123'
    dir0_1 = f'{dir_in}/train/Mag4'
    dir0_2 = f'{dir_in}/train/Mag5'
    dir0_3 = f'{dir_in}/train/Mag6'
    dir0_4 = f'{dir_in}/train/Mag7'

    # Mag 0, 1, 2, 3, 4
    fs0_0 = [ os.sep.join([dir0_0, f]) for f in os.listdir(dir0_0)]
    fs0_1 = [ os.sep.join([dir0_1, f]) for f in os.listdir(dir0_1)]
    # Mag 5
    fs0_2 = [ os.sep.join([dir0_2, f]) for f in os.listdir(dir0_2)]
    fs0_2 = random.sample(fs0_2, num2)
    # Mag 6
    fs0_3 = [ os.sep.join([dir0_3, f]) for f in os.listdir(dir0_3)]
    fs0_3 = random.sample(fs0_3, num3)
    # Mag 7
    fs0_4 = [ os.sep.join([dir0_4, f]) for f in os.listdir(dir0_4)]
    fs0_4 = random.sample(fs0_4, num4)
    fs0 = fs0_0 + fs0_1 + fs0_2 + fs0_3 + fs0_4
    random.shuffle(fs0)

    # Valid Data
    dir1_0 = f'{dir_in}/valid/Mag0123'
    dir1_1 = f'{dir_in}/valid/Mag4'
    dir1_2 = f'{dir_in}/valid/Mag5'
    dir1_3 = f'{dir_in}/valid/Mag6'
    dir1_4 = f'{dir_in}/valid/Mag7'   
    num2, num3, num4 = 1, 1, 1
    # Mag 0, 1, 2, 3, 4
    fs1_0 = [ os.sep.join([dir1_0, f]) for f in os.listdir(dir1_0)]
    fs1_1 = [ os.sep.join([dir1_1, f]) for f in os.listdir(dir1_1)]
    # Mag 5
    fs1_2 = [ os.sep.join([dir1_2, f]) for f in os.listdir(dir1_2)]
    fs1_2 = random.sample(fs1_2, num2)
    # Mag 6
    fs1_3 = [ os.sep.join([dir1_3, f]) for f in os.listdir(dir1_3)]
    fs1_3 = random.sample(fs1_3, num3)
    # Mag 7
    fs1_4 = [ os.sep.join([dir1_4, f]) for f in os.listdir(dir1_4)]
    fs1_4 = random.sample(fs1_4, num4)
    fs1 = fs1_0 + fs1_1 + fs1_2 + fs1_3 + fs1_4
    random.shuffle(fs1)    

    # Test Data
    dir2 = f'{dir_in}/test'
    fs2 = [ os.sep.join([dir2, f]) for f in os.listdir(dir2)]

    # Kum Data
    dir3 = f'{dir_in}/kum'
    fs3 = [ os.sep.join([dir3, f]) for f in os.listdir(dir3)]

    tra = Dataset(fs0, rat, shuf, btc_sz)
    val = Dataset(fs1, rat, shuf, btc_sz)
    tes = Dataset(fs2, rat, shuf, btc_sz)
    kum = Dataset(fs3, rat, shuf, btc_sz)

    return tra, val, tes, kum


def tvtkdatatim(dir_in='./MagTFRecs', btc_sz=1024, shuf=512, rat=1, num=[2, 2, 2]):

    num2, num3, num4 = num[0], num[1], num[2]

    # training
    fs0 = []
    for t in range(1,31):
        dir0t0 = f'{dir_in}/train/{t}/Mag0123'
        dir0t1 = f'{dir_in}/train/{t}/Mag4'
        dir0t2 = f'{dir_in}/train/{t}/Mag5'
        dir0t3 = f'{dir_in}/train/{t}/Mag6'
        dir0t4 = f'{dir_in}/train/{t}/Mag7'
        
        # Mag 0, 1, 2, 3, 4
        fst0 = [ os.sep.join([dir0t0, f]) for f in os.listdir(dir0t0)]
        fst1 = [ os.sep.join([dir0t1, f]) for f in os.listdir(dir0t1)]
        
        # Mag 5
        fst2 = [ os.sep.join([dir0t2, f]) for f in os.listdir(dir0t2)]
        fst2 = random.sample(fst2, num2)
        # Mag 6
        fst3 = [ os.sep.join([dir0t3, f]) for f in os.listdir(dir0t3)]
        fst3 = random.sample(fst3, num3)
        # Mag 7
        fst4 = [ os.sep.join([dir0t4, f]) for f in os.listdir(dir0t4)]
        fst4 = random.sample(fst4, num4)
        fs0l = fst0 + fst1 + fst2 + fst3 + fst4
        random.shuffle(fs0)
        fs0 += fs0l
        
    # Valid Data
    dir1_0 = f'{dir_in}/valid/Mag0123'
    dir1_1 = f'{dir_in}/valid/Mag4'
    dir1_2 = f'{dir_in}/valid/Mag5'
    dir1_3 = f'{dir_in}/valid/Mag6'
    dir1_4 = f'{dir_in}/valid/Mag7'   
    num2, num3, num4 = 1, 1, 1
    # Mag 0, 1, 2, 3, 4
    fs1_0 = [ os.sep.join([dir1_0, f]) for f in os.listdir(dir1_0)]
    fs1_1 = [ os.sep.join([dir1_1, f]) for f in os.listdir(dir1_1)]
    # Mag 5
    fs1_2 = [ os.sep.join([dir1_2, f]) for f in os.listdir(dir1_2)]
    fs1_2 = random.sample(fs1_2, num2)
    # Mag 6
    fs1_3 = [ os.sep.join([dir1_3, f]) for f in os.listdir(dir1_3)]
    fs1_3 = random.sample(fs1_3, num3)
    # Mag 7
    fs1_4 = [ os.sep.join([dir1_4, f]) for f in os.listdir(dir1_4)]
    fs1_4 = random.sample(fs1_4, num4)
    fs1 = fs1_0 + fs1_1 + fs1_2 + fs1_3 + fs1_4
    random.shuffle(fs1)    

    # Test Data
    dir2 = f'{dir_in}/test'
    fs2 = [ os.sep.join([dir2, f]) for f in os.listdir(dir2)]

    # Kum Data
    dir3 = f'{dir_in}/kum'
    fs3 = [ os.sep.join([dir3, f]) for f in os.listdir(dir3)]

    tra = DatasetNShuf(fs0, rat, shuf, btc_sz)
    val = DatasetNShuf(fs1, rat, shuf, btc_sz)
    tes = DatasetNShuf(fs2, rat, shuf, btc_sz)
    kum = DatasetNShuf(fs3, rat, shuf, btc_sz)

    return tra, val, tes, kum
