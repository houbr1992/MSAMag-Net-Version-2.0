# -*- coding: utf-8 -*-
import os, pickle, math, random
import numpy as np
from tqdm import tqdm
from itertools import chain

class dataparams():

    def __init__(self, indl, ratl, magl, num0=32, trin=3, eps=0., train=False):

        super(dataparams, self).__init__()
        self.indl = indl
        self.ratl = ratl
        self.magl = magl
        self.num0 = num0
        self.trin = trin
        self.eps = eps
        self.train = train

    def opfile(self, x):

        # with open(x.decode('utf-8'), 'rb') as f:
        #     dic = pickle.load(f)

        with open(x, 'rb') as f:
            dic = pickle.load(f)

        Qname, Depth, Mag = dic['Qname'], dic['Depth'], dic['Mag']
        Qlat, Qlon, names = dic['Qlat'], dic['Qlon'], dic['Stations']
        lat, lon, R, pt = dic['lat'], dic['lon'], dic['R'], dic['pt']
        Pa, Pv, Pd = dic['Pa'], dic['Pv'], dic['Pd']
        IV2, CAV, tc = dic['IV2'], dic['CAV'], dic['tc']

        # location info [ lat, lon, R, pt ]
        loc_info = np.array([pt, lat, lon, R])
        loc_info = np.transpose(loc_info, (1, 0))

        # Params info
        Params_info = np.array([Pa, Pv, Pd, IV2, CAV, tc])
        Params_info = np.transpose(Params_info, (1, 0, 3, 2))
        
        return [[Qname, Depth, Mag, Qlat, Qlon ], names, loc_info, Params_info]

    def mulparams(self, Q_info, names, S_info, Params, t0):

        """
        Input,
        Out:
        ParaInfo, array -> [ Pa, Pv, Pd, IV2, CAV, tc ] -> (len_sta, 6, component, len_t)
        t_index, shape
        S_info,
        Mag, Qlat, Q_lon, Depth, Ori_Time
        """

        s0, s1, s2, s3 = Params.shape
        # Pwave loc based Ori_Time
        pt = S_info[:, 0][:, np.newaxis]
        # Stations trigger t agter the 1st Trigger t sec
        t_mat = ((np.min(pt)) + t0 - pt) * 10
        # Different Stations trigger stages after 1st Trigger t time
        t_ind = np.less_equal(np.ones((s0, 1)) * np.arange(1, 301), t_mat) + 0.
        # Station info after 1 st trigger t sec
        S_info0 = np.max(t_ind, axis=1, keepdims=True) * np.concatenate([S_info, t_mat/10], axis=1)
        # Params  Pa, Pv, Pd, IV2, CAV, tc
        Params = Params[:, :, :, 5:] * t_ind[:, np.newaxis, np.newaxis, :]
        # Station names
        names = [names[i] for i in range(len(names)) if t_mat[i] > 0]

        if s0 < self.num0:
            s = self.num0 - s0
            Params = np.pad(Params, ((0, s), (0, 0), (0, 0), (0, 0)), constant_values=(0., 0.))
            t_ind = np.pad(t_ind, ((0, s), (0, 0)), constant_values=(0., 0.))
            S_info0 = np.pad(S_info0, ((0, s), (0, 0)), constant_values=(0., 0.))
        
        return Params, t_ind, S_info0, Q_info[2], Q_info[3], Q_info[4], Q_info[1], Q_info[0]

    def obtind(self, x):
        ind = len(self.indl) - np.sum(np.greater_equal(x, self.magl) + 0 )
        return int(ind)

    def trigger_num(self, x):
        bools = False
        if len(x) > 0:
            x = np.max(x, axis=1)
            x = np.sum(x)
            if x >= self.trin:
                bools = True
        return bools

    def zipfunc(self, x):
        """
        zip function
        Input:
        names, Params, S_info
        Output:
        Zip information
        """
        x = [list(Info) for Info in x]
        x0, x1, x2 = x
        x = list(zip(x0, x1, x2))
        return x

    def unzipfunc(self, x):
        """
        unzip function
        Input:
        zip infomation
        output:
        names, Params, S_info
        """
        x = sorted(x, key=lambda x: (float(x[2][0])))
        x = [t for t in zip(*x)]
        x0 = list(x[0])
        x1, x2 = [np.array(i).astype('float32') for i in x[1:3]]
        return [x0, x1, x2]

    def __call__(self, files, t0):

        eve_lis = []
        for f in files:

            Q_info, names, S_info, Params = self.opfile(f)
            Mag = Q_info[2]

            if self.train:
                MagSeg = self.obtind(Mag)
                ind = self.indl[MagSeg]
                names = names[:ind]
                S_info = S_info[:ind, :]
                Params = Params[:ind, :, :, :]
                zipinfo = self.zipfunc([names, Params, S_info])

                # 先进行重采样
                for r in range(self.ratl[MagSeg]):

                    if len(names) >= self.num0:
                        zipinfo0 = random.sample(zipinfo, self.num0)
                    else:
                        zipinfo0 = zipinfo
                        random.shuffle(zipinfo0)

                    names0, Params0, S_info0 = self.unzipfunc(zipinfo0)
                    sin_eve = [Q_info, names0, S_info0, Params0]
                    eve_lis.append(sin_eve)
            else:
                sin_eve = [Q_info, names[:self.num0], S_info[:self.num0, :], Params[:self.num0, :, :, :]]
                eve_lis.append(sin_eve)
        # random event list
        random.shuffle(eve_lis)
        # earth_inp = [ [], [], [], [], [], [], [], [], []]
        if len(eve_lis) > 0:
            # 构建数据流
            earth_inp = [[] for i in range(8)]
            earth_ = [self.mulparams(x[0], x[1], x[2], x[3], t0) for x in eve_lis]
            for x in earth_:
                if self.trigger_num(x[1]):
                    earth_inp = [earth_inp[i] + [x[i]] for i in range(8)]

            if earth_inp == []:
                earth_inp = [[] for i in range(8)]
        else:

            earth_inp = [[] for i in range(8)]

        return earth_inp
