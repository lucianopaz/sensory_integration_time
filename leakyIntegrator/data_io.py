#!/usr/bin/python
#-*- coding: UTF-8 -*-
"""
Module for loading the behavioral dataset

Defines the AllData class that provides an interface to load
the experimental data from a given base path and handle it as a pandas
dataframe.

Author: Luciano Paz
Year: 2017

"""

from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
import pandas as pd
from scipy.io import loadmat
import os, itertools, sys, re, scipy.integrate, logging, copy, six


class BaseHandler(object):
    def __init__(self, label, path):
        self.label = label
        self.path = path
    
    def load_data(self):
        raise NotImplementedError('BaseHandler instances cannot load data. Use a derived handler instead.')

class SliderHandler(BaseHandler):
    def __init__(self, label, path, array_name):
        super(SliderHandler, self).__init__(label, path)
        self.array_name = array_name
    
    def load_data(self, filter_first_trials=True, rescale_answer=True,
                  discard_empty_time=True, **kwargs):
        temp = loadmat(self.path)[self.array_name]
        subject    = temp['Subject'][0][0].flatten().astype(int)
        session    = temp['Session'][0][0].flatten().astype(int)
        trial      = temp['Trial'][0][0].flatten().astype(int)
        duration1  = np.round(temp['Duration1'][0][0].flatten(), 3)
        intensity1 = np.round(np.round(temp['Sigma1'][0][0].flatten(), 3) * 3.2, 3)
        duration2  = np.nan * np.ones_like(duration1)
        intensity2 = np.nan * np.ones_like(intensity1)
        correct    = temp['Correct'][0][0].flatten().astype(int)
        answer     = temp['Answer'][0][0].flatten().astype(float)
        emptytime  = 1-temp['Type'][0][0].flatten().astype(int)
        task       = temp['Task'][0][0].flatten().astype(int)
        df = pd.DataFrame(data = list(zip(subject, session, trial, duration1,
                                     intensity1, intensity2, duration2, correct,
                                     answer, emptytime, task)),
                         columns=['subject', 'session', 'trial', 'duration1',
                                  'intensity1', 'intensity2', 'duration2', 'correct',
                                  'answer', 'emptytime', 'task'])
        if filter_first_trials:
            df = df.query('trial>100')
        if discard_empty_time:
            df = df.query('emptytime==0').dropna(axis=0, subset=['emptytime'],
                         inplace=False).reset_index(drop=True)
        df = df.query('answer>-(1/9) and answer<(1+1/9)').dropna(axis=0,
                subset=['emptytime'], inplace=False).reset_index(drop=True)
        if discard_empty_time:
            df = df.drop(['emptytime'], axis=1)
        if rescale_answer:
            df.answer = df.answer*9+1
        return df

class HumanDiscriminationHandler(BaseHandler):
    def __init__(self, label, path, array_name):
        super(HumanDiscriminationHandler, self).__init__(label, path)
        self.array_name = array_name
    
    def load_data(self, filter_first_trials=False, **kwargs):
        temp = loadmat(self.path)[self.array_name]
        subject    = temp[:,0].astype(int)
        session    = temp[:,1].astype(int)
        trial      = temp[:,2].astype(int)
        intensity1 = np.round(np.round(temp[:,3], 3) * 3.2, 3)
        duration1  = np.round(temp[:,4], 3)
        intensity2 = np.round(np.round(temp[:,7], 3) * 3.2, 3)
        duration2  = np.round(temp[:,8], 3)
        answer     = temp[:,13].astype(float)
        correct    = temp[:,12].astype(int)
        emptytime  = np.nan * np.ones_like(answer)
        task       = temp[:,11].astype(int)
        df = pd.DataFrame(data = list(zip(subject, session, trial, duration1,
                                     intensity1, intensity2, duration2, correct,
                                     answer, task)),
                         columns=['subject', 'session', 'trial', 'duration1',
                                  'intensity1', 'intensity2', 'duration2', 'correct',
                                  'answer', 'task'])
        if filter_first_trials:
             df = df.query('trial>100')
        df = df.dropna(how='any').reset_index(drop=True)
        return df

class RatDiscriminationHandler(BaseHandler):
    def __init__(self, label, path, array_name):
        super(RatDiscriminationHandler, self).__init__(label, path)
        self.array_name = array_name
    
    def load_data(self, filter_first_trials=False, **kwargs):
        temp = loadmat(self.path)[self.array_name]
        subject    = temp[:,0].astype(int)
        session    = np.ones_like(subject)
        trial      = temp[:,1].astype(int)
        intensity1 = np.round(np.round(temp[:,2])*0.8)
        duration1  = np.round(temp[:,3])
        intensity2 = np.round(np.round(temp[:,5])*0.8)
        duration2  = np.round(temp[:,6])
        answer     = temp[:,7].astype(float)
        task       = temp[:,8].astype(int)  # 0 intensity, 1 duration
        correct    = np.where(task==0, (intensity1>intensity2).astype(np.int),
                                       (duration1>duration2).astype(np.int))
        zero_pair_int = np.logical_and(task==0, intensity1==intensity2)
        correct[zero_pair_int] = np.random.randint(2, size=np.sum(zero_pair_int.astype(int)))
        zero_pair_dur = np.logical_and(task==1, duration1==duration2)
        correct[zero_pair_dur] = np.random.randint(2, size=np.sum(zero_pair_dur.astype(int)))
        df = pd.DataFrame(data = list(zip(subject, session, trial, duration1,
                                     intensity1, intensity2, duration2, correct,
                                     answer, task)),
                         columns=['subject', 'session', 'trial', 'duration1',
                                  'intensity1', 'intensity2', 'duration2', 'correct',
                                  'answer', 'task'])
        if filter_first_trials:
             df = df.query('trial>100')
        df = df.dropna(how='any')\
             .groupby(['duration1','duration2','intensity1','intensity2'])\
             .filter(lambda x: len(x)>10)\
             .reset_index(drop=True)
        return df

class AllData(object):
    def __init__(self, basepath=None,
                datahandlers=None, **kwargs):
        if basepath is None:
            basepath = os.getcwd()
        self.basepath = basepath
        if datahandlers is None:
            self.datahandlers = [SliderHandler(label='slider',
                                               path=os.path.join(self.basepath, 'Slider.mat'),
                                               array_name='TimeEstimationBoth'),
                                 HumanDiscriminationHandler(label='human_dur_disc',
                                                            path=os.path.join(self.basepath, 'DurationDiscrimination.mat'),
                                                            array_name='a'),
                                 HumanDiscriminationHandler(label='human_int_disc',
                                                            path=os.path.join(self.basepath, 'IntensityDiscrimination.mat'),
                                                            array_name='c'),
                                 RatDiscriminationHandler(label='rat_disc',
                                                          path=os.path.join(self.basepath, 'RatMatrix.mat'),
                                                          array_name='c')
                                ]
        else:
            self.datahandlers = datahandlers
        self.load_data(**kwargs)
    
    def load_data(self, **kwargs):
        temp = []
        for handler in self.datahandlers:
            data = handler.load_data(**kwargs)
            data = pd.concat([data, pd.DataFrame(data=[(handler.label,)]*len(data),
                                                 columns=['experiment'])], axis=1)
            temp.append(data)
        self.data = pd.concat(temp)
        self.data = self.data.assign(NSD=np.round((self.data['intensity2']-self.data['intensity1'])/(self.data['intensity2']+self.data['intensity1']), 3))
        self.data = self.data.assign(NTD=np.round((self.data['duration2']-self.data['duration1'])/(self.data['duration2']+self.data['duration1']), 3))
        return self.data
    
    def query(self, query, experiment=None, subject_type=None, **kwargs):
        inplace = kwargs.pop('inplace', False)
        if subject_type is not None:
            subject_type = str(subject_type).lower()
        if experiment is not None:
            experiment = str(experiment).lower()
            if experiment=='slider':
                query = "experiment=='slider'"
            elif experiment=='discrimination':
                if subject_type=='rat':
                    query = "experiment=='rat_disc'"
                elif subject_type in ['human', None]:
                    query = "experiment=='human_dur_disc' or "\
                            "experiment=='human_int_disc'"
                else:
                    raise ValueError('Unknown subject_type = {0}'.\
                                        format(subject_type))
            else:
                raise ValueError('Unknown experiment = {0}'.\
                                    format(experiment))
        return self.data.query(query, inplace=False, **kwargs)

def flag_invalid_performance(dataframe, query, threshold):
    temp = dataframe.query(query)
    return (temp.answer==temp.correct).mean()<threshold

def drop_subjects_by_performance(dataframe, performance_filters):
    out = dataframe
    for query, threshold in performance_filters:
        temp = dataframe.groupby('subject').\
                   apply(flag_invalid_performance, query, threshold)
        discarded_subjects = temp[temp].index
        out = out.drop(out[out.subject.isin(discarded_subjects)].index).reset_index(drop=True)
    return out

def get_humans_with_all_experiments(alldata, performance_filters=None):
    a = alldata.data.groupby(['subject']).experiment.nunique()
    valid_subjects = a.loc[a.values==3].index.values
    out = alldata.data.loc[alldata.data['subject'].isin(valid_subjects)].reset_index()
    if performance_filters is not None:
        out = drop_subjects_by_performance(out, performance_filters)
    return out

def test():
    basepath = '/home/lpaz/Dropbox/Luciano/Duration/leaky/raw_data'
    alldata = AllData(basepath)
    alldata.load_data()
    print(alldata.data)

if __name__=="__main__":
    test()
