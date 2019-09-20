from __future__ import division, print_function, absolute_import, unicode_literals

from six import iteritems
import numpy as np
import pandas as pd
from leakyIntegrator import fits_module as fm
from leakyIntegrator.fits_module import Fitter
from leakyIntegrator import data_io as io
from leakyIntegrator.model import Leaky, Stimulator, prob2AFC
from matplotlib import pyplot as plt
import os, re, pprint
from tqdm import tqdm

humans_meta = {'subjects': [1, 2, 3, 4, 5, 6, 8, 10, 11, 15, 16, 17, 18, 20],
              'method_shorthand': 'slider_disc',
              'merit_criteria': 'nll',
              'optimizer': 'cma',
              'alpha_cond': ['single_alpha', '2alpha'],
              'suffix': ''
             }
rats_meta = {'subjects': [67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
              'method_shorthand': 'disc',
              'merit_criteria': 'nll',
              'optimizer': 'cma',
              'alpha_cond': '2alpha',
              'suffix': ''
             }

base_path = os.path.join(os.getcwd(), 'fits/')
def get_files(subject, method_shorthand, merit_criteria, optimizer, alpha_cond, suffix, base_path=base_path):
    alpha = '' if alpha_cond=='single_alpha' else '_2alpha'
    pattern = 'fit_subject_\[{subs}\]_session_\[[1-9 ]+\]_{meth}{alpha}_{mer}_{opt}{suff}.pkl'.\
            format(subs=subject, meth=method_shorthand, alpha=alpha, mer=merit_criteria, opt=optimizer, suff=suffix)
    return [os.path.join(base_path, f) for f in os.listdir(base_path) if re.match(pattern, f)]

human_single_alpha = []
human_2alpha = []
for subj in humans_meta['subjects']:
    kwargs = {k:v for k, v in iteritems(humans_meta) if k not in ['subjects', 'alpha_cond']}
    files = get_files(subject=subj, alpha_cond='single_alpha', **kwargs)
    human_single_alpha.append(files[0])
    files = get_files(subject=subj, alpha_cond='2alpha', **kwargs)
    human_2alpha.append(files[0])

rat = []
for subj in rats_meta['subjects']:
    kwargs = {k:v for k, v in iteritems(rats_meta) if k not in ['subjects']}
    files = get_files(subject=subj, **kwargs)
    rat.append(files[0])

parameters = {'1a': [], '2a': [], 'rats': []}
for f in tqdm(human_single_alpha + human_2alpha + rat):
    key = '1a' if f in human_single_alpha else '2a' if f in human_2alpha else 'rats'
    p = parameters[key]
    fitter = fm.load_Fitter_from_file(f)
    pars = fitter.get_parameters_dict_from_fit_output()
    stopcond = list(fitter._fit_output[-1].keys())[0]
    if stopcond in ['tolfun', 'tolx', 'tolfunhist']:
        relevance = True
    else:
        relevance = False
    if key=='1a':
        alpha = pars['intensity_alpha']
        backm = pars['duration_background_mean']
        try:
            taud = pars['duration_tau']
            taui = pars['intensity_tau']
        except:
            taud = 1./pars['duration_tau_inv']
            taui = 1./pars['intensity_tau_inv']
        p.append([alpha, backm, taud, taui, relevance])
    elif key=='2a':
        alphad = pars['duration_alpha']
        alphai = pars['intensity_alpha']
        try:
            taud = pars['duration_tau']
            taui = pars['intensity_tau']
        except:
            taud = 1./pars['duration_tau_inv']
            taui = 1./pars['intensity_tau_inv']
        p.append([alphad, alphai, taud, taui, relevance])
    elif key=='rats':
        alphad = pars['duration_alpha']
        alphai = pars['intensity_alpha']
        try:
            taud = pars['duration_tau']
            taui = pars['intensity_tau']
            if taud is None:
                taud = np.nan
            if taui is None:
                taui = np.nan
        except:
            taud = 1./pars['duration_tau_inv']
            taui = 1./pars['intensity_tau_inv']
            if taud is None:
                taud = np.nan
            if taui is None:
                taui = np.nan
        alphad = np.nan if alphad is None else alphad
        alphai = np.nan if alphai is None else alphai
        p.append([alphad, alphai, taud, taui, relevance])

single_alpha_pars = np.array(parameters['1a'], dtype=np.float)
_2alpha_pars = np.array(parameters['2a'], dtype=np.float)
rat_pars = np.array(parameters['rats'], dtype=np.float)

print(single_alpha_pars)
print(_2alpha_pars)
print(rat_pars)
print(single_alpha_pars.shape)
print(_2alpha_pars.shape)
print(rat_pars.shape)

print(rat_pars[:, [0, 1]])
print(np.around(rat_pars[:, [2, 3, 4]]))
print(np.around(single_alpha_pars[:, [2, 3, 4]]))
print(np.around(_2alpha_pars[:, [2, 3, 4]]))

#~ single_alpha_pars = np.random.rand(7,4)
#~ _2alpha_pars = np.random.rand(7,4)
#~ rat_pars = np.random.rand(10,4)

plt.figure(figsize=(12, 9))
gs = plt.GridSpec(1, 3, width_ratios=[5, 6, 1], wspace=0.40)
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])

human_single_alpha = np.nanmean(single_alpha_pars[:, 0], axis=0)
human_single_alpha_err = np.nanmean(single_alpha_pars[:, 0], axis=0)
human_alpha_dur = np.nanmean(_2alpha_pars[:, 0], axis=0)
human_alpha_dur_err = np.nanmean(_2alpha_pars[:, 0], axis=0)
human_alpha_int = np.nanmean(_2alpha_pars[:, 1], axis=0)
human_alpha_int_err = np.nanmean(_2alpha_pars[:, 1], axis=0)

alphas = np.concatenate((np.nanmean(single_alpha_pars[:, 0], axis=0)[None],
                         np.nanmean(_2alpha_pars[:, :2], axis=0),
                         np.nanmean(rat_pars[:, :2], axis=0)),
                        axis=0)
alphas_err = np.concatenate((np.nanstd(single_alpha_pars[:, 0], axis=0)[None] / np.sqrt(len(single_alpha_pars)),
                             np.nanstd(_2alpha_pars[:, :2], axis=0) / np.sqrt(len(_2alpha_pars)),
                             np.nanstd(rat_pars[:, :2], axis=0) / np.sqrt(len(rat_pars))),
                            axis=0)
taus   = np.concatenate((np.nanmean(single_alpha_pars[:, 2:4], axis=0),
                         np.nanmean(_2alpha_pars[:, 2:4], axis=0),
                         np.nanmean(rat_pars[:, 2:4], axis=0)),
                        axis=0)
taus_err   = np.concatenate((np.nanstd(single_alpha_pars[:, 2:4], axis=0) / np.sqrt(len(single_alpha_pars)),
                             np.nanstd(_2alpha_pars[:, 2:4], axis=0) / np.sqrt(len(_2alpha_pars)),
                             np.nanstd(rat_pars[:, 2:4], axis=0) / np.sqrt(len(rat_pars))),
                            axis=0)
mub    = np.nanmean(single_alpha_pars[:, 1], axis=0)
mub_err    = np.nanstd(single_alpha_pars[:, 1], axis=0)

ax1.bar(range(5), alphas, yerr=alphas_err,
        color='r', ecolor='k')
ax1.set_xticks(range(5))
ax1.set_xticklabels([r'$\mathrm{H}_{1\mathrm{A}}$ $\alpha$',
                     r'$\mathrm{H}_{2\mathrm{A}}$ $\alpha_{T}$',
                     r'$\mathrm{H}_{2\mathrm{A}}$ $\alpha_{I}$',
                     r'$\mathrm{R}_{2\mathrm{A}}$ $\alpha_{T}$',
                     r'$\mathrm{R}_{2\mathrm{A}}$ $\alpha_{I}$'],
            rotation=60)
ax1.set_ylabel(r'$\alpha$', fontsize=16)

ax2.bar(range(6), taus, yerr=taus_err,
        color='r', ecolor='k')
ax2.set_xticks(range(6))
ax2.set_xticklabels([r'$\mathrm{H}_{1\mathrm{A}}$ $\tau_{T}$',
                     r'$\mathrm{H}_{1\mathrm{A}}$ $\tau_{I}$',
                     r'$\mathrm{H}_{2\mathrm{A}}$ $\tau_{T}$',
                     r'$\mathrm{H}_{2\mathrm{A}}$ $\tau_{I}$',
                     r'$\mathrm{R}_{2\mathrm{A}}$ $\tau_{T}$',
                     r'$\mathrm{R}_{2\mathrm{A}}$ $\tau_{I}$'],
            rotation=60)
ax2.set_ylabel(r'$\tau$ [ms]', fontsize=16)

ax3.bar(range(1), mub, yerr=mub_err,
        color='r', ecolor='k')
ax3.set_xticks(range(1))
ax3.set_xticklabels([r'$\mathrm{H}_{1\mathrm{A}}$ $\mu_{b}$'],
            rotation=60)
ax3.set_ylabel(r'$\mu_{b}$', fontsize=16)

plt.savefig('figs/parameters.png', bbox_inches='tight')
plt.show(True)
