#!/usr/bin/env python
# coding: utf-8

#


import os
import glob
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, ttest_rel, ttest_1samp

from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

import pandas as pd
import seaborn as sns

import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker
from nilearn.plotting import plot_glass_brain, plot_stat_map, view_img, view_img_on_surf

from nltools.data import Brain_Data, Adjacency
from nltools.mask import roi_to_brain, expand_mask
from nltools.stats import fdr, threshold

from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state
from sklearn.manifold import TSNE

import datalad.api as dl
from nibabel.freesurfer.mghformat import load





###############################################
### For reading fsaverage vertex annot file ###
###############################################


from nibabel.freesurfer.io import read_annot


#


### run for each hemisphere

data_dir2 = 'path/to/fsaverage/label'

annotdata = read_annot(os.path.join(data_dir2, 'rh.aparc.annot'), orig_ids=True)
annotdata


#


np.savetxt(os.path.join(data_dir2, 'annotdata.csv'), annotdata[0], fmt='%s', delimiter = ",")

len(annotdata[0])





##############
### IS-RSA ###
##############


### get subject data

data_dir = 'your/data/directory'
sub_data = pd.read_csv(os.path.join(data_dir, 'your_subject_pheno_data.csv'), sep = ',')
sub_list = sub_data["Subjects"]

sub_list = np.array(sub_list)


#


### get subjects' vertexwise thickness data
### run for each hemisphere

data = []

for sub in sub_list:
    loaddata = load(os.path.join(data_dir, 'rh', f'{sub}_rh.thickness.fsaverage.mgh'))
    for_reshape = loaddata.get_fdata()
    for_reshape = for_reshape.reshape(163842, 1*1)
    data.append(for_reshape)

np.shape(data)

###(sub, 163842, 1) expected



#



##############
### IS-RSA ###
##############


roi_data_dir = 'path/to/ROIvertexlabel/files/fsaverage_ROI_vertices/LH'
aparc = ['LH_caudalanteriorcingulate_10511485_1439vs', 'LH_caudalmiddlefrontal_6500_3736vs', 'LH_cuneus_6558940_1630vs', 'LH_entorhinal_660700_1102vs', 'LH_frontalpole_6553700_272vs', 'LH_fusiform_9231540_4714vs', 'LH_inferiorparietal_14433500_7871vs', 'LH_inferiortemporal_7874740_4415vs', 'LH_insula_2146559_5229vs', 'LH_isthmuscingulate_9180300_2531vs', 'LH_lateraloccipital_9182740_6379vs', 'LH_lateralorbitofrontal_3296035_4188vs', 'LH_lingual_9211105_4205vs', 'LH_medialorbitofrontal_4924360_2653vs', 'LH_middletemporal_3302560_4452vs', 'LH_paracentral_3988540_3294vs', 'LH_parahippocampal_3988500_1838vs', 'LH_parsorbitalis_3302420_956vs', 'LH_parspopercularis_9221340_3119vs', 'LH_parstriangularis_1326300_2046vs', 'LH_pericalcarine_3957880_1912vs', 'LH_postcentral_1316060_9519vs', 'LH_posteriorcingulate_14464220_3266vs', 'LH_precentral_14423100_10740vs', 'LH_precuneus_11832480_7308vs', 'LH_rostralanteriorcingulate_9180240_1350vs', 'LH_rostralmiddlefrontal_8204875_7243vs', 'LH_superiorfrontal_10542100_12179vs', 'LH_superiorparietal_9221140_10456vs', 'LH_superiortemporal_14474380_7271vs', 'LH_supramarginal_1351760_8600vs', 'LH_temporalpole_11146310_839vs', 'LH_transversetemporal_13145750_1064vs']
#aparc = ['RH_caudalanteriorcingulate_10511485_1608vs', 'RH_caudalmiddlefrontal_6500_3494vs', 'RH_cuneus_6558940_1638vs', 'RH_entorhinal_660700_902vs', 'RH_frontalpole_6553700_369vs', 'RH_fusiform_9231540_4661vs', 'RH_inferiorparietal_14433500_9676vs', 'RH_inferiortemporal_7874740_4198vs', 'RH_insula_2146559_5090vs', 'RH_isthmuscingulate_9180300_2388vs', 'RH_lateraloccipital_9182740_5963vs', 'RH_lateralorbitofrontal_3296035_4354vs', 'RH_lingual_9211105_3894vs', 'RH_medialorbitofrontal_4924360_2801vs', 'RH_middletemporal_3302560_5057vs', 'RH_paracentral_3988540_3831vs', 'RH_parahippocampal_3988500_1742vs', 'RH_parsopercularis_9221340_2472vs', 'RH_parsorbitalis_3302420_946vs', 'RH_parstriangularis_1326300_2380vs', 'RH_pericalcarine_3957880_1823vs', 'RH_postcentral_1316060_9138vs', 'RH_posteriorcingulate_14464220_2994vs', 'RH_precentral_14423100_10705vs', 'RH_precuneus_11832480_7975vs', 'RH_rostralanteriorcingulate_9180240_1051vs', 'RH_rostralmiddlefrontal_8204875_7864vs', 'RH_superiorfrontal_10542100_11878vs', 'RH_superiorparietal_9221140_10222vs', 'RH_superiortemporal_14474380_6868vs', 'RH_supramarginal_1351760_8150vs', 'RH_temporalpole_11146310_817vs', 'RH_transversetemporal_13145750_781vs']

behav_data = pd.read_csv(os.path.join(data_dir, 'your_pheno_data.csv'), sep = ',')
behav = behav_data["SCARED_GAD"]
behav_rank = rankdata(behav)
isrsa_r, isrsa_p = {}, {}

for roi in aparc: 
    
    roi_data = pd.read_csv(os.path.join(roi_data_dir, f'{roi}.csv'), sep = ',')
    roi_data = roi_data['Vertexnumber']
    
    theROI = []
    for sub in range(len(sub_list)):
        sub_allvertex = data[sub]
        sub_theROI = []
        for vertex in roi_data:
            sub_theROI.append(sub_allvertex[vertex])
        theROI.append(sub_theROI)   
    for sub in range(len(theROI)):
        theROI[sub] = np.reshape(theROI[sub], (-1, 1))
    theROI = np.array(theROI)
    
    similarity_matrix = Adjacency(pairwise_distances(theROI[:, :, 0], metric='euclidean'), matrix_type='distance')
    similarity_matrix = similarity_matrix.distance_to_similarity(metric='euclidean')
    
    n_subs = len(sub_list)
    behav_sim_annak = np.zeros((n_subs, n_subs))

    for i in range(n_subs):
        for j in range(n_subs):
            if i < j:
                sim_ij = np.mean([behav_rank[i], behav_rank[j]])/n_subs
                behav_sim_annak[i,j] = sim_ij
                behav_sim_annak[j,i] = sim_ij
            elif i==j:
                behav_sim_annak[i,j] = 1

    behav_sim_annak = Adjacency(behav_sim_annak, matrix_type='similarity')

    isrsa_annak = similarity_matrix.similarity(behav_sim_annak, metric='spearman', n_permute=1, n_jobs=1)['correlation']
    
    if isrsa_annak > 0.15 or isrsa_annak < -0.15:
        print('*** correlation for', roi, ':', isrsa_annak, end='\n')
    else:
        print('correlation for', roi, ':', isrsa_annak, end='\n')
    
    stats = similarity_matrix.similarity(behav_sim_annak, metric='spearman', n_permute=5000, n_jobs=-1)
    isrsa_r[roi] = stats['correlation']
    isrsa_p[roi] = stats['p']


#


isrsa_p


#


### FDR thresholding ###

fdr_thr = fdr(pd.Series(isrsa_p).values)
print(f'FDR Threshold: {fdr_thr}')




