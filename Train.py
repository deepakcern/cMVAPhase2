# cMVAv2 Retraining for CMS Phase II Upgrade
#
# Author: Spandan Mondal
# Email: spandan.mondal@cern.ch
#
# Thanks to Joosep Pata for the cMVAv2 Phase-I training code
# (https://github.com/jpata/CMSCSTagger/blob/master/python/notebooks/learning.ipynb)


#*****************************Imports************************************
from datetime import datetime
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def printlog(text):
	logger.info(str(datetime.now())+": "+text)
def printwarn(text):
	logger.warning(str(datetime.now())+": "+text)
	
printlog("cMVAv2 training started")

import sys
from collections import OrderedDict

import numpy as np
import root_numpy as rnpy
import pandas
import dask

from matplotlib.colors import LogNorm
import rootpy
import rootpy.plotting

import sklearn
from sklearn import metrics
from sklearn.model_selection import KFold, ShuffleSplit

import ROOT

import matplotlib.pyplot as plt
import seaborn

import xgboost

import rootpy.plotting.root2matplotlib as rplt

#import the CMSCSTagger training library
import sklearn_cls

from pandas.plotting import scatter_matrix

from sklearn import decomposition

#mlglue********
from mlglue.tree import tree_to_tmva, BDTxgboost, BDTsklearn
from sklearn.datasets import load_svmlight_files
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
#**************

printlog("Import all modules complete.")

#**********************************************

def getcMVAver(fn,vdef):
	try:
		ddtest = pandas.concat([sklearn_cls.preprocess(sklearn_cls.load_data(
			fn,
			"tree_{0}".format(fl),
			False,
			#selection="index<1",
			start=0,
			stop=20,
			branches=vdef + ["Jet_cMVAv2"]
			),False) for fl in ["b", "c", "l"]])
		cMVAver="Jet_cMVAv2"
	except Exception:
		cMVAver="Jet_cMVA"	
	
	return cMVAver

vset=[]
defaults=[]
labels=[]
pltvars=[]
varlabels=[]


#**************************Inputs*********************************************************************

fn2 = "/afs/cern.ch/work/s/spmondal/private/cMVA/inputs/TT_91X_PU200_MiniAOD_Split.root" #TVETTDCSV_0-45_Split.root" #TVETT0-4_Split.root" #

fn3 = "/afs/cern.ch/work/s/spmondal/private/cMVA/inputs/QCD_91X_PU200_MiniAOD_Split.root" #TVEQCDDCSV_All6m_0-45_Split.root"  #TVEQCD0-4_Split.root" #TVEQCDDCSV_0-45_Split.root" #

fn2name="TT"
fn3name="QCD"

vdef=["Jet_DeepCSVBDisc"]
vdef += ["Jet_CSV", "Jet_CSVIVF", "Jet_JP", "Jet_JBP", "Jet_SoftMu", "Jet_SoftEl", "Jet_flavour", "Jet_pt", "Jet_eta", "index", "Jet_genpt"]	#All variables to load

printlog("Detecting cMVA version...")
cMVAver=getcMVAver(fn2,vdef)
printlog("cMVA version detected: "+cMVAver+"\nLoading "+fn2name+" data from "+fn2)

#Training variable sets:
#=============================================
#vset.append(["Jet_CSV", "Jet_CSVIVF", "Jet_JP", "Jet_JBP", "Jet_SoftMu", "Jet_SoftEl"])
#vset.append(["Jet_DeepCSVBDisc", "Jet_JP", "Jet_JBP", "Jet_SoftMu", "Jet_SoftEl"])
#vset.append(["Jet_CSV", "Jet_CSVIVF","Jet_DeepCSVBDisc", "Jet_JP", "Jet_JBP", "Jet_SoftMu", "Jet_SoftEl"])

#vset.append(["Jet_CSV", "Jet_CSVIVF", "Jet_JP", "Jet_JBP", "Jet_SoftMuSuppressed", "Jet_SoftEl"])
#vset.append(["Jet_DeepCSVBDisc", "Jet_JP", "Jet_JBP", "Jet_SoftMuSuppressed", "Jet_SoftEl"])
#vset.append(["Jet_CSV", "Jet_CSVIVF","Jet_DeepCSVBDisc", "Jet_JP", "Jet_JBP", "Jet_SoftMuSuppressed", "Jet_SoftEl"])

#vset.append(["Jet_CSV", "Jet_CSVIVF", "Jet_JP", "Jet_JBP", "Jet_SoftMuasEta", "Jet_SoftElasEta"])
#vset.append(["Jet_DeepCSVBDisc", "Jet_JP", "Jet_JBP", "Jet_SoftMuasEta", "Jet_SoftElasEta"])
#vset.append(["Jet_CSV", "Jet_CSVIVF","Jet_DeepCSVBDisc", "Jet_JP", "Jet_JBP", "Jet_SoftMuasEta", "Jet_SoftElasEta"])

#vset.append(["Jet_CSV", "Jet_CSVIVF", "Jet_JP", "Jet_JBP", "Jet_SoftMuSupPTEta", "Jet_SoftElasEta"])
#vset.append(["Jet_DeepCSVBDisc", "Jet_JP", "Jet_JBP", "Jet_SoftMuSupPTEta", "Jet_SoftElasEta"])
vset.append(["Jet_CSV", "Jet_CSVIVF","Jet_DeepCSVBDisc", "Jet_JP", "Jet_JBP", "Jet_SoftMuSupPTEta", "Jet_SoftElasEta"])

#vset.append(["Jet_CSVPCA", "Jet_CSVIVFPCA","Jet_DeepCSVBDiscPCA", "Jet_JP", "Jet_JBP", "Jet_SoftMu", "Jet_SoftEl"])
#vset.append(["PCA0_0", "Jet_JP", "Jet_JBP", "Jet_SoftMu", "Jet_SoftEl"])
#vset.append(["Jet_CSV", "Jet_CSVIVF","Jet_DeepCSVBDisc", "PCA0_0", "Jet_JP", "Jet_JBP", "Jet_SoftMu", "Jet_SoftEl"])
#vset.append(["Jet_CSVPCA", "Jet_CSVIVFPCA","Jet_DeepCSVBDiscPCA", "Jet_JPPCA", "Jet_JBPPCA", "Jet_SoftMuPCA", "Jet_SoftElPCA"])

#=============================================
#Labels of retrained plots

#labels.append("cMVAv2 r/w CSVv2P2")
#labels.append("cMVAv2 r/w DeepCSV")
#labels.append("cMVAv2 r/w CSVv2P2 and DeepCSV")

#labels.append("cMVAv2 r/w CSVv2P2 with SoftMu suppressed at $p_T$ > 300 GeV")
#labels.append("cMVAv2 r/w DeepCSV with SoftMu suppressed at $p_T$ > 300 GeV")
#labels.append("cMVAv2 r/w CSVv2P2 and DeepCSV with SoftMu suppressed at $p_T$ > 300 GeV")

#labels.append("cMVAv2 r/w CSVv2P2: no softLep at |eta|>1.5")
#labels.append("cMVAv2 r/w DeepCSV: no softLep at |eta|>1.5")
#labels.append("cMVAv2 r/w CSVv2P2 and DeepCSV: no softLep at |eta|>1.5")

#labels.append("cMVAv2 r/w CSVv2P2")#: no softEl at $|\eta|$>1.5, no softMu at $p_T$>300 GeV or $|\eta|$>2.4")
#labels.append("cMVAv2 r/w DeepCSV")#: no softEl at $|\eta|$>1.5, no softMu at $p_T$>300 GeV or $|\eta|$>2.4")
labels.append("cMVAv2 r/w CSVv2P2+DeepCSV")#: no softEl at $|\eta|$>1.5, no softMu at $p_T$>300 GeV or $|\eta|$>2.4")

#labels.append("cMVAv2 r/w PCA0, PCA1, PCA2 (3-comp PCA on CSVAVR, CSVIVFP2, DeepCSV)")
#labels.append("cMVAv2 r/w only major component of PCA with CSVv2P2 and DeepCSV, Weight=1/5")
#labels.append("cMVAv2 r/w CSVv2P2, DeepCSV and major PCA component, Weight=4/8")
#labels.append("cMVAv2 r/w CSVv2P2 and DeepCSV with PCA on all input features")
#==============================================

#Other variables to plot
#pltvars.append("Jet_CSVIVF")
pltvars.append(cMVAver)
#pltvars.append("Jet_DeepCSVBDisc")
pltvars.append("OldplusNewcMVA")

#Corresponding Labels
#varlabels.append("CSVv2 (Phase II)")
varlabels.append("cMVAv2 (Phase I)")
#varlabels.append("DeepCSV (Phase I)")
varlabels.append("cMVAv2 (Phase II with Phase I)")

colorlist=["red","blue","green","magenta","yellow","cyan","black","orange"]
TrainVerbose=True
saveTraining=False
saveSetindex=0     #This element of vset will be exported as .xml saved training.
savefilename="cMVAv2CD"

#********************************************************************************************************************
#Replace vset variables to look in the form "f0", "f1", etc.

vsetuni=[]
for vs in vset:
	vsetuni += vs
vsetuni=list(set(vsetuni))
vsetuni.sort()

if saveTraining:
	saveSetf=["f{0}".format(i) for i in range(len(vset[saveSetindex]))]
	saveSetnames=vset[saveSetindex][:]
	vset[saveSetindex]=saveSetf

def adapttofn(dd):
	for ind in range(len(saveSetf)):
		dd[saveSetf[ind]]=dd[saveSetnames[ind]]
		printlog("Setting \""+saveSetf[ind]+"\" equal to "+saveSetnames[ind])

#*********************************************************

def calc_roc(h1, h2, rebin=1):
    h1 = h1.Clone()
    h2 = h2.Clone()
    h1.Rebin(rebin)
    h2.Rebin(rebin)

    if h1.Integral()>0:
        h1.Scale(1.0 / h1.Integral())
    if h2.Integral()>0:
        h2.Scale(1.0 / h2.Integral())
    roc = np.zeros((h1.GetNbinsX()+2, 2))
    err = np.zeros((h1.GetNbinsX()+2, 2))
    e1 = ROOT.Double(0)
    e2 = ROOT.Double(0)
    for i in range(0, h1.GetNbinsX()+2):
        I1 = h1.Integral(0, h1.GetNbinsX())
        I2 = h2.Integral(0, h2.GetNbinsX())
        if I1>0 and I2>0:
            roc[i, 0] = float(h1.IntegralAndError(i, h1.GetNbinsX()+2, e1)) / I1
            roc[i, 1] = float(h2.IntegralAndError(i, h2.GetNbinsX()+2, e2)) / I2
            err[i, 0] = e1
            err[i, 1] = e2
    return roc, err
    
seaborn.set_style("white")
current_palette = seaborn.color_palette()


#***************Data suppressions************************
missing=-10

def suppressData(dd):
	dd["Jet_SoftMuSuppressed"] = dd["Jet_SoftMu"]
	dd.loc[dd["Jet_pt"]>300, "Jet_SoftMuSuppressed"] = missing#-10

	dd["Jet_SoftMuasEta"] = dd["Jet_SoftMu"]
	dd.loc[abs(dd["Jet_eta"])>2.4, "Jet_SoftMuasEta"] = missing#-10

	dd["Jet_SoftElasEta"] = dd["Jet_SoftEl"]
	dd.loc[abs(dd["Jet_eta"])>1.5, "Jet_SoftElasEta"] = missing#-10
	
#	dd["Jet_CSVIVFuncorr"] = dd["Jet_CSVIVF"]
#	dd.loc[np.abs(dd["Jet_CSVIVF"]-dd["Jet_CSV"])<0.00000000001, "Jet_CSVIVFuncorr"] = missing#-10
#	
#	dd["Jet_DeepCSVuncorr"] = dd["Jet_DeepCSVBDisc"]
#	dd.loc[np.abs(dd["Jet_DeepCSVBDisc"]-dd["Jet_CSV"])<0.00000000001, "Jet_DeepCSVuncorr"] = missing#-2
#	
#	dd.loc[np.abs(dd["Jet_DeepCSVBDisc"]-dd["Jet_CSVIVF"])<0.00000000001, "Jet_DeepCSVuncorr"] = missing#-2

	dd["Jet_SoftMuSupPTEta"] = dd["Jet_SoftMuasEta"]
	dd.loc[dd["Jet_pt"]>300, "Jet_SoftMuSupPTEta"] = missing#-10
	
	
#******************Principal Component Analysis*************************
def fitPCA(dd,varlist):
	n=len(varlist)
	
	Xtrain=dd.loc[dd["is_training"]==1, varlist]
	pca = decomposition.PCA(n_components=n)
	pca.fit(Xtrain)
	return pca
	
PCAlist=[]
PCAvarlist=[]

isPCA=[False for vs in vset]

def PCATrain(dd):
	PCAcount=0
	for vsind in range(len(vset)):
		foundPCA=False
		varlist=[]
		vstemp=[]
		for ind in range(len(vset[vsind])):
			if vset[vsind][ind][-3:]=="PCA":
				foundPCA=True
				varlist.append(vset[vsind][ind][:-3])
#				vset[vsind][ind] += str(PCAcount)
			else:
				vstemp.append(vset[vsind][ind])		
		if foundPCA:
			vset[vsind]=vstemp[:]
			for ind2 in range(len(varlist)):
				vset[vsind].append("PCA"+str(PCAcount)+"_"+str(ind2))
			isPCA[vsind]=True
			PCAvarlist.append(varlist)
			pca=fitPCA(dd2,varlist)
			PCAlist.append(pca)
			cov=np.matrix(pca.get_covariance())
			cor=cov/cov.trace()
			printlog("Correlations between "+str(varlist)+":\n"+str(cor))
			printlog("PCA with "+str(varlist)+" gives variance: "+str(pca.explained_variance_))
			printlog("Normalized: "+str(pca.explained_variance_ratio_))
			PCAcount += 1
		
def applyPCA(dd):
	for ind in range(len(PCAlist)):
		pca=PCAlist[ind]
		varlist=PCAvarlist[ind]
		
		Xall=dd[varlist]
		Z=pca.transform(Xall)
	
		for ind2 in range(len(varlist)):
			dd["PCA"+str(ind)+"_"+str(ind2)]=Z[:,ind2]

#************Load the training data******************
	
dd2 = pandas.concat([sklearn_cls.preprocess(sklearn_cls.load_data(
    fn2,
    "tree_{0}".format(fl),
#    selection="index<=3",
    #start=0,
    #stop=2000000,
    branches=vdef + [cMVAver]
    ),TrainVerbose) for fl in ["b", "c", "l"]])

#printwarn("NOT ALL "+fn2name+" DATA IS BEING USED!")

suppressData(dd2) 

PCATrain(dd2) 
applyPCA(dd2)

if saveTraining:   
	adapttofn(dd2)

#dd2["cMVAv2clone"] = dd2[cMVAver]

#vs2 = ["Jet_CSV", "Jet_CSVIVF", "Jet_JP", "Jet_JBP", "Jet_SoftMuSuppressed", "Jet_SoftEl"] #Suppressed

#print "dd2 index:"
#print dd2["index"].value_counts()

#print "\ndd2 flavour:"
#print dd2["flavour_category"].value_counts()

#print "\ndd2 DeepCSV:"
#print dd2["Jet_DeepCSVBDisc"].value_counts()

#print dd2.at[2,vset[0]]

#print "\ndd2 SoftMu:"
#print dd2["Jet_SoftMu"].value_counts()

#print "\ndd2 SoftEl:"
#print dd2["Jet_SoftEl"].value_counts()

#print "\ndd2 training:"
#print dd2["is_training"].value_counts()

#print dd2["is_training"].tolist()

print dd2.head()

dd2trainsize=dd2[(dd2["is_training"]==1)].shape[0]
dd2testsize=dd2[(dd2["is_training"]==0)].shape[0]

#print dd2.shape

def draw_corr(data, v1, vn1, v2, vn2):
    c = (np.abs(data[v1] -
         data[v2])<10   #>0.00000000001
    )
    hd = np.histogram2d(
        data[c][v1],
        data[c][v2],
        bins=[np.linspace(0,1,21), np.linspace(0,1,21)]
    )
    if float(np.sum(hd[0])) != 0.:
    	hd = hd[0] / float(np.sum(hd[0]))
    else:
    	hd = hd[0]
    corr = np.corrcoef(
        data[c][v1],
        data[c][v2]
    )

    plt.figure(figsize=(6,5))
    ax = plt.axes()
    ret = ax.imshow(
        hd,
        cmap="hot",
        interpolation="none",
        origin="lower",
        aspect="auto",
        #vmin=0,
        #vmax=60000,
        extent=[
            0,
            1,
            0,
            1
        ],
        norm=LogNorm(vmin=0.000001, vmax=1)
    )
    plt.title(vn1+" vs. "+vn2 + " corr={0:.6f}".format(corr[0,1]), fontsize=16)
    plt.xlabel(vn1, fontsize=16)
    plt.ylabel(vn2, fontsize=16)
    plt.colorbar(ret)
    plt.savefig('00.corr_'+vn1+'_'+vn2+'.png')
    plt.clf()
    
#draw_corr(dd2, "Jet_CSV", "CSVAVR", "Jet_CSVIVF", "CSVIVF")
#draw_corr(dd2, "Jet_CSVIVF", "CSVIVF", "Jet_DeepCSVBDisc", "DeepCSV")
#draw_corr(dd2,  "Jet_DeepCSVBDisc", "DeepCSV","Jet_CSV", "CSVAVR")

#draw_corr(dd2, "PCA0_0", "First PCA Comp", "PCA0_1", "Second PCA Comp")
#draw_corr(dd2, "PCA0_1", "Second PCA Comp", "PCA0_2", "Third PCA Comp")
#draw_corr(dd2, "PCA0_2", "Third PCA Comp", "PCA0_0", "First PCA Comp")

#draw_corr(dd2, "Jet_CSV", "CSVAVR", "Jet_CSVIVFuncorr", "CSVIVF-no-dups")
#draw_corr(dd2, "Jet_CSVIVFuncorr", "CSVIVF-no-dups", "Jet_DeepCSVuncorr", "DeepCSV-no-dups")
#draw_corr(dd2, "Jet_CSV", "CSVAVR", "Jet_DeepCSVuncorr", "DeepCSV-no-dups")
#printlog("Exported correlation plots.")
# 

def scmatrix(corrvs,nm):
	printlog("Exporting correlation plots with "+str(corrvs))

	for igroup, (name, group) in enumerate(dd2.groupby("flavour_category")):
		scatter_matrix(group.head(1000)[corrvs], alpha=0.8, figsize=(9, 9), diagonal='hist')
		
		seaborn.plt.suptitle("flavour_category = {0}".format(name), fontsize=20, y=1.02)
		seaborn.plt.tight_layout()
		
		for i in range(9):
			seaborn.plt.gcf().get_axes()[i].set_xlim(0,1)
			seaborn.plt.gcf().get_axes()[i].set_ylim(0,1)
		
		flnm='00.corrmatrix'+str(nm)+str(igroup)

		seaborn.plt.savefig(flnm+'.png')
		seaborn.plt.clf()

	printlog("Done.")

#corrvs=["Jet_CSV","Jet_CSVIVF","Jet_DeepCSVBDisc"]
#scmatrix(corrvs,"corr")
#corrvs=["Jet_CSV","Jet_CSVIVFuncorr","Jet_DeepCSVuncorr"]
#scmatrix(corrvs,"uncorr")


#***************************************
#Training using dd2
#
#***************************************

is_training = dd2["is_training"]==1

eval_sets=[]

for vs in vset:
	eval_sets.append([
		#training set, ttjets
		(dd2.ix[is_training, vs], dd2.ix[is_training, "flavour_category"]==2, dd2.ix[is_training, "weight"]),
		#testing set
		(dd2.ix[np.invert(is_training), vs], dd2.ix[np.invert(is_training), "flavour_category"]==2, dd2.ix[np.invert(is_training), "weight"])
	])

print dd2.groupby(["flavour_category", "is_training"]).apply(lambda x: len(x))

printlog("Beginning training using xgboost...")

clses=[]
clstst=[]
	
#Manual
#clses.append(xgboost.XGBClassifier(n_estimators=2000, learning_rate=0.2, nthread=32, subsample = 0.8, gamma=1.5))  #cMVAv2 r/w CSVv2P2
#clses.append(xgboost.XGBClassifier(n_estimators=2000, nthread=32, learning_rate=0.1)) #cMVAv2 r/w DeepCSV
#clses.append(xgboost.XGBClassifier(n_estimators=2000, learning_rate=0.2, nthread=32, subsample = 0.5, gamma=5))  #cMVAv2 r/w CSVv2P2 and DeepCSV
##clses.append(xgboost.XGBClassifier(n_estimators=2000, learning_rate=0.2, nthread=32, gamma=1.8))  #cMVAv2 r/w CSVv2P2 and DeepCSV for QCD training

#clstst.append(xgboost.XGBClassifier(n_estimators=500, nthread=32, learning_rate=0.2))
#clstst.append(xgboost.XGBClassifier(n_estimators=500, nthread=32, learning_rate=0.2))
#clstst.append(xgboost.XGBClassifier(n_estimators=500, nthread=32, learning_rate=0.2))
	
##Automatic
for vsind in range(len(clses),len(vset)):
	clses.append(xgboost.XGBClassifier(n_estimators=200, nthread=32, learning_rate=0.2))
#	clstst.append(xgboost.XGBClassifier(n_estimators=2000, nthread=32, learning_rate=0.2, gamma=1.8))


features=[]
target_names = ["cls{0}".format(i) for i in [0,1]]

for ind in range(len(clses)):
	printlog("Training on "+str(vset[ind]))
	clses[ind].fit(eval_sets[ind][0][0], eval_sets[ind][0][1], eval_sets[ind][0][2], eval_set=eval_sets[ind], early_stopping_rounds=100, eval_metric=["error"], verbose=TrainVerbose)
	if len(clstst)>0:
		clstst[ind].fit(eval_sets[ind][0][0], eval_sets[ind][0][1], eval_sets[ind][0][2], eval_set=eval_sets[ind], early_stopping_rounds=50, eval_metric=["error"], verbose=TrainVerbose)
	printlog("Training step "+str(ind+1)+" of "+str(len(clses))+" done.")
	
	if saveTraining and ind==saveSetindex:
		printlog("Saving training with features: "+str(saveSetnames))
		#features.append(["f{0}".format(i) for i in range(len(vset[ind]))])
		bdt=BDTxgboost(clses[ind], saveSetf, target_names)
		bdt.to_tmva(savefilename+".xml")
		bdt.setup_tmva(savefilename+".xml")
		printlog("Exported training step "+str(ind+1)+" of "+str(len(clses))+" to \""+savefilename+".xml\"")
	


def insertcMVAdefault(dd):
	for ind in range(len(clses)):
		cond = (dd["Jet_JP"]==0.) & (dd["Jet_JBP"]==0.)
		for feat in ["Jet_CSV", "Jet_CSVIVF", "Jet_DeepCSVBDisc", "Jet_SoftMu", "Jet_SoftEl", "Jet_SoftMuasEta", "Jet_SoftMuSuppressed", "Jet_SoftMuSupPTEta", "Jet_SoftElasEta"]:
			if feat in vset[ind]:			
				cond = cond & (dd[feat]==missing)
				
#		cond=(dd["Jet_SoftMu"]==missing) & (dd["Jet_SoftEl"]==missing)
		
#		for feat in ["Jet_CSV", "Jet_CSVIVF", "Jet_DeepCSVBDisc"]:
#			if feat in vset[ind]:
#				cond=cond & (dd[feat]==missing)
#				cond=(dd["Jet_CSV"]==missing) & (dd["Jet_CSVIVF"]==missing) & (dd["Jet_DeepCSVBDisc"]==missing) & (dd["Jet_SoftMu"]==missing) & (dd["Jet_SoftEl"]==missing)
#		print vset[ind]
##		print str(cond)
#		print dd[cond]
		dd.loc[cond,"cls_p"+str(ind)]=missing
		
def insertNewCombinedDisc(dd,newcMVAname):
	dd["OldplusNewcMVA"]=dd[newcMVAname]
	cond = (dd[newcMVAname]==missing) & (dd[cMVAver]!=-1.)
	dd.loc[cond, "OldplusNewcMVA"] = (dd.loc[cond, cMVAver]+1.)/2
	
	dd["OldcMVAplusDeepCSV"]=dd["Jet_DeepCSVBDisc"]
	cond = (dd["Jet_DeepCSVBDisc"]==missing) & (dd[cMVAver]!=-1.)
	dd.loc[cond, "OldcMVAplusDeepCSV"] = (dd.loc[cond, cMVAver]+1.)/2

printlog("Testing on "+fn2name+" samples...")

for ind in range(len(clses)):
	dd2["cls_p"+str(ind)] = clses[ind].predict_proba(dd2[vset[ind]])[:, 1]
	if len(clstst)>0:
		dd2["cls_tst"+str(ind)] = clstst[ind].predict_proba(dd2[vset[ind]])[:, 1]
	
insertcMVAdefault(dd2)
insertNewCombinedDisc(dd2,"cls_p0")

printlog("Done.")

#b_vs_udsg = dd["flavour_category"] != 1
#b_vs_c = dd["flavour_category"] != 0


#***************************************
#Discriminator performance
#
#***************************************

printlog("Generating discriminator plots...")
#toprobe="OldplusNewcMVA"
#print dd2.loc[(dd2[toprobe]>0.15) & (dd2[toprobe]<0.85),toprobe].value_counts()
#print "\n\n\n"
#spike=dd2.loc[(dd2[toprobe]>0.15) & (dd2[toprobe]<0.85),toprobe].value_counts().index[0]
#print dd2.loc[(dd2[toprobe]>spike-1.e-6) & (dd2[toprobe]<spike+1.e-6)]


def hist_from_array(arr, bins):
#	#*******Weed out default values*********
#    arr2=[]
#    for x in arr:
#    	if x!=-10:
#    		arr2.append(x)
#    arr=np.array(arr2)
#    #***************************************
    h = rootpy.plotting.Hist(*bins)
    weights = np.ones(len(arr), dtype=np.double)
    h.FillN(len(arr), arr.astype(np.double), weights)  
    h.SetBinContent(1, h.GetBinContent(0) + h.GetBinContent(1))
    h.SetBinContent(0, 0)
    
    h.SetBinContent(h.GetNbinsX(), h.GetBinContent(h.GetNbinsX()) + h.GetBinContent(h.GetNbinsX() + 1))
    h.SetBinContent(h.GetNbinsX()+1, 0)
    
    return h
    
def get_hists(dd, cut, var, bins):
    h1b = hist_from_array(dd[(dd["flavour_category"] == 2) & cut][var].as_matrix(), bins)
    h1c = hist_from_array(dd[(dd["flavour_category"] == 1) & cut][var].as_matrix(), bins)
    h1u = hist_from_array(dd[(dd["flavour_category"] == 0) & cut][var].as_matrix(), bins)
    return h1b, h1c, h1u


for etalow, etahigh in [(0.,3.5),(0.,1.5),(1.5,2.5),(2.5,3.5)]:
	h1b, h1c, h1u = get_hists(dd2, (dd2["is_training"]==0) & (abs(dd2["Jet_eta"])>=etalow) & (abs(dd2["Jet_eta"])<etahigh), "cls_p0", (200, 0, 1))

	h2b, h2c, h2u = get_hists(dd2, (dd2["is_training"]==1) & (abs(dd2["Jet_eta"])>=etalow) & (abs(dd2["Jet_eta"])<etahigh), "cls_p0", (200, 0, 1))


	for h in [h1b, h2b, h1c, h2c, h1u, h2u]:
		h.Scale(1.0/h.Integral())

	plt.figure(figsize=(9,3))
	plt.title("Discriminator plots, "+str(etalow)+" < $|\eta|$ < "+str(etahigh))
	plt.subplot(1,3,1)
	rplt.step(h1b, color="red", label="testing")
	rplt.step(h2b, color="blue", label="training")
	plt.xlabel("discriminator")
	plt.ylabel("fraction of jets")
	plt.title("b")
	plt.yscale("log")
	plt.legend(loc="best")

	plt.subplot(1,3,2)
	rplt.step(h1c, color="red")
	rplt.step(h2c, color="blue")
	plt.xlabel("discriminator")
	plt.ylabel("fraction of jets")
	plt.title("charm")
	plt.yscale("log")

	plt.subplot(1,3,3)
	rplt.step(h1u, color="red")
	rplt.step(h2u, color="blue")
	plt.xlabel("discriminator")
	plt.ylabel("fraction of jets")
	plt.title("udsg")
	plt.yscale("log")

	plt.tight_layout()

	#plt.savefig("01.disriminators.pdf")
	plt.savefig("01.disriminators_eta"+str(etalow)+"-"+str(etahigh)+".png")
	plt.clf()
	
	
#********************Export discriminator in a .root file********************
#printlog("Done. Now exporting discriminators.")

#def exp_disc(dd,cond,colname,fil,discname,histtitle,low,high):
#	hist=ROOT.TH1F(discname,histtitle,1000,low,high)
#	for val in dd.loc[cond, colname]:
#		if val!=missing:
#			hist.Fill(val,1)
#		else:
#			hist.Fill(0.,1)
#	fil.Write()
#	
#try:
#	f1=ROOT.TFile(fn2name+"_Discs.root","RECREATE")
#except Exception:
#	f1=ROOT.TFile(fn2name+"_Discs.root","CREATE")

#cond1=(dd2["is_training"]==0)  & (dd2["Jet_genpt"]>30) #& (dd2["Jet_pt"]>30)

#for colname,leafname in [("Jet_CSVIVF","CSVIVF"),("Jet_DeepCSVBDisc","DeepCSV"),("cls_p0","NewcMVA"),("Jet_pt","Jet_pt"),(cMVAver,"OldcMVA"),("OldplusNewcMVA","CombDisc"), ("OldcMVAplusDeepCSV","OldcMVAplusDeepCSV")]:
#	for flav,flavname in [(-1,"All"),(0,"Light"),(1,"C"),(2,"B")]:
#		if flav!=-1:
#			cond2 = cond1 & (dd2["flavour_category"]==flav)
#		else:
#			cond2 = cond1
#		for etalow, etahigh in [(0.,3.5),(0.,1.5),(1.5,2.5),(2.5,3.5)]:
#			cond3 = cond2 & (abs(dd2["Jet_eta"])>=etalow) & (abs(dd2["Jet_eta"])<etahigh)		
#			discname=leafname+"_"+flavname+"_eta"+str(etalow)+"-"+str(etahigh)
#			if colname=="Jet_pt":
#				high=2000.
#				histtitle="Value"
#			else:
#				high=1.
#				histtitle="Discriminator"
#			if colname==cMVAver:
#				low=-1
#			else:
#				low=0.
#			exp_disc(dd2,cond3,colname,f1,discname,histtitle,low,high)		
#	

#f1.Close()
	
#****************************************************************************	
	

printlog("Done. Now generating ROC curves.")


#***************************************
#The main performance plot (mistag vs tag eff). This is being reported for dd2 non-is_training samples only.
#
#***************************************
 
def pltfilew(fil,xaxisu,yaxixu,xaxisc,yaxisc,label,color):
	fil.write("%label="+label+"\n")
	fil.write("%color="+color+"\n")
	for axis in [xaxisu,yaxixu,xaxisc,yaxisc]:
		for elem in axis:
			fil.write(str(elem)+" ")
		fil.write("\n")
	fil.write("\n")

def plot_perf_histo(ddn,cuts,plottitle="CMS Phase II",filenm="mistagvstageff"):
 
#	hb_csv, hc_csv, hu_csv = get_hists(ddn, cuts, "Jet_CSVIVF", (1000, 0, 1))
#	hb_cmva, hc_cmva, hu_cmva = get_hists(ddn, cuts, cMVAver, (2000, -1, 1))
	
	ru=[[] for ind in range(len(clses))]
	rc=[[] for ind in range(len(clses))]
	rpu=[[] for ind in range(len(pltvars))]
	rpc=[[] for ind in range(len(pltvars))]
	
	for ind in range(len(clses)):
		hb, hc, hu = get_hists(ddn, cuts, "cls_p"+str(ind), (1000, 0, 1))
		ru[ind], e = calc_roc(hb, hu)
		rc[ind], e = calc_roc(hb, hc)

	for ind in range(len(pltvars)):
		if pltvars[ind]!=cMVAver:
			lowlim=0
		else:
			lowlim=-1
		hb, hc, hu = get_hists(ddn, cuts, pltvars[ind], (1000*(1-lowlim), lowlim, 1))
		rpu[ind], e = calc_roc(hb, hu)
		rpc[ind], e = calc_roc(hb, hc)


	plt.figure(figsize=(10,10))
	
	pltfile=open(filenm+".pltdata","w")
	
	pltfile.write("#Title="+plottitle+"\n\n\n")
	
	col=0
	col2=-1
	
	for ind in range(len(clses)):
		label=labels[ind]+" AUC={0:.3f} ({1:.3f})".format(
			sklearn.metrics.auc(ru[ind][:, 0], ru[ind][:, 1]),
			sklearn.metrics.auc(rc[ind][:, 0], rc[ind][:, 1])
		)
		plt.plot(ru[ind][:, 0], ru[ind][:, 1], color=colorlist[col], label=label)
		plt.plot(rc[ind][:, 0], rc[ind][:, 1], color=colorlist[col])
	
		pltfilew(pltfile,ru[ind][:, 0], ru[ind][:, 1],rc[ind][:, 0], rc[ind][:, 1],label,colorlist[col])
		
		col += 1
	
	for ind in range(len(pltvars)):
		label2=varlabels[ind]+" AUC={0:.3f} ({1:.3f})".format(
			sklearn.metrics.auc(rpu[ind][:, 0], rpu[ind][:, 1]),
			sklearn.metrics.auc(rpc[ind][:, 0], rpc[ind][:, 1])
		)
		plt.plot(rpu[ind][:, 0], rpu[ind][:, 1], color=colorlist[col2], label=label2)
		plt.plot(rpc[ind][:, 0], rpc[ind][:, 1], color=colorlist[col2])
	
		pltfilew(pltfile,rpu[ind][:, 0], rpu[ind][:, 1],rpc[ind][:, 0], rpc[ind][:, 1],label2,colorlist[col2])
		
		col2 -= 1
	
	plt.yscale("log")
	plt.xlim(0.2,1.0)
	plt.ylim(0.001, 1.0)
	plt.legend(loc=2)
	plt.xlabel("b-tagging efficiency", fontsize=16)
	plt.ylabel("mistag probability", fontsize=16)
	plt.title(plottitle)
	
	pltfile.close()
#	plt.savefig(filenm+".pdf")
	plt.savefig(filenm+".png")
	plt.clf()


for etalow, etahigh in [(0.,3.5),(0.,1.5),(1.5,2.5),(2.5,3.5)]:
			
	jetcut=30
	
	cuts=(dd2["is_training"]==0) & (abs(dd2["Jet_eta"])>=etalow) & (abs(dd2["Jet_eta"])<etahigh) & (dd2["Jet_pt"]>jetcut)
	plot_perf_histo(dd2,cuts,"CMS Phase II, "+fn2name+", $p_T > "+str(jetcut)+"$ GeV, "+str(etalow)+" < |eta| < "+str(etahigh),"02."+fn2name+"_jet"+str(jetcut)+"_eta"+str(etalow)+"-"+str(etahigh))
		
for jetlow, jethigh in [(150.,300.),(300.,600.),(600.,1000.)]:
	etalow=0.
	etahigh=3.5	
	
	cuts=(dd2["is_training"]==0) & (abs(dd2["Jet_eta"])>=etalow) & (abs(dd2["Jet_eta"])<etahigh) & (dd2["Jet_pt"]>=jetlow) & (dd2["Jet_pt"]<jethigh)
	plot_perf_histo(dd2,cuts,"CMS Phase II, "+fn2name+", "+str(jetlow)+" < $p_T$ < "+str(jethigh)+" GeV, "+str(etalow)+" < $|\eta|$ < "+str(etahigh),"03."+fn2name+"_jet"+str(jetlow)+"-"+str(jethigh)+"_eta"+str(etalow)+"-"+str(etahigh))

printlog("Done. Generating error and AUC plots...")

for ind in range(len(clses)):
	plt.figure(figsize=(8,8))
	plt.plot(clses[ind].evals_result_["validation_0"]["error"], label="Train")
	plt.plot(clses[ind].evals_result_["validation_1"]["error"], label="Test")
	if len(clstst)>0:
		plt.plot(clstst[ind].evals_result_["validation_0"]["error"], "b--", label="Train (old)")
		plt.plot(clstst[ind].evals_result_["validation_1"]["error"], "g--", label="Test (old)")
	
	plt.legend()
	plt.xlabel("Boosting iteration", fontsize=16)
	plt.ylabel("Error", fontsize=16)
	plt.title(fn2name+"\n"+labels[ind])
	#plt.savefig("04.error_"+str(ind)+".pdf")
	plt.savefig("04.error_"+str(ind)+".png")
	plt.clf()
	

def get_auc_by_group(dd2, c1, c2, discs, col):
    """
    c1 (int): first category, e.g. signal
    c2 (int): second category, e.g. background
    discs (list of strings): discriminators to plot
    col (string): column to split on
    """
    vals = {d: [] for d in discs}
    vals_err = {d: [] for d in discs}
    for grname, gr in dd2.groupby(col):
        vals_bin = {d: [] for d in discs}
        
        #randomly select subsample of events to calculate metric on
        kf = ShuffleSplit(n_splits = 5)
        
        shuf_index = np.random.permutation(gr.index)
        
        for train, test in kf.split(shuf_index):
            for disc in discs:
                mask = np.zeros(len(gr), dtype=np.bool)
                mask[train] = True
                ha = hist_from_array(gr[(gr["flavour_category"] == c1) & mask][disc].as_matrix(), (1000, -1, 1))
                hb = hist_from_array(gr[(gr["flavour_category"] == c2) & mask][disc].as_matrix(), (1000, -1, 1))
                ru, e = calc_roc(ha, hb)
                vals_bin[disc] += [sklearn.metrics.auc(ru[:, 0], ru[:, 1])]
        for d in discs:
            vals[d] += [np.mean(vals_bin[d])]
            vals_err[d] += [np.std(vals_bin[d])]
    return vals, vals_err



for fl in [0,1]:
	for xax in ["ptbin2","etabin2"]:
		discs = pltvars[:] 			#["Jet_CSVIVF", cMVAver]#, "cls_p1", "cls_p4"]		#, "cls_p2"]
		for ind in range(len(clses)):
			discs.append("cls_p"+str(ind))
		vals, vals_err = get_auc_by_group(dd2, 2, fl, discs, xax)
		for d in discs:
			try:
				if xax=="ptbin2":
					plt.errorbar(sklearn_cls.ptbins2, vals[d], yerr=vals_err[d], label=d, marker="o")
				else:
					plt.errorbar(sklearn_cls.etabins2, vals[d], yerr=vals_err[d], label=d, marker="o")
			except Exception:
				print "EXCEPTION in "+fn2name+"! Debug:"
				print xax
				print fl
				print len(sklearn_cls.etabins2[:])
				print len(vals[d])
				print len(vals)
				print d
				pass
			
		plt.legend(loc=4)
		if xax=="ptbin2":
			plt.xlabel("$p_T$ [GeV]")
			propname="pT"
		else:
			plt.xlabel("$|\eta|$")
			propname="eta"
			
		if fl==1:
			flname="c"
		else:
			flname="l"
		plt.ylabel("AUC")
		plt.title("CMS Phase II, "+fn2name+", b vs "+flname)

#		plt.savefig("05.AUC "+propname+" b vs. "+flname+".pdf")
		plt.savefig("05.AUC "+propname+" b vs. "+flname+".png")
		plt.clf()


#discs = pltvars[:] #["Jet_CSVIVF", cMVAver]#, "cls_p1", "cls_p4"]		#, "cls_p2"]
#for ind in range(len(clses)):
#	discs.append("cls_p"+str(ind))
#vals, vals_err = get_auc_by_group(dd2, 2, 1, discs, "ptbin2")
#for d in discs:
#    plt.errorbar(sklearn_cls.ptbins2, vals[d], yerr=vals_err[d], label=d, marker="o")
#plt.legend(loc=4)
#plt.xlabel("$p_T$ [GeV]")
#plt.ylabel("AUC")
#plt.title("CMS Phase II, tt+jets, b vs charm")

#plt.savefig("08.AUC pT b vs. c.pdf")
#plt.savefig("08.AUC pT b vs. c.png")
#plt.clf()

#discs = pltvars[:]#["Jet_CSVIVF", cMVAver]#, "cls_p1", "cls_p4"]#, "cls_p2"]
#for ind in range(len(clses)):
#	discs.append("cls_p"+str(ind))
#vals, vals_err = get_auc_by_group(dd2, 2, 0, discs, "ptbin2")
#for d in discs:
#    plt.errorbar(sklearn_cls.ptbins2, vals[d], yerr=vals_err[d], label=d, marker="o")
#plt.legend(loc=4)
#plt.xlabel("$p_T$ [GeV]")
#plt.ylabel("AUC")
#plt.title("CMS Phase II, tt+jets, b vs light")

#plt.savefig("09.AUC pT b vs. l.pdf")
#plt.savefig("09.AUC pT b vs. l.png")
#plt.clf()

#print str(datetime.now())
#print "Generated AUC pT plots."


#discs = pltvars[:]#["Jet_CSVIVF", cMVAver]#, "cls_p1", "cls_p4"]#, "cls_p2"]
#for ind in range(len(clses)):
#	discs.append("cls_p"+str(ind))
#vals, vals_err = get_auc_by_group(dd2, 2, 1, discs, "etabin2")
#for d in discs:
#	try:
#		plt.errorbar(sklearn_cls.etabins2[:], vals[d], yerr=vals_err[d], label=d, marker="o") #sklearn_cls.etabins2[:-1]
#	except Exception:
#		print "EXCEPTION!! Debug:"
#		print len(sklearn_cls.etabins2[:])
#		print len(vals[d])
#		print len(vals)
#		print d
#		pass
#    	
#    	
#plt.legend(loc=4)
#plt.xlabel("$|\eta|$")
#plt.ylabel("AUC")
#plt.title("CMS Phase II, tt+jets, b vs charm")

#plt.savefig("10.AUC eta b vs. c.pdf")
#plt.savefig("10.AUC eta b vs. c.png")
#plt.clf()

#discs = pltvars[:]#["Jet_CSVIVF", cMVAver]#, "cls_p1", "cls_p4"]#, "cls_p2"]
#for ind in range(len(clses)):
#	discs.append("cls_p"+str(ind))
#vals, vals_err = get_auc_by_group(dd2, 2, 0, discs, "etabin2")
#for d in discs:
#	try:
#		plt.errorbar(sklearn_cls.etabins2[:], vals[d], yerr=vals_err[d], label=d, marker="o")
#	except Exception:
#		print "EXCEPTION!! Debug:"
#		print len(sklearn_cls.etabins2[:])
#		print len(vals[d])
#		print len(vals)
#		print d
#		pass
#		
#plt.legend(loc=4)
#plt.xlabel("$|\eta|$")
#plt.ylabel("AUC")
#plt.title("CMS Phase II, tt+jets, b vs light")

#plt.savefig("11.AUC eta b vs. l.pdf")
#plt.savefig("11.AUC eta b vs. l.png")
#plt.clf()

printlog("Done. Loading "+fn3name+" data from "+fn3)

#dd2["Jet_pt"].hist(bins=np.linspace(0, 300, 300), lw=0)
#plt.yscale("log")
#plt.xlabel("Jet pt")

#plt.savefig("06.eventsvspT.pdf")
#plt.savefig("06.eventsvspT.png")
#plt.clf()

#dd2["Jet_eta"].hist(bins=np.linspace(-2.5, 2.5, 300), lw=0)
#plt.yscale("log")
#plt.xlabel("Jet eta")

#plt.savefig("06.eventsvseta.pdf")
#plt.savefig("06.eventsvseta.png")
#plt.clf()

#*********************************************************************************************************
#QCD Sample
#
#*********************************************************************************************************


dd3 = pandas.concat([sklearn_cls.preprocess(sklearn_cls.load_data(
    fn3,
    "tree_{0}".format(fl),
    #selection="index>=8",
    #start=0,
    #stop=10000000,
    branches=vdef + [cMVAver]
),TrainVerbose) for fl in ["b", "c", "l"]])

suppressData(dd3)
applyPCA(dd3)

if saveTraining:
	adapttofn(dd3)

#print "\ndd2 training:"
#print dd2["is_training"].value_counts()

#print "\ndd3 index:"
#print dd3["index"].value_counts()

#print "\ndd3 flavour:"
#print dd3["flavour_category"].value_counts()

#print dd3.shape

#print "\ndd3 training:"
#print dd3["is_training"].value_counts()

print dd3.head()

#dd3["Jet_pt"].hist(bins=np.linspace(0, 2000, 300), lw=0)
#plt.yscale("log")
#plt.savefig("14."+fn3name+"JetPT.pdf")
#plt.savefig("14."+fn3name+"JetPT.png")
#plt.clf()

#dd3[vs].head()

dd3testsize=dd3.shape[0]

printlog("Testing on "+fn3name+" samples...")
dev=[]

for ind in range(len(clses)):
	dd3["cls_p"+str(ind)] = clses[ind].predict_proba(dd3[vset[ind]])[:, 1]
insertcMVAdefault(dd3)
insertNewCombinedDisc(dd3,"cls_p0")

#***************Find number of null values***********************************************
if len(clses)>=0:
	cols=["Jet_CSV","Jet_CSVIVF", "Jet_DeepCSVBDisc", "cls_p0", "OldplusNewcMVA"]
	nullstats=[]

	for col in cols:
		colstat=[]
		cond=(dd3["Jet_genpt"]>30) #(dd3["Jet_pt"]>30) & 
		TotNull=float(dd3[(dd3[col]==missing) & cond].shape[0])
		TotVal=float(dd3[(dd3[col]!=missing) & cond].shape[0])
		TotPer=TotNull/(TotNull+TotVal)*100.
	
		LowNull=float(dd3[(dd3[col]==missing) & (abs(dd3["Jet_eta"])<=2.5) & cond].shape[0])
		LowVal=float(dd3[(dd3[col]!=missing) & (abs(dd3["Jet_eta"])<=2.5) & cond].shape[0])
		LowPer=LowNull/(LowNull+LowVal)*100.
	
		HighNull=float(dd3[(dd3[col]==missing) & (abs(dd3["Jet_eta"])>=2.5) & cond].shape[0])
		HighVal=float(dd3[(dd3[col]!=missing) & (abs(dd3["Jet_eta"])>=2.5) & cond].shape[0])
		HighPer=HighNull/(HighNull+HighVal)*100.
	
		colstat=[TotPer,LowPer,HighPer]
	
	#	printlog("Total points with "+col+"==-10: "+str(TotNull))
	#	printlog("Total points with "+col+"!=-10: "+str(TotVal))

	#	printlog("Points with "+col+"==-10 at low eta: "+str(LowNull))
	#	printlog("Points with "+col+"!=-10 at low eta: "+str(LowVal))
	#	
	#	printlog("Points with "+col+"==-10 at high eta: "+str(HighNull))
	#	printlog("Points with "+col+"!=-10 at high eta: "+str(HighVal))
	
		nullstats.append(colstat)

	N = 3

	ind = np.arange(N)  # the x locations for the groups
	width = 0.15       # the width of the bars

	fig, ax = plt.subplots()
	rects1 = ax.bar(ind, nullstats[0], width, color='r')
	rects2 = ax.bar(ind + width, nullstats[1], width, color='y')
	rects3 = ax.bar(ind + 2*width, nullstats[2], width, color='b')
	rects4 = ax.bar(ind + 3*width, nullstats[3], width, color='g')
	rects5 = ax.bar(ind + 4*width, nullstats[4], width, color='magenta')

	ax.set_ylabel('Percentage of NULL value outputs (Jet gen $p_T$ > 30 GeV)')
	ax.set_xticks(ind + 2*width)
	ax.set_xticklabels(("Overall", "$|\eta|<2.5$","$|\eta|>2.5$"))

	ax.legend((rects1[0], rects2[0],rects3[0],rects4[0],rects5[0]), ["CSVAVR","CSVIVF", "DeepCSV", "P-2 cMVAv2 (C+D)", "Old+New cMVA"], loc="best")

	def autolabel(rects):
		"""
		Attach a text label above each bar displaying its height
		"""
		for rect in rects:
		    height = rect.get_height()
		    ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
		            '{:.2f}'.format(height),
		            ha='center', va='bottom', fontsize=8)

	autolabel(rects1)
	autolabel(rects2)
	autolabel(rects3)
	autolabel(rects4)
	autolabel(rects5)

	plt.savefig("09.NullStats.png")
	plt.clf()

#***************************************************************************************************

#	if saveTraining:
#		dev.append(0)
#		for irow in range(data_x.shape[0]):
#		 	predA1 = bdt.eval_tmva(dd3.at[irow, vset[ind]])
#		 	predB1 = dd3.at[irow,"cls_p"+str(ind)][0]
#		 	dev[ind] += np.abs((predA1 - predB1)/predA1)
#		print dev[ind]	
	
#dd3["cls_p2"] = cls2.predict_proba(dd3[vs2])[:, 1]
#dd3["cls_p4"] = cls4.predict_proba(dd3[vs3])[:, 1]

printlog("Done. Generating plots...")

#===========
#Plot 2: QCD ===> dd3
#===========


for etalow, etahigh in [(0.,3.5),(0.,1.5),(1.5,2.5),(2.5,3.5)]:
		
	jetcut=30
	
	cuts=(abs(dd3["Jet_eta"])>=etalow) & (abs(dd3["Jet_eta"])<etahigh) & (dd3["Jet_pt"]>jetcut)
	plot_perf_histo(dd3,cuts,"CMS Phase II, "+fn3name+", $p_T$ > "+str(jetcut)+" GeV, "+str(etalow)+" < $|\eta|$ < "+str(etahigh),"06."+fn3name+"_jet"+str(jetcut)+"_eta"+str(etalow)+"-"+str(etahigh))
		
for jetlow, jethigh in [(150.,300.),(300.,600.),(600.,1000.)]:
	etalow=0.
	etahigh=3.5	
	
	cuts=(abs(dd3["Jet_eta"])>=etalow) & (abs(dd3["Jet_eta"])<etahigh) & (dd3["Jet_pt"]>=jetlow) & (dd3["Jet_pt"]<jethigh)
	plot_perf_histo(dd3,cuts,"CMS Phase II, "+fn3name+", "+str(jetlow)+" < $p_T$ < "+str(jethigh)+" GeV, "+str(etalow)+" < |$\eta$| < "+str(etahigh),"07."+fn3name+"_jet"+str(jetlow)+"-"+str(jethigh)+"_eta"+str(etalow)+"-"+str(etahigh))
   

ptbins3 = np.logspace(1, 4, 11)

dd3["ptbin3"] = map(lambda x: ptbins3.searchsorted(x), dd3["Jet_pt"])

for fl in [0,1]:
	for xax in ["ptbin2","etabin2"]:
		discs = pltvars[:] 			#["Jet_CSVIVF", cMVAver]#, "cls_p1", "cls_p4"]		#, "cls_p2"]
		for ind in range(len(clses)):
			discs.append("cls_p"+str(ind))
		vals, vals_err = get_auc_by_group(dd3, 2, fl, discs, xax)
		for d in discs:
			try:
				if xax=="ptbin2":
					plt.errorbar(sklearn_cls.ptbins2, vals[d], yerr=vals_err[d], label=d, marker="o")
				else:
					plt.errorbar(sklearn_cls.etabins2, vals[d], yerr=vals_err[d], label=d, marker="o")
			except Exception:
				print "EXCEPTION in "+fn3name+"! Debug:"
				print xax
				print fl
				print len(sklearn_cls.etabins2[:])
				print len(vals[d])
				print len(vals)
				print d
				pass
		plt.legend(loc=4)
		if xax=="ptbin2":
			plt.xlabel("$p_T$ [GeV]")
			propname="pT"
		else:
			plt.xlabel("$|\eta|$")
			propname="eta"
			
		if fl==1:
			flname="c"
		else:
			flname="l"
		plt.ylabel("AUC")
		plt.title("CMS Phase II, "+fn3name+", b vs "+flname)

#		plt.savefig("08.AUC "+fn3name+" "+propname+" b vs. "+flname+".pdf")
		plt.savefig("08.AUC "+fn3name+" "+propname+" b vs. "+flname+".png")
		plt.clf()

#discs = pltvars[:]#["Jet_CSVIVF", cMVAver]#, "cls_p1", "cls_p4"]#, "cls_p2"]
#for ind in range(len(clses)):
#	discs.append("cls_p"+str(ind))
#vals, vals_err = get_auc_by_group(dd3, 2, 0, discs, "ptbin2")
#for d in discs:
#    plt.errorbar(sklearn_cls.ptbins2, vals[d], yerr=vals_err[d], label=d, marker="o")
#plt.legend(loc=4)
#plt.xlabel("$p_T$ [GeV]")
#plt.ylabel("AUC")
#plt.title("CMS Phase II, QCD, b vs light")
#plt.savefig("16.AUC QCD pt b vs l.pdf")
#plt.savefig("16.AUC QCD pt b vs l.png")
#plt.clf()

#discs = pltvars[:]#["Jet_CSVIVF", cMVAver]#, "cls_p1", "cls_p4"]#, "cls_p2"]
#for ind in range(len(clses)):
#	discs.append("cls_p"+str(ind))
#vals, vals_err = get_auc_by_group(dd3, 2, 1, discs, "ptbin2")
#for d in discs:    
#	try:
#		plt.errorbar(sklearn_cls.ptbins2, vals[d], yerr=vals_err[d], label=d, marker="o")
#	except Exception:
#		print "EXCEPTION at AUC QCD pt b vs c!! Debug:"
#		print len(sklearn_cls.etabins2[:])
#		print len(vals[d])
#		print len(vals)
#		print d
#		pass
#plt.legend(loc=4)
#plt.xlabel("$p_T$ [GeV]")
#plt.ylabel("AUC")
#plt.title("CMS Phase II, QCD, b vs charm")
#plt.savefig("17.AUC QCD pt b vs c.pdf")
#plt.savefig("17.AUC QCD pt b vs c.png")
#plt.clf()

#discs = pltvars[:]#["Jet_CSVIVF", cMVAver]#, "cls_p1", "cls_p4"]#, "cls_p2"]
#for ind in range(len(clses)):
#	discs.append("cls_p"+str(ind))
#vals, vals_err = get_auc_by_group(dd3, 2, 1, discs, "etabin2")
#for d in discs:
#	try:
#		plt.errorbar(sklearn_cls.etabins2[:], vals[d], yerr=vals_err[d], label=d, marker="o") #sklearn_cls.etabins2[:-1]
#	except Exception:
#		print "EXCEPTION at AUC QCD eta b vs. c!! Debug:"
#		print len(sklearn_cls.etabins2[:])
#		print len(vals[d])
#		print len(vals)
#		print d
#		pass    	
#    	
#plt.legend(loc=4)
#plt.xlabel("$|\eta|$")
#plt.ylabel("AUC")
#plt.title("CMS Phase II, tt+jets, b vs charm")

#plt.savefig("18.AUC QCD eta b vs. c.pdf")
#plt.savefig("18.AUC QCD eta b vs. c.png")
#plt.clf()

#discs = pltvars[:]#["Jet_CSVIVF", cMVAver]#, "cls_p1", "cls_p4"]#, "cls_p2"]
#for ind in range(len(clses)):
#	discs.append("cls_p"+str(ind))
#vals, vals_err = get_auc_by_group(dd3, 2, 0, discs, "etabin2")
#for d in discs:
#	try:
#		plt.errorbar(sklearn_cls.etabins2[:], vals[d], yerr=vals_err[d], label=d, marker="o")
#	except Exception:
#		print "EXCEPTION at AUC QCD eta b vs. l!! Debug:"
#		print len(sklearn_cls.etabins2[:])
#		print len(vals[d])
#		print len(vals)
#		print d
#		pass
#		
#plt.legend(loc=4)
#plt.xlabel("$|\eta|$")
#plt.ylabel("AUC")
#plt.title("CMS Phase II, tt+jets, b vs light")

#plt.savefig("19.AUC QCD eta b vs. l.pdf")
#plt.savefig("19.AUC QCD eta b vs. l.png")
#plt.clf()
printlog("All plots generated.")

summary="Trained on:\n* "+fn2+" : "+str(dd2trainsize)+" jets.\n\nTested on:\n* "+fn2+" : "+str(dd2testsize)+" jets.\n* "+fn3+" : "+str(dd3testsize)+" jets.\n\nOutputs generated successfully on "+str(datetime.now())+"."
if saveTraining:
	summary += "\n\nTraining exported to \""+savefilename+".xml\".\nTraining variables:\n"+str(saveSetnames)+"\nreplaced with:\n"+str(saveSetf)
print "\n"+summary
outfile=open("info.txt","w")
outfile.write(summary)
outfile.close()

printlog("Completed.")
