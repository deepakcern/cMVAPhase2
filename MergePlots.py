# Script to combine selective/multiple plots obtained as outputs from cMVAv2 Phase-II retraining script.
#
# Author: Spandan Mondal
# Email: spandan.mondal@cern.ch

paths=[]
paths.append("output/91XSamples/Run12AllDisc/")
paths.append("output/91XSamples/Run13ProperOldcMVA/")
outdir="output/91XSamples/Merge1213_2/"    #<-- Remember to change this to prevent overwriting!

plotstokeep=[[1,6],[2]]
addtitle=[["","","",""],["","","",""]]

for i in range(len(plotstokeep)):
	for j in range(len(addtitle[i]),max(plotstokeep[i])):
		addtitle[i].append("")

import os, sys, matplotlib.pyplot as plt

colorlist=["red","black","orange","blue","green","magenta","yellow","cyan"]

files=[[] for i in paths]

for i in range(len(paths)):
	for file in os.listdir(paths[i]):
		if file.endswith(".pltdata"):
			files[i].append(file)

for i in files:   
	if files[0]!=i:
		print "Filelist mismatch."
		sys.exit()
	
	
for filename in files[0]:
	openfiles=[open(i+filename,"r") for i in paths]
	colorcount=0
	axes=[]
	
	plt.figure(figsize=(10,10))	
	
	for fileno in range(len(openfiles)):
		pltno=1
		for line in openfiles[fileno]:
			if line.startswith("#Title="):
				plttitle=str(line[7:-1])
			elif line=="\n" or line.startswith("#"):
				continue
			elif line.startswith("%label="):
				pltlabel=str(line[7:-1])
			elif line.startswith("%color="):
				continue
			else:
				axes.append([float(x) for x in line.split()])
			
			if len(axes)==4:
				if pltno in plotstokeep[fileno]:
					pltlabel = addtitle[fileno][pltno-1]+" "+pltlabel
					###If any plot's label has to be changed
#					if "r/w" in pltlabel:
#						pltlabel=" cMVAv2 (Retrained for Phase-2)"+pltlabel[-18:-1]+")"
#					###
					plt.plot(axes[0],axes[1],color=colorlist[colorcount],label=pltlabel,linewidth=1.5)
					plt.plot(axes[2],axes[3],color=colorlist[colorcount],linewidth=1.5)
					colorcount += 1			
				axes=[]
				pltno += 1
	

	plt.yscale("log")
	plt.xlim(0.2,1.0)
	plt.ylim(0.001, 1.0)
	plt.legend(loc=2, fancybox=True, framealpha=0.0, fontsize=10)
	plt.xlabel("b-tagging efficiency", fontsize=16)
	plt.ylabel("mistag probability", fontsize=16)
	plt.title(plttitle)
	outfile=outdir+filename[:-8]+".png"
	plt.savefig(outfile)
	plt.clf()
	
	print "Exported %s." %outfile
	
	for fl in openfiles:
		fl.close()

	
