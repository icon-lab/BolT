import cv2

import numpy as np

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, default="HCP_TASK")
parser.add_argument("-c", "--classs", type=str, default="6")

argv = parser.parse_args()


if(argv.dataset == "HCP_TASK"):
    sourceDir = "fullPathToTheWorkingFolder/Analysis/BrainMapping/Viz/{}/{}/".format(argv.dataset, argv.classs)
else:
    sourceDir = "fullPathToTheWorkingFolder/Analysis/BrainMapping/Viz/{}/".format(argv.dataset)

print(sourceDir)

optionY = 300
optionX = 0
axial = cv2.resize(cv2.imread(sourceDir + "axial.png")[150+optionX:-150-optionX, 300+optionY:-300-optionY], dsize=(0,0), fx=1.6, fy=1.6)
sagittal_lh_lt = cv2.imread(sourceDir + "LH_sagittal_lt.png")[200+optionX:-200-optionX, 200+optionY:-200-optionY]
sagittal_rh_lt = cv2.imread(sourceDir + "RH_sagittal_lt.png")[200+optionX:-200-optionX, 200+optionY:-200-optionY]
sagittal_lh_rt = cv2.imread(sourceDir + "LH_sagittal_rt.png")[200+optionX:-200-optionX, 200+optionY:-200-optionY]
sagittal_rh_rt = cv2.imread(sourceDir + "RH_sagittal_rt.png")[200+optionX:-200-optionX, 200+optionY:-200-optionY]
sagittal_lt = np.concatenate([sagittal_lh_lt, sagittal_rh_lt])
sagittal_rt = np.concatenate([sagittal_rh_rt, sagittal_lh_rt])
axial = np.concatenate([axial, np.zeros_like(axial)[:len(sagittal_lt)-len(axial)]])

network_figure = np.concatenate([sagittal_lt, axial, sagittal_rt], axis=1)
cv2.imwrite(sourceDir + "full.png", network_figure)

