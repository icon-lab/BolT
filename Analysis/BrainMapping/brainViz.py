import gl
from glob import glob
import os

VMIN = 0.01
VMAX = 3.0

targetDataset = "HCP_TASK"
targetClass = 6


colorIndexer = {
    "Vis" : 1,
    "SomMot" : 2,
    "DorsAttn" : 3,
    "SalVentAttn" : 4,
    "Limbic" : 5,
    "Cont" : 6,
    "Default" : 7
}

sourceFolder = "fullPathToTheWorkingFolder/Analysis/BrainMapping/Data/{}/Results/".format(targetDataset)


if(targetDataset == "HCP_TASK"):
    saveDir = "fullPathToTheWorkingFolder/Analysis/BrainMapping/Viz/{}/{}".format(targetDataset, targetClass)
else:
    saveDir = "fullPathToTheWorkingFolder/Analysis/BrainMapping/Viz/{}".format(targetDataset)

os.makedirs(saveDir, exist_ok=True)

def getLabelFromFileName(fileName):
    return fileName.split("/")[-1].split("_")[1]

def getPosFromFileName(fileName):
    return int(fileName.split("/")[-1].split("_")[-3]) > 200

def initialize():
    gl.resetdefaults()
    gl.loadimage("mni152")
    gl.overlayloadsmooth(True)
    gl.opacity(0,85)
    gl.colorbarposition(0)
    gl.backcolor(255,255,255)

def visualize_axial(saveDir, roiFileArray, colors):
    

    gl.viewaxial(1)
    initialize()

    
    for i, roiFile in enumerate(roiFileArray):


        color = colors[i]
        gl.overlayload(roiFile)
        gl.wait(100)
        gl.minmax(i+1, VMIN, VMAX)
        gl.wait(10)
        gl.colorname(i+1, color)
        gl.wait(10)
    
    gl.viewaxial(1)
    gl.savebmp(saveDir + "/axial.png")
    gl.overlaycloseall()



def visualize_sagittal(saveDir, leftRoiFiles, rightRoiFiles, leftColors, rightColors):

    for j, hemisphere in enumerate(["LH", "RH"]):
        initialize()

        targetRoiFiles = leftRoiFiles if hemisphere == "LH" else rightRoiFiles
        targetColors = leftColors if hemisphere == "LH" else rightColors

        for i, roiFile in enumerate(targetRoiFiles):
            color = targetColors[i]
            gl.overlayload(roiFile)
            gl.wait(100)
            gl.minmax(i+1, VMIN, VMAX)
            gl.wait(10)
            gl.colorname(i+1, color)
            gl.wait(10)        

        if hemisphere =='LH': gl.clipazimuthelevation(0.49, 90, 0)
        elif hemisphere =='RH': gl.clipazimuthelevation(0.49, 270, 0)        
                    

        gl.viewsagittal(1)
        gl.savebmp( saveDir + f'/{hemisphere}_sagittal_lt.png')
        gl.viewsagittal(0)
        gl.savebmp( saveDir + f'/{hemisphere}_sagittal_rt.png')
        gl.overlaycloseall()            



if(targetDataset == "HCP_TASK"):
    targetRoiFiles = [filee for filee in glob(sourceFolder + "/*") if "nii.gz" in filee and str(targetClass) == getLabelFromFileName(filee)]
else:
    targetRoiFiles = [filee for filee in glob(sourceFolder + "/*") if "nii.gz" in filee]



colors = []

for roiFile in targetRoiFiles:

    if(targetDataset == "HCP_REST"):

        isFemale = roiFile.split("/")[-1].split("_")[1] == "0"
        if(isFemale):
            color = "4hot"
        else:
            color = "electric_blue"
        colors.append(color)

    elif(targetDataset == "ABIDE"):

        isAsd = roiFile.split("/")[-1].split("_")[1] == "0"
        if(isAsd):
            color = "4hot"
        else:
            color = "electric_blue"
        colors.append(color)        

    elif(targetDataset == "HCP_TASK"):

        label = roiFile.split("/")[-1].split("_")[1]

        colors_ = ["1red", "2green", "3blue", "5winter", "bronze", "6bluegrn", "8redyell"]

        color = colors_[int(label)] 

        colors.append(color)


leftRoiFiles = []
leftColors = []
rightRoiFiles = []
rightColors = []

for i, filee in enumerate(targetRoiFiles):
    if(getPosFromFileName(filee)==1):
        rightRoiFiles.append(filee)
        rightColors.append(colors[i])
    else:
        leftRoiFiles.append(filee)
        leftColors.append(colors[i])

print(rightRoiFiles)


visualize_axial(saveDir, targetRoiFiles, colors)
visualize_sagittal(saveDir, leftRoiFiles, rightRoiFiles, leftColors, rightColors)

