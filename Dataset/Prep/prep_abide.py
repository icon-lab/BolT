import os

import nilearn as nil
import nilearn.datasets
import nilearn.image
from glob import glob
import pandas
import torch
from tqdm import tqdm

from .prep_atlas import prep_atlas
from nilearn.input_data import NiftiLabelsMasker


datadir = "./Dataset/Data"


def prep_abide(atlas):

    bulkDataDir = "{}/Bulk/ABIDE".format(datadir)

    atlasImage = prep_atlas(atlas)



    if(not os.path.exists(bulkDataDir)):
        nil.datasets.fetch_abide_pcp(data_dir=bulkDataDir, pipeline="cpac", band_pass_filtering=False, global_signal_regression=False, derivatives="func_preproc", quality_checked=True)

    dataset = []


    temp = pandas.read_csv(bulkDataDir + "/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv").to_numpy()
    phenoInfos = {}
    for row in temp:
        phenoInfos[str(row[2])] = {"site": row[5], "age" : row[9], "disease" : row[7], "gender" : row[10]}

    print("\n\nExtracting ROIS...\n\n")

    for scanImage_fileName in tqdm(glob(bulkDataDir+"/ABIDE_pcp/cpac/nofilt_noglobal/*"), ncols=60):
        
        if(".gz" in scanImage_fileName):

            scanImage = nil.image.load_img(scanImage_fileName)
            roiTimeseries =  NiftiLabelsMasker(atlasImage).fit_transform(scanImage)

            subjectId = scanImage_fileName.split("_")[-3][2:]
            
            dataset.append({
                "roiTimeseries": roiTimeseries,
                "pheno": {
                    "subjectId" : subjectId, **phenoInfos[subjectId]
                }
            })

    torch.save(dataset, datadir + "/dataset_abide_{}.save".format(atlas) )

    
