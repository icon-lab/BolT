# TO GET DATA READY FOR THE TESTER


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, choices=["abide1", "hcpRest", "hcpTask"], default="abide1")
parser.add_argument("-a", "--atlas", type=str, choices=["schaefer7_400"], default="schaefer7_400")
argv = parser.parse_args()


from Dataset.Prep.prep_abide import prep_abide


if(argv.dataset == "abide1"):
    prep = prep_abide


prep(argv.atlas)


