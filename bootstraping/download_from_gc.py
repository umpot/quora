import sys
import subprocess
import os

def run(instance_name, zone, folder, files, destination='.'):
    for f in files:
        f=os.path.join(folder, f)# folder+f
        f='{}:{}'.format(instance_name, f)
        print f
        #gcloud compute copy-files --zone us-central1-c kg2:/home/dd_petrovskiy/kg/data/distances/test_metrics_fuzzy_lemmas.csv .
        subprocess.call(['gcloud', 'compute', 'copy-files', '--zone', zone, f, destination])


# instance_name = sys.argv[1]
# zone=sys.argv[2]
# folder=sys.argv[3]
# files=sys.argv[4:]
#
# run(instance_name, zone, folder, files)



instance_name = 'instance-1'
zone='us-west1-b'
folder='/home/dd_petrovskiy/kg/data/top_k_freq'




files = [
    'folds.json',  'out_of_fold_contains.json',  'out_of_fold_freq.json',  'tokens.json',  'train_freq.json'
]

destination='/home/dpetrovskyi/PycharmProjects/data/top_k_freq'

run(instance_name, zone, folder, files, destination)