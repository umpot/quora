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



instance_name = 'kg1'
zone='us-west1-b'
folder='/home/dd_petrovskiy/kg/quora/src/glove'





destination='.'

files = [
    'nohup.out'
]
run(instance_name, zone, folder, files, destination)
