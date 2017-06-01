import subprocess
import sys
from time import gmtime, strftime

def get_time_str():
    return strftime("%Y_%m_%d__%H_%M_%S", gmtime())

name = sys.argv[1]
script_name = sys.argv[2]
archive_name = '{}_{}.tar.gz'.format(name, get_time_str())
descr_fp = 'descr.txt'
probs_fp='probs.csv'
bucket = 'gs://ubbikk/submit_stacking/'

# tar -cvzf may_arch.tar.gz my_folder
code=subprocess.call(['mkdir', name])
print code
if code!=0:
    raise Exception('1111')

code=subprocess.call(['mv', descr_fp, name])
print code
if code!=0:
    raise Exception('2222')

code=subprocess.call(['mv', probs_fp, name])
print code
if code!=0:
    raise Exception('3333')

code=subprocess.call(['cp', script_name, name])
print code
if code!=0:
    raise Exception('4444')

code=subprocess.call(['tar', '-cvzf', archive_name, name])
print code
if code!=0:
    raise Exception('5555')

code=subprocess.call(['gsutil', 'cp', archive_name, bucket])
print code
