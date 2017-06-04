import subprocess
prefix_map = [
    'dirty_lower_no_stops',
    'dirty_upper',
    'tokens_lower',
    'tokens_lower_no_stops',
]

# proceses =[
# subprocess.Popen(['python', '-u', 'create_tfidf_features.py', 'true','false', 'dirty_lower_no_stops']),
# subprocess.Popen(['python', '-u', 'create_tfidf_features.py', 'false', 'true','dirty_lower_no_stops']),
# subprocess.Popen(['python', '-u', 'create_tfidf_features.py', 'false', 'true','dirty_upper'])]

#TRAIN
processes = [
    subprocess.Popen(['python', '-u', 'create_tfidf_features.py', 'true','false', x])  for x in prefix_map
]

#TEST
# processes = [
#     subprocess.Popen(['python', '-u', 'create_tfidf_features.py', 'false','true', x])  for x in prefix_map
#     ]

for proc in processes:
    proc.wait()
# check for results:
if any(proc.returncode != 0 for proc in processes):
    print 'Something failed'