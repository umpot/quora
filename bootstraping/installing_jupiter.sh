#!/usr/bin/env bash

#on remote and local
pip install jupyter
pip install IPython

#optionaly nohup xxx &
jupyter notebook --ip=0.0.0.0 --port=9000 --no-browser

#configure pycharm:
#1)File>Settings>Language and Frameworks>Jupyter Notebook ==> specify url +folder(???)
#2)create new notebook, run cell, enter token