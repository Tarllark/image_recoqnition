###############################################
#
# Title:	webget.py
#
# Descr:	Download file from URL
#
# Author:	Tarllark
#
# Team:		Successful Story
#
###############################################

import sys
import os
import urllib.request
from urllib.parse import urlparse
import shutil

def download(url, to=None):
	if to == None:
		file_name = urlparse(url).path.split('/')[-1]
	else:
		file_name = to + urlparse(url).path.split('/')[-1]
	with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
		shutil.copyfileobj(response, out_file)
	return file_name
