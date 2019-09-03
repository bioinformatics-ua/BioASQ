import json
import os
import sys
import gc

dir_path = "/backup/pubmed_archive_json/untar"

l=sorted(os.listdir(dir_path))

f=open(os.path.join(dir_path,l[0]),"r")

j=json.load(f)
print("LOAD DONE")
f.close()

input()
