#perhaps we can import the various AI models here. Im still very confused about the whole AWS thing so uh...
import json
import os
import tarfile
  
def uncompress(filename):
    file = tarfile.open(filename)
    file.extractall('./Destination_FolderName')
    file.close()

def read_json(filename):
    f = open(filename)
    data = json.load(f)
    lst = []
    for i in data:
        lst.append(i)
    f.close()
    os.remove(filename)
