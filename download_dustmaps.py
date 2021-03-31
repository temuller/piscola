#!/usr/bin/env python

import wget
import tarfile
import os
import sys

def download_dustmaps(mapsdir='.'):
    """ Downloads dust maps of Schlegel, Fikbeiner & Davis (1998).

    mapsdir : str, default '.' (current directory)
        Directory where the dust maps of Schlegel, Fikbeiner & Davis (1998) are found.
    """

    sfdmaps_url = 'https://github.com/kbarbary/sfddata/archive/master.tar.gz'
    master_tar = wget.download(sfdmaps_url)

    # extract tar file under mapsdir directory
    tar = tarfile.open(master_tar)
    tar.extractall(mapsdir)
    tar.close()

    os.remove(master_tar)

if __name__ == '__main__':
    if len(sys.argv)==1:
        download_dustmaps()
    else:
        download_dustmaps(str(sys.argv[1]))
