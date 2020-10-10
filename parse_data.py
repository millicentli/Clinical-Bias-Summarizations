# For parsing data for the two datasets (MIMIC-III, MIMIC-CXR)

import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--crawl', dest='crawl', action='store_true', help='crawls the original MIMIC-CXR dataset')
parser.add_argument('--nocrawl', dest='crawl', action='store_false', help='do not crawl the original MIMIC-CXR dataset')

parser.set_defaults(crawl=False)
args = parser.parse_args()

# Crawls through the directories, filtering and putting the files in the filtered directory 
def directory_crawl(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        if os.path.isdir(s):
            directory_crawl(s, dst, symlinks, ignore)
        else:
            if 'txt' in s:
                with open(s, 'r') as f:
                    if 'IMPRESSION' and 'FINDINGS' in f.read():
                        f = s.rfind("/")
                        d = os.path.join(dst, s[f + 1:])
                        if not os.path.exists(dst):
                            os.makedirs(dst)
                        shutil.copy2(s, d)

if __name__ == "__main__":

    # start directory crawling for MIMIC-CXR, putting the text files in one folder
    if args.crawl:
        dst = "mimic-cxr/files/filtered/"

        for i in range(10, 20):
            s = "p" + str(i)
            print("directory: " +  s)
            directory_crawl("/data2/limill01/Clinical-Bias-Summarizations/mimic-cxr/files/" + s, dst)
    else:
        raise NotImplemented
        
