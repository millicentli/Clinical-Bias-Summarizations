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
                    text = f.read()
                    if 'FINDINGS' in text and 'IMPRESSION' in text:
                        i = s.rfind("/")
                        d = os.path.join(dst, s[i + 1:])
                        if not os.path.exists(dst):
                            os.makedirs(dst)
                        shutil.copy2(s, d)

# Check the files in the same folder and paste things into a single file
def consolidate(src, dst):
    for item in os.listdir(src):
        # Get the path for the file
        s = os.path.join(src, item)
        with open(s, 'r') as f:
            with open(dst, 'a') as f1:
                doc = " "
                summary = " "
                check_sum = False
                for line in f:
                    stripped = line.strip()
                    if len(stripped) == 0:
                        continue
                    if "IMPRESSION" in line:
                        check_sum = True
                    if check_sum:
                        summary = summary + stripped + " "
                    else:
                        doc = doc + stripped + " "

                f1.write(doc + "\t")
                f1.write(summary)
                f1.write("\n")

def clean_cxr(filename):
    # Read from the file
    doc = []
    summary = []
    with open(filename, 'r') as f:
        for line in f:
            print("line pre-split:", line)
            split = line.split('\t')
            print("here's split:", split)
            doc.append(split[0])
            summary.append(split[1])
            print("line 1:", split[0])
            print("line 2:", split[1])
            exit()
    
if __name__ == "__main__":
    # testing - no other purpose

    # start directory crawling for MIMIC-CXR, putting the text files in one folder
    if args.crawl:
        dst = "mimic-cxr/files/filtered/"

        for i in range(10, 20):
            s = "p" + str(i)
            print("directory: " +  s)
            directory_crawl("/data2/limill01/Clinical-Bias-Summarizations/mimic-cxr/files/" + s, dst)
    # clean the data and put into a single file
    else:
        dst = "mimic-cxr/files/data.txt"
        src = "mimic-cxr/files/filtered/"
        # consolidate(src, dst)
        #clean_cxr(dst)

