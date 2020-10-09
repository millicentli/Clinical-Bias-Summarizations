# For parsing data for the two datasets (MIMIC-III, MIMIC-CXR)

import os
import shutil



# Crawls through the directories, putting the files in the filtered directory
def directory_crawl(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        if os.path.isdir(s):
            directory_crawl(s, dst, symlinks, ignore)
        else:
            if 'txt' in s:
                f = s.rfind("/")
                d = os.path.join(dst, s[f + 1:])
                if not os.path.exists(dst):
                    os.makedirs(dst)
                shutil.copy2(s, d)

if __name__ == "__main__":
    dst = "mimic-cxr/files/filtered/"

    # p10
    # directory_crawl("/data2/limill01/Clinical-Bias-Summarizations/mimic-cxr/files/p10", dst)
    
    for i in range(11, 20):
        s = "p" + str(i)
        print("directory:", s)
        directory_crawl("/data2/limill01/Clinical-Bias-Summarizations/mimic-cxr/files/" + s, dst)
