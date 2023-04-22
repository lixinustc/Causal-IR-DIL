import os
import shutil
import glob
import random

names_hr = sorted(
            glob.glob(os.path.join('/data/libc/DF2K_crop/train', '*.png'))
        )

random.shuffle(names_hr)
# print(os.path.basename(names_hr[0]))
support = names_hr[:len(names_hr) // 2]
query = names_hr[len(names_hr) // 2:]
for name in support:
    shutil.copy(name, os.path.join('/data/libc/DF2K_crop/train_support', os.path.basename(name)))
for name in query:
    shutil.copy(name, os.path.join('/data/libc/DF2K_crop/train_query', os.path.basename(name)))