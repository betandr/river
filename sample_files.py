import os
import shutil
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, required=True, help='input directory')
parser.add_argument('--stride', type=int, required=True, help='distance between samples')
parser.add_argument('--dest', type=str, required=True, help='output directory')
args = parser.parse_args()

in_dir = args.src
stride = args.stride
out_dir = args.dest

# If stride is 3 and you have 4 files, file 3 will be moved leaving files 1, 2, and 4.
# If stride is 3 you have 6 files files 3 and 6 will be moved leaving files 1, 2, 4, and 5.

src = os.path.dirname(in_dir)
dest = os.path.dirname(out_dir)

if not os.path.exists(dest):
    os.makedirs(dest)

i=0
for basename in os.listdir(src):
    if basename.endswith('.wav'):
        pathname = os.path.join(src, basename)
        if os.path.isfile(pathname):
            if (i == 0):
              shutil.move(pathname, dest)
              # print("moving " + pathname + " to " + dest)
            i = (i + 1) % stride