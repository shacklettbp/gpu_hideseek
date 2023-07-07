import sys
import os
import subprocess
import math
from pathlib import Path

inputs = ""
dump_path = sys.argv[1]
num_wide = int(sys.argv[2])
num_tall = int(sys.argv[3])
seq_len = int(sys.argv[4])
video_seconds = int(sys.argv[5])
num_frames = video_seconds * 30

num_tiles = num_wide * num_tall

os.makedirs("/tmp/mosaic/", exist_ok=True)

final_width = 1920 / num_wide
final_height = 1080 / num_tall

def easeOutCirc(x):
    return math.sqrt(1 - math.pow(x - 1, 2))

def easeInOutCubic(x):
    if x < 0.5:
        return 4 * x * x * x
    else:
        return 1 - math.pow(-2 * x + 2, 3) / 2

dir_1920 = dump_path + "/1920/"
dir_960 = dump_path + "/960/"
dir_480 = dump_path + "/480/"
dir_240 = dump_path + "/240/"

total_num_imgs = sum(1 for _ in Path(dir_1920).glob("*bmp"))
num_seqs = total_num_imgs // seq_len

for frame in range(num_frames):
    seq_frame = frame % seq_len
    t = frame / (num_frames - 1)
    t = easeInOutCubic(t)

    t = 1 - t

    width = (1920 - (1920 - final_width) * t)
    height = width * 9/16

    min_col = 10000
    max_col = -10000

    min_row = 10000
    max_row = -10000

    if width > 960:
        size_dir = dir_1920
        size_class = 1920
    elif width > 480:
        size_dir = dir_960
        size_class = 960
    elif width > 240:
        size_dir = dir_480
        size_class = 480
    else:
        size_dir = dir_240
        size_class = 240
    
    with open("/tmp/mosaic/montage_inputs.txt", "w") as montage_file:
        for i in range(num_tiles):
            seq_idx = i % num_seqs
            input_path = f"file '{size_dir}/frame{seq_idx * seq_len + seq_frame + 1}.bmp'"

            row = i // num_wide - num_tall // 2
            col = i % num_wide - num_wide // 2

            if (col != 0 and (abs(col) - 1) * width + width / 2 >= 960):
                continue

            if (row != 0 and (abs(row) - 1) * height + height / 2 >= 540):
                continue

            if col < min_col:
                min_col = col

            if col > max_col:
                max_col = col

            if row < min_row:
                min_row = row

            if row > max_row:
                max_row = row

            montage_file.write(f"{input_path}\n")

        num_wide_filtered = max_col - min_col + 1
        num_tall_filtered = max_row - min_row + 1

        print(f"{num_wide_filtered}x{num_tall_filtered}")

    print(width)

    input_width = num_wide_filtered * size_class
    prescale = 1
    while input_width / prescale > 8192:
        prescale *= 2

    if prescale != 1:
        prescale_filter = f"scale={size_class / prescale}:-1,"
        print(prescale_filter)
    else:
        prescale_filter = ''

    subprocess.run(f"ffmpeg -y -f concat -safe 0 -i /tmp/mosaic/montage_inputs.txt -vf '{prescale_filter}tile={num_wide_filtered}x{num_tall_filtered},scale={width * num_wide_filtered}:-1' -vframes 1 /tmp/mosaic/mosaic.bmp", shell=True)
    subprocess.run(f"magick /tmp/mosaic/mosaic.bmp -gravity center -crop 1920x1080+0+0 /tmp/mosaic/merged_{frame}.bmp", shell=True)

    #subprocess.run(f"montage -limit memory 8589934592 @/tmp/mosaic/montage_inputs.txt -geometry {width}x -tile {num_wide_filtered}x{num_tall_filtered} - | magick - -gravity center -crop 1920x1080+0+0 /tmp/mosaic/merged_{frame}.bmp ", shell=True)
