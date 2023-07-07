import sys
import os
import subprocess
import math

inputs = ""
chunk_path = sys.argv[1]
num_chunks = int(sys.argv[2])
num_wide = int(sys.argv[3])
num_tall = int(sys.argv[4])
seq_len = int(sys.argv[5])
video_seconds = int(sys.argv[6])
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
    
    with open("/tmp/mosaic/montage_inputs", "w") as montage_file:
        for i in range(num_tiles):
            chunk_idx = i % num_chunks
            input_path = f"{chunk_path}/{chunk_idx}/frame{seq_frame}.bmp"

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

    subprocess.run(f"montage @/tmp/mosaic/montage_inputs -geometry {width}x -tile {num_wide_filtered}x{num_tall_filtered} - | magick - -gravity center -crop 1920x1080+0+0 /tmp/mosaic/merged_{frame}.bmp ", shell=True)
