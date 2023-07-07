import sys
import os
import pathlib
import subprocess
import shutil

os.makedirs("/tmp/final", exist_ok=True)

subprocess.run("convert presentation/1-0.png -strip -depth 8 -colorspace sRGB /tmp/final/first.png",
    shell = True)

total_num_mosaic = sum(1 for _ in pathlib.Path("mosaic").glob("merged_*.bmp"))

end_start = total_num_mosaic - 60

for i in range(0, total_num_mosaic):
    if i >= end_start:
        diff = i - end_start
        if diff < 4:
            factor = diff / 4
            subprocess.run(f"convert presentation/1-1.png -alpha set -background none -channel A -evaluate multiply {factor} +channel /tmp/final/t.png", shell=True)
            overlay = "/tmp/final/t.png"
        else:
            overlay = "presentation/1-1.png"

        subprocess.run(f"convert mosaic/merged_{i}.bmp {overlay} -composite /tmp/final/mosaic_{i}.bmp",
            shell = True)
    else:
        subprocess.run(f"cp mosaic/merged_{i}.bmp /tmp/final/mosaic_{i}.bmp", shell=True)

ffmpeg_cmd = f"ffmpeg -framerate 30 -t 6.0 -i /tmp/final/first.png -framerate 30 -i /tmp/final/mosaic_%d.bmp -framerate 30 -t 2.5 -i /tmp/final/mosaic_{total_num_mosaic - 1}.bmp -r 30 -filter_complex \"[0:v]loop=180:1:0[first];[2:v]loop=73:1:0[second];[first][1:v][second]concat=n=3:v=1\" -vcodec libx264 -pix_fmt yuv420p o.mp4"


subprocess.run(ffmpeg_cmd, shell=True)
