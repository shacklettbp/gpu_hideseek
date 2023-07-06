import os
import sys
from pathlib import Path

num_bmps = sum(1 for _ in Path(sys.argv[1]).glob("*bmp"))
print(num_bmps);

for i in range(0, num_bmps):
    fname = f"{sys.argv[1]}/frame{i + 1}.bmp"

    step = i % 60
    episode = i // 60

    episode_dir = f"{sys.argv[2]}/{episode}"
    os.makedirs(episode_dir, exist_ok = True)

    new_path = f"{episode_dir}/frame{step}.bmp"
    print(episode, step, fname, new_path)

    os.rename(fname, new_path)
