import sys
import os

cmd_prefix = "ffmpeg"
cmd_postfix = "-r 30 -pix_fmt yuv420p -vcodec libx264 out.mp4"
#crop="crop=1920:1080:iw/2 - iw/4:ih/2 - ih/4"

inputs = ""

chunk_path = sys.argv[1]
num_chunks = int(sys.argv[2])
num_tiles = int(sys.argv[3])
num_wide = int(sys.argv[4])
num_tall = int(sys.argv[5])
video_seconds = int(sys.argv[6])
num_loops = video_seconds // 2 - 1

assert(num_wide * num_tall == num_tiles)

scale_b = 1080 * num_tall
scale_a = scale_b / video_seconds

scale_fn = "max(trunc({scale_b} - {scale_a} * t), 1080)"

scale = f"scale=-1:'{scale_fn}':eval='frame'"
#crop=f"crop=1920:1080:'trunc({crop_w_b} - 1.25 * {crop_w_a} * t)':'trunc({crop_h_b} - 1.25 * {crop_h_a} * t)'"
crop=f"crop=1920:1080:'({scale_fn}*1920/1080-1920)/2':'(scale_fn-1080)/2'"

input_desc = ""
xstack_prefix = f"xstack=inputs={num_chunks}:layout="
xstack_positions = []

subchunk_w = 5
subchunk_h = 5
subchunk_size = subchunk_w * subchunk_h
assert(num_wide % subchunk_w == 0)
assert(num_tall % subchunk_h == 0)
num_subchunks_wide = num_wide // subchunk_w
num_subchunks_tall = num_tall // subchunk_h

num_subchunks = num_tiles // subchunk_size

for i in range(num_tiles):
    chunk_idx = i % num_chunks
    inputs += f" -stream_loop {num_loops} -framerate 30 -i {chunk_path}/{chunk_idx}/frame%d.bmp"


def build_xstack_layout(i, hstack):
    col = i % hstack 
    row = i // hstack

    if col == 0:
        col_desc = "0"
    else:
        cols = []
        for i in range(col):
            cols.append(f"w{i + row * hstack}")
        col_desc = "+".join(cols)

    if row == 0:
        row_desc = "0"
    else:
        rows = []
        for i in range(row):
            rows.append(f"h{i * hstack}")
        row_desc = "+".join(rows)

    return f"{col_desc}_{row_desc}"

subchunk_cmds = ""
for subchunk_idx in range(num_subchunks):
    subchunk_inputs = ""

    subchunk_layouts = []
    for i in range(subchunk_size):
        input_idx = subchunk_idx * subchunk_size + i 

        subchunk_inputs += f"[{input_idx}:v]"

        subchunk_layouts.append(build_xstack_layout(i, subchunk_w))

    subchunk_cmds += f"{subchunk_inputs}xstack=inputs={subchunk_size}:layout={'|'.join(subchunk_layouts)}[subchunk_{subchunk_idx}];"

full_inputs = ""
full_layouts = []
for subchunk_idx in range(num_subchunks):
    full_inputs += f"[subchunk_{subchunk_idx}]"
    full_layouts.append(build_xstack_layout(subchunk_idx, num_subchunks_wide))

full_xstack_cmd = f"{full_inputs}xstack=inputs={num_subchunks}:layout={'|'.join(full_layouts)}[mosaic]"

final_cmd = f"{cmd_prefix}{inputs} -filter_complex \"{subchunk_cmds}{full_xstack_cmd};[mosaic]{scale}[scale];[scale]{crop}\" {cmd_postfix}"
print(final_cmd)

os.system(final_cmd)
