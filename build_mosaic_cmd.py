import sys

cmd_prefix = "ffmpeg -framerate 30"
cmd_postfix = "-pix_fmt yuv420p -vcodec libx264 out.mp4"
inputs = ""

chunk_path = sys.argv[1]
num_chunks = int(sys.argv[2])
num_wide = int(sys.argv[3])
num_tall = int(sys.argv[4])

assert(num_wide * num_tall == num_chunks)

input_desc = ""
xstack_prefix = f"xstack=inputs={num_chunks}:layout="
xstack_positions = []

for i in range(num_chunks):
    inputs += f" -i {chunk_path}/{i}/frame%d.bmp"

    input_desc += f"[{i}:v]"

    col = i % num_wide
    row = i // num_wide

    if col == 0:
        col_desc = "0"
    else:
        cols = []
        for i in range(col):
            cols.append(f"w{i + row * num_wide}")
        col_desc = "+".join(cols)

    if row == 0:
        row_desc = "0"
    else:
        rows = []
        for i in range(row):
            rows.append(f"h{i * num_wide}")
        row_desc = "+".join(rows)

    xstack_positions.append(f"{col_desc}_{row_desc}")

xstack_desc = xstack_prefix + "|".join(xstack_positions)

final_cmd = f"{cmd_prefix}{inputs} -filter_complex \"{input_desc}{xstack_desc}\" -map \"[out]\" {cmd_postfix}"
print(final_cmd)
