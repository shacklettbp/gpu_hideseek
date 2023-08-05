Hide & Seek GPU Batch Simulator
===============================

This repository is a high-performance GPU batch simulator implementation of OpenAI's Hide and Seek Environment from the paper [Emergent Tool Use From Multi-Agent Autocurricula](https://openai.com/research/emergent-tool-use) (Baker et al. 2020). The batch simulator is built on the [Madrona Engine](https://madrona-engine.github.io), and is one of the primary examples used for performance analysis in the SIGGRAPH 2023 paper about the engine.

**WARNING** This repository is not currently under development and does not represent best practices for using the Madrona Engine.The codebase is provided here primarily for the purposes of reproducing performance results from our SIGGRAPH paper. For a clean, well documented, 3D environment written in Madrona with similar functionality, refer to the [Madrona Escape Room](https://github.com/shacklettbp/madrona_escape_room) example project.

Build and Profile
==============
First, make sure you have all the dependencies listed [here](https://github.com/shacklettbp/madrona#dependencies) (briefly, recent python and cmake, as well as Xcode or Visual Studio on MacOS or Windows respectively).

Next, fetch the repo (don't forget `--recursive`!):
```bash
git clone --recursive https://github.com/shacklettbp/gpu_hideseek.git
cd gpu_hideseek
```

Next, for Linux and MacOS: Run `cmake` and then `make` to build the simulator:
```bash
mkdir build
cd build
cmake ..
make -j # cores to build with
cd ..
```

Or on Windows, open the cloned repository in Visual Studio and build
the project using the integrated `cmake` functionality.

Now, setup the python components of the repository with `pip`:
```bash
pip install -e . # Add -Cpackages.gpu_hideseek.ext-out-dir=PATH_TO_YOUR_BUILD_DIR on Windows
```

You can profile the simulator as follows (first, [install pytorch](https://pytorch.org/get-started/locally/)):
```bash
python scripts/benchmark.py 16000 1920 0 0 1    # Benchmark 16K worlds on the GPU backend
python scripts/cpu_benchmark.py 2000 1920 0 0 1 # Benchmark 2K worlds on the CPU backend
```

Performance numbers for both backends are expected to be slightly faster than results published in the paper due to engine-level improvements.
