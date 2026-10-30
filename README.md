# pv-ollama

Ollama for the Pantavisor AI Box on the Jetson Nano (Tegra X1, sm_53),
packaged as a Pantavisor app container.

- Ollama b9f74ff (v0.1.33 era), the last that builds against the Nano's
  CUDA 10.2 and GCC 8 (ollama/ollama#10090), with its CUDA runner on the GPU;
  `patches/` adds the GCC 8 NEON loads llama.cpp expects from GCC 9.
- Carries libcuda.so.1 and the Tegra libraries it needs from L4T R32.7.6,
  the release of the BSP's nvgpu.
- Serves the Ollama API on the device's port 11434, open to the Hollama UI
  on port 4173 (`OLLAMA_ORIGINS=*`). Models live in the persistent /models.

Pantavisor gives the container the GPU device nodes of
`pvpkg/generic/ollama/mdev.json` and loads the nvidia driver
(`PV_DRIVERS_REQUIRED` in `pvpkg.json`). Build it with the stage builder:

	pvr -b https://api.stage.pantahub.com build start --project nano-ai-box \
	  --platform linux/arm64 --limit-seconds 14400 \
	  --git-url https://github.com/pantavisor/pv-ollama --git-ref main
