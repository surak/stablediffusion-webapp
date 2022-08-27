# stablediffusion-webapp
The minimal app for stable diffusions, using gradio
<img alt="stablediffusion-demo" src="https://user-images.githubusercontent.com/878399/186395603-e34d925c-10ca-4236-8fa0-7511a777997e.png">

This is a shameless clone of https://huggingface.co/spaces/stabilityai/stable-diffusion ,  which works with the downloaded model.

How to install the whole thing:


- Accept the agreement here: https://huggingface.co/CompVis/stable-diffusion-v-1-4-original
- Create a conda virtual environment, activate it and start installing stuff.

First, you probably need a gpu-enabled working version of PyTorch. For now, this means an NVIDIA GPU and CUDA. Do something like this:

```bash
mkdir ~/gpu-stuff
curl -o ~/gpu-stuff https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
cd ~/gpu-stuff 
./Miniconda3-latest-Linux-x86_64.sh -b -f -p ~/gpu-stuff/miniconda3
conda install -y mamba -n base -c conda-forge
mamba install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 python-dotenv flask Pillow -y
pip install --upgrade pip
pip install image_to_numpy opencv-contrib-python
pip install huggingface_hub
pip install --upgrade diffusers transformers scipy gradio
huggingface-cli login
```

- Get a token from huggingface. The login thing above should help.
- Add the token to the line which says `YOUR_TOKEN=""` on the app.py
- Run it: `python app.py`
- Open `http://localhost:1234` on your browser.
