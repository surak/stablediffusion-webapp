# stablediffusion-webapp
The minimal app for stable diffusions, using gradio
<img alt="stablediffusion-demo" src="https://user-images.githubusercontent.com/878399/186395603-e34d925c-10ca-4236-8fa0-7511a777997e.png">

This is a shameless clone of https://huggingface.co/spaces/stabilityai/stable-diffusion ,  which works with the downloaded model.

How to install the whole thing:

- Accept the agreement here: https://huggingface.co/CompVis/stable-diffusion-v-1-4-original
- Create a python virtual environment, activate it and start installing stuff: 

```bash
pip install huggingface_hub
pip install --upgrade diffusers transformers scipy gradio
huggingface-cli login
```

- Get a token from huggingface. The login thing above should help.
- Add the token to the line which says `YOUR_TOKEN=""` on the app.py
- Run it: `python app.py`
- Open `http://localhost:1234` on your browser.
