# A pokemon images generator build from scratch in pytorch implementing DDPM

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white&style=flat-square)](https://pytorch.org)
[![Diffusion Model](https://img.shields.io/badge/Diffusion-Model-9966CC?logo=ai&logoColor=white&style=flat-square)](https://arxiv.org/abs/2006.11239)
[![LightWeight](https://img.shields.io/badge/Size-41Mo-brightgreen?logo=feather&style=flat-square)](https://github.com)
[![Pokémon](https://img.shields.io/badge/Pokémon-API-FFCB05?logo=pokemon&logoColor=white&style=flat-square)](https://pokeapi.co)


## Unconditional generation
Current results for ~1750 Epoch on 1200 images with the unconditional DDPM 

> On cosine scheduler
<p align="center" style="display: flex; justify-content: center; width:95%">
  <img src="./backend/samples/generated_samples_1700.png" width="76%" alt="Chat demo" />
  <img src="./backend/video_samples/diffusion_process_1700.gif" width="19%" alt="Chat demo" />
</p>


___


## Conditional generation with free guidance
Current results for ~345 Epoch on 5883 images with the conditionnal DDPM (conditionnal encoding of dimension 42) 

>weight : [model on google drive](https://drive.google.com/file/d/131QypAm5bKnQFkletqcY4LHWCdWo8TLS/view?usp=sharing)



> On cosine scheduler
> cond : color="red", is_sprite=False
<p align="center" style="display: flex; justify-content: center; width:95%">
  <img src="./backend/samples/generated_samples_red_official.png" width="76%" alt="Chat demo" />
  <img src="./backend/video_samples/diffusion_process_red_official.gif" width="19%" alt="Chat demo" />
</p>

> cond : color="blue", is_sprite=True
<p align="center" style="display: flex; justify-content: center; width:95%">
  <img src="./backend/samples/generated_samples_blue_sprite.png" width="76%" alt="Chat demo" />
  <img src="./backend/video_samples/diffusion_process_blue_sprite.gif" width="19%" alt="Chat demo" />
</p>

> cond : no condition
<p align="center" style="display: flex; justify-content: center; width:95%">
  <img src="./backend/samples/generated_samples.png" width="76%" alt="Chat demo" />
  <img src="./backend/video_samples/diffusion_process.gif" width="19%" alt="Chat demo" />
</p>