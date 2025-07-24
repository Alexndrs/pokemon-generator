# A modest explaination of the DDPM [1] process

## 1. Context

Using the modelisation of [1] we define a noising of an initial image $x_0$ from step $0$ to step $T$ with de equation : 
$$q(x_t | x_{t-1}) := \mathcal{N}(x_{t};\sqrt{1-\beta_t} x_{t-1}, \beta_t I) $$
where $\beta_t$ is our noise scheduler. With $\alpha_t = 1-\beta_t$ and $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$ we have directly :
$$ q(x_t | x_0) =\mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$$

The key idea of the denoising diffusion model is to invert this process by estimating an image with less noise from the more noised image : i.e : trying to recover $x_{t-1}$ from $x_{t}$ : 
$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_t; \mu_\theta(x_t, t), \Sigma_\theta(x_t,t))$$
If we manage to compute $\mu_\theta$ and $\Sigma_\theta$ we will be able to go from a noisy image $x_T$ to a denoised clean image $x_0$

## 2. Estimation of $\mu_\theta$ and $\Sigma_\theta$
To estimate $\mu_\theta$ and $\Sigma_\theta$ (i.e : to find $p_\theta(x_{t-1}|x_t)$), DDPM proposes to calculate $q(x_{t-1}|x_t, x_{0,pred})$ where $x_{0,pred}$ is an estimation of the denoised image that is obtained thanks to a machine learning model : the model usually used is a *U-NET* neural network [2].

After some fastidious calculus using bayes and common probabilistic technics we can show that :

$$q(x_{t-1}|x_t, x_{0,pred}) = \mathcal{N}(x_{t-1}; c_t^1 x_{0,pred} + c_t^2 x_t, c_t^3 I) $$
where : $c_t^1 = \frac{\beta_t\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_t}$ ; $c_t^1 = \frac{(1-\bar{\alpha}_{t-1})\sqrt{\alpha_t}}{1-\bar{\alpha}_t}$ ; $c_t^3=\frac{(1-\bar{\alpha}_{t-1})\beta_t}{1-\bar{\alpha}_t}$
and we use a neural network to predict the noise from an image. 
i.e: if $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ with $\epsilon\in\mathcal{N}(0,I)$ then the neural network will predict $\epsilon_\theta$ and we can easily get $x_{0,pred}$ from $\epsilon_\theta$ and $x_t$ with
$$x_{0,pred} = \frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta}{\sqrt{\bar{\alpha}_t}}$$

Of course, to have the best prediction of denoised image $x_{0,pred}$ we need to get the best prediction of noise $\epsilon_\theta$. This give us the intuitive loss that minimize the predicted noise from an image that we noised with $\epsilon$ :

$$\min \mathbb{E}_{x_0\in X, \epsilon\in\mathcal{N}(0,I)}[(\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon))^2]$$

## 3. Implementation in my code

```
class GaussianDiffusion:
```
`__init__` : this function create a scheduler : I used the cosine scheduler because it's the state of art scheduler for low resolution image resolution. All the constants necessary for $c_t^1 ,c_t^2, c_t^3$ are precomputed and vectorized.

`q_sample` : implement the forward diffusion (the noising phase)

`predict_x0_from_eps` : get $x_{0,pred}$ from $\epsilon_\theta$

`p_sample` : *core* of the diffusion : calculate $p_\theta(x_{t-1}|x_t)$ using posterior $q(x_{t-1}|x_t, x_{0,pred})$ as seen above

`p_sample_loop` : iterate p_sample `T`-times

`sample` : alias for `p_sample_loop` with different API

`loss` : compute the MSE loss of a batch as seen above


___
**Bibliography**


[1] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. Advances in neural information processing systems, 33, 6840-6851.

[2] Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Cham: Springer international publishing.

[3] Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E. L., ... & Norouzi, M. (2022). Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35, 36479-36494.