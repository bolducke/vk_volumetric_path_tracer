# Vulkan Volumetric Path Tracer

I decided to give Vulkan another shot, so I stumbled upon this excellent recommendation online. It led me to this cool [resource](https://nvpro-samples.github.io/vk_mini_path_tracer/index.html), which gives an overview of Vulkan. It's a really great resource that I warmly recommend for beginners and path-tracing enthusiasts.

## Briefly
* Loaded custom extensions like **VK_KHR_ray_query** and **VK_KHR_acceleration_structure**
* Allocated buffers to be used in our shader
* Loaded a custom model
* Used **BLAS** and **TLAS** for our custom model
* Created a compute pipeline 
* Synchronized properly the pipeline and submit it through the command buffer using a pipeline barrier.
* Wrote back the data in a file

## Contribution
* Created a **blender** scene with a way to **add custom attributes** and **save them** into a '.npy'.
* Added a way to **load material** information to the shader stage.
* Added different **materials** such as light, glossy reflection, and more.
* Implemented a **volumetric** path tracer based on my previous course using 
    * **Free-path Sampling**
    * **Henyey-Greenstein Phase Function**
    * **Spectral Sampling**

With 2056 samples and a depth of 32, I could render the volumetric path tracing in **less than 1 second** using an RTX 4070 Laptop.

## Gallerie

| ![image](https://github.com/bolducke/vk_volumetric_path_tracer/assets/26026137/4595f9f3-64ec-4d67-9960-5ed19063e02e) | ![image](https://github.com/bolducke/vk_volumetric_path_tracer/assets/26026137/39a61935-7793-445d-9c57-24f7823a3193) |
|---|---|
| ![image](https://github.com/bolducke/vk_volumetric_path_tracer/assets/26026137/a66b8242-ad0e-488f-b0c6-b362136bd93c) | ![image](https://github.com/bolducke/vk_volumetric_path_tracer/assets/26026137/20f2c62c-33d9-4b39-9157-34bc38a63523) |
