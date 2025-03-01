HybridHands - Generating hand pose estimation data using the NIMBLE model and blenderproc pipeline
========

assets available under https://drive.google.com/drive/folders/12yMqqaYVs83P4lpWkjuobIjd--9erchv?usp=drive_link

This project represents HybridHands, a pipeline designed for generating 2D training images for hand pose estimation. The pipeline utilizes both the NIMBLE and MANO hand models to create a scalable dataset. Alongside the images, it generates .json annotations that are fully compatible with the FreiHAND dataset format.

Anatomically accurate random generated hand poses are rendered through BlenderProc, which then introduces variability by applying random camera angles, orientations, lighting setups, and background generation (WIP). This ensures the creation of a robust and diverse dataset for training hand pose estimation models.

# Hand model based on Nimble, as well as MANO

![](https://liyuwei.cc/proj/img/nimble_teaser.jpg)

We present NIMBLE, a non-rigid parametric hand model that includes bones and muscles, bringing 3D hand model to a new level of realism. By enforcing the inner bones and muscles to match anatomic and kinematic rules, NIMBLE can animate 3D hands to new poses at unprecedented realism.

To learn about NIMBLE, please visit our website: https://liyuwei.cc/proj/nimble

You can find the NIMBLE paper at: https://arxiv.org/abs/2202.04533


---
## Requirements:
1. numpy
2. Pytorch
3. Pytorch3d
4. opencv-python
5. trimesh (with pyembree for faster collision detection)


## Getting Started:

1. Download model files from [here](https://drive.google.com/drive/folders/1g7DWuDW5nYI2VDbdemDK2dGVwVHV2a1X?usp=sharing).
2. `python demo.py` 
   - ***_bone.obj**: bone mesh
   - ***_joint.xyz**: joints (order see [here](https://github.com/reyuwei/PIANO_mri_data/raw/master/piano_joint_id.png))
   - ***_muscle.obj**: muscle mesh
   - ***_skin.obj/mtl**: skin mesh with diffuse texture
   - ***_diffuse/normal/spec.png**: texture images
   - ***_manov.xyz**: corresponding skin vertices in [MANO](http://mano.is.tue.mpg.de/) topology
3. Render in `assets/render.blend` 
   - Scale down the meshes and manually assign texture images.
   - Select `non-color` for color space of normal and specular textures. 
   - Change mesh to `shade smooth`.
   - Adjust normal map strength.

## Notes:
1. Use `handle_collision=True` to make sure skin is always outside muscle. Otherwise, skin and muscle might collide. 
2. Joint order see [PIANO Dataset](https://github.com/reyuwei/PIANO_mri_data?tab=readme-ov-file#joint-id).

## Acknowledgements

This model and code was developped and used for the paper *NIMBLE: A Non-rigid Hand Model with Bones and Muscles* for SIGGRAPH22.
See [project page](https://liyuwei.cc/proj/nimble).

It reuses part of the great code from 
[manopth](https://github.com/hassony2/manopth/blob/master/manopth) by [Yana Hasson](https://hassony2.github.io/),
[pytorch_HMR](https://github.com/MandyMo/pytorch_HMR) by [Zhang Xiong](https://github.com/MandyMo) and
[SMPLX](https://github.com/vchoutas/smplx) by [Vassilis Choutas](https://github.com/vchoutas)!


If you find this code useful for your research, consider citing:

```
@article{li2022nimble,
        author = {Li, Yuwei and Zhang, Longwen and Qiu, Zesong and Jiang, 
            Yingwenqi and Li, Nianyi and Ma, Yuexin and Zhang, Yuyao and
            Xu, Lan and Yu, Jingyi},
        title = {NIMBLE: A Non-Rigid Hand Model with Bones and Muscles},
        year = {2022},
        issue_date = {July 2022},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        volume = {41},
        number = {4},
        issn = {0730-0301},
        url = {https://doi.org/10.1145/3528223.3530079},
        doi = {10.1145/3528223.3530079},
        journal = {ACM Trans. Graph.},
        month = {jul},
        articleno = {120},
        numpages = {16}
        }
```
