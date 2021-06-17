# Bidirectional Projection Network for Cross Dimension Scene Understanding

***CVPR 2021 (Oral)***

[ [Project Webpage](https://wbhu.github.io/projects/BPNet) ]    [ [arXiv](https://arxiv.org/abs/2103.14326) ]    [ [Video](https://youtu.be/Wt9J1l_UBaA) ]

Existing segmentation methods are mostly unidirectional, i.e. utilizing 3D for 2D segmentation or vice versa. Obviously 2D and 3D information can nicely complement each other in both directions, during the segmentation. This is the goal of bidirectional projection network.

![bpnet](imgs/bpnet.jpg)



## Environment


```bash
# Torch
$ pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
# MinkowskiEngine 0.4.1
$ conda install numpy openblas
$ git clone https://github.com/StanfordVL/MinkowskiEngine.git
$ cd MinkowskiEngine
$ git checkout f1a419cc5792562a06df9e1da686b7ce8f3bb5ad
$ python setup.py install
# Others
$ pip install imageio==2.8.0 opencv-python==4.2.0.32 pillow==7.0.0 pyyaml==5.3 scipy==1.4.1 sharedarray==3.2.0 tensorboardx==2.0 tqdm==4.42.1
# Please refer to env.yml for details
```

## Prepare data
- 2D: The scripts is from 3DMV repo, it is based on python2, other code in this repo is based on python3
	```python prepare_2d_data.py --scannet_path data/scannetv2 --output_path data/scannetv2_images --export_label_images```
	
- 3D: dataset/preprocess_3d_scannet.py

## Config
- BPNet_5cm: config/scannet/bpnet_5cm.yaml 

## Training
- Download pretrained 2D ResNets on ImageNet  from PyTorch website, and put them into the `initmodel` folder.
```python
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
```
- Start training:
```sh tool/train.sh EXP_NAME /PATH/TO/CONFIG NUMBER_OF_THREADS```

- Resume: 
```sh tool/resume.sh EXP_NAME /PATH/TO/CONFIG(copied one) NUMBER_OF_THREADS```

NUMBER_OF_THREADS is the threads to use per process (gpu), so optimally, it should be **Total_threads / gpu_number_used**

## Testing

- Testing using your trained model or our [pre-trained model](https://xxx) (voxel_size: 5cm):
```sh tool/test.sh EXP_NAME /PATH/TO/CONFIG(copied one) NUMBER_OF_THREADS)```


## Copyright and License

You are granted with the [LICENSE](./LICENSE) for both academic and commercial usages.



## Acknowledgment

Our code is based on [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine). We also referred to [SparseConvNet](https://github.com/facebookresearch/SparseConvNet) and [semseg](https://github.com/hszhao/semseg).



## Citation

```tex
@inproceedings{hu-2021-bidirectional,
        author      = {Wenbo Hu, Hengshuang Zhao, Li Jiang, Jiaya Jia and Tien-Tsin Wong},
        title       = {Bidirectional Projection Network for Cross Dimensional Scene Understanding},
        booktitle   = {CVPR},
        year        = {2021}
    }
```

