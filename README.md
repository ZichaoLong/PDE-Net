# PDE-Net

The code are for the paper

[PDE-Net: Learning PDEs from Data](https://arxiv.org/abs/1710.09668)[(ICML 2018)](https://icml.cc/Conferences/2018)<br />
[Long Zichao](http://zlong.me/), [Lu Yiping](http://about.2prime.cn/), [Ma Xianzhong](https://www.researchgate.net/profile/Xianzhong_Ma) and [Dong Bin](http://bicmr.pku.edu.cn/~dongbin)

<div  align="center">
<img src="figures/pdenet.jpg" width = "60%" />
</div>

If you find the code useful in your research then please cite
```
@inproceedings{long2018pdeI,
  title={PDE-Net: Learning PDEs from Data},
  author={Long, Zichao and Lu, Yiping and Ma, Xianzhong and Dong, Bin},
  booktitle={International Conference on Machine Learning},
  pages={3214--3222},
  year={2018}
}
```

# Setup

All code were developed and tested on CentOS 7 with Python 3.6, and were implemented by pytorch 0.3.1

The code are based on a submodule: [aTEAM](https://github.com/ZichaoLong/aTEAM), a pyTorch Extension for Applied Mathematics. 
One can either download [a proper version of aTEAM](https://github.com/ZichaoLong/aTEAM/archive/v0.1.tar.gz) and extract it to some directory in your python path with name 'aTEAM', or use 'git submodule' to set up the dependency.

For example, you can create a conda environment and set up PDE-Net like this:

```
conda create -n PDE-Net python=3.6 scipy pyyaml jupyter matplotlib
source activate PDE-Net
conda install pytorch=0.3.1 torchvision cuda90 -c pytorch
git clone git@github.com:ZichaoLong/PDE-Net.git
cd PDE-Net
git checkout PDE-Net # switch your branch from 'master' to the 'PDE-Net'
git submodule init
git submodule update
```


# Training, Testing and Plot

| Model                                     | example of config file    | training                    | testing          | plot             |
| ---                                       | ---                       | ---                         | ---              | ---              |
| Convection-Diffusion Equations            | checkpoint/linpde.yaml    | learn_variantcoelinear2d.py | linpdetest.py    | linpdeplot.py    |
| Diffusion Equations with Nonlinear Source | checkpoint/nonlinpde.yaml | learn_singlenonlinear2d.py  | nonlinpdetest.py | nonlinpdeplot.py |


## Training

- Default options can be found in learn_variantcoelinear.py and learn_singlenonlinear2d.py. You can simply modify the default options in learn_variantcoelinear.py(learn_singlenonlinear2d.py), and simply run code like:
```
python learn_variantcoelinear2d.py
```
- Configure training by command line options:
```
TASKDESCRIPTOR=linpde-test
python learn_variantcoelinear2d.py --taskdescriptor=$TASKDESCRIPTOR \
  --kernel_size=7 --max_order=4 --constraint=moment
```
Training information and learned parameters will be stored in `checkpoint/${TASKDESCRIPTOR}`.

## Testing
```
python linpdetest.py $TASKDESCRIPTOR
# or python nonlinpdetest.py $TASKDESCRIPTOR
```
Then the testing results will be stored in `checkpoint/$TASKDESCRIPTOR/errs.pkl`. 

## Show Results
Set your TASKDESCRIPTOR in `*test.py, *plot.py, errs_compare.py` and run.

# Pretrained Model

[Download pretrained models](https://gitlab.com/ZichaoLong/PDE-Net-Checkpoints) and make your working directory like this:
```
PDE-Net/
  aTEAM/
  figures/
  learn_variantcoelinear2d.py
  linpdetest.py
  ...
  checkpoint/
      linpde5x5frozen4order0.015dt0.015noise-double/
      linpde5x5moment4order0.015dt0.015noise-double/
      linpde7x7frozen4order0.015dt0.015noise-double/
      linpde7x7moment4order0.015dt0.015noise-double/
      nonlinpde7x7frozen2order-double/
      nonlinpde7x7moment2order-double/
```




