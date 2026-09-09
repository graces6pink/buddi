# BUDDI
<b> Generative Proxemics: A Prior for 3D Social Interaction from Images </b>\
[![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://muelea.github.io/buddi/) [![arXiv](https://img.shields.io/badge/arXiv-2305.20091-00ff00.svg)](https://arxiv.org/abs/2306.09337) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1P7x2gY_VuFz5yjZHTRfEBzsEmA-JYGlE?usp=sharing)

https://github.com/muelea/buddi/assets/13314980/b0de0db7-e24f-4c74-8f4d-5029b7d320a2

> ### About this fork
>
> This is a fork of [muelea/buddi](https://github.com/muelea/buddi) with added
> support for the [Inter-X](https://liangxuy.github.io/inter-x/) two-person motion
> dataset, plus reproduction scripts and evaluation tooling. Changes over upstream:
>
> - **Inter-X pipeline** — interaction-segment detection and subsampling
>   (`process_interx.py`), and a synthetic six-view BEV stage that renders the mocap
>   meshes from orbiting cameras so the BEV-conditioned model can train on a dataset
>   that has no images (`process_interx_bev.py`). See [Dataset](./documentation/DATA.md#inter-x).
> - **Reproduction pipelines** — end-to-end scripts for CHI3D
>   (`datasets/scripts/CHI3D/repro/`, stages `s00`–`s08`) and FlickrCI3D
>   (`datasets/scripts/FlickrCI3D/repro/`).
> - **Evaluation tooling** — `sample_and_rank.py` (generate a candidate pool and keep
>   the closest / farthest interactions), `eval_conditional.py`, and the
>   `compare_samples_*` scripts for side-by-side model comparison.
> - **[Custom training guide](./documentation/CUSTOM_TRAINING_GUIDE.md)** for bringing
>   your own two-person dataset.
>
> **No datasets or model weights are included in this repository.** CHI3D, FlickrCI3D,
> Hi4D and Inter-X each require individual registration and prohibit redistribution, as
> do the SMPL/SMPL-X body models. See [Dataset](./documentation/DATA.md) for where to
> obtain each one and the directory layout the code expects.
>
> Upstream README follows.

## In this repo you will find ...

... [BUDDI](#unconditional-sampling), a diffusion model that learned the joint distribution of two people in close proxeminty -- thus a a <b>BUD</b>dies <b>DI</b>ffusion model. BUDDI directly generates [SMPL-X](https://smpl-x.is.tue.mpg.de) body model parameters for two people.

... [Optimization with BUDDI](#optimization-with-buddi), we use BUDDI as a prior during optimization via an SDS loss inspired by [DreamFusion](https://arxiv.org/pdf/2209.14988.pdf). This approach does not require ground-truth contact annotations.

... [Flickr Fits](#flickr-fits), we create SMPL-X fits for [FlickrCI3D](https://ci3d.imar.ro) via an optimization method that takes ground-truth contact annotations between two people into account.

## NEWS!!
:boom: [Google Colab](https://colab.research.google.com/drive/1P7x2gY_VuFz5yjZHTRfEBzsEmA-JYGlE?usp=sharing) with custom images is available :boom:

:boom: Demo code available to run the optimization with BUDDI on your own images :boom:

:boom: Improved installation scripts (with ViTPose and BEV included) :boom:

We have a new version of BUDDI with BEV conditioning 

## Release status

| BUDDI inference | BUDDI training | Optimization with BUDDI | Optimization with BUDDI conditional | Training Data / Flickr Fits |
| :----: | :----: | :----: | :----: | :----: |
| &check; | &check;  | &check; | &check; | &#x2717; |


## Installation and Quick Start
Please see [Installation](./documentation/INSTALL.md) for details. 

```
# install conda environment
./install_conda_env.sh

# download essentials and models
./fetch_data.sh

# download body models (SMPL-X, SMPL, SMIL). The script will ask for you username
# and password for the SMPL-X and SMPL website. If you don't have an account, please
# register under https://smpl-x.is.tue.mpg.de/ and https://smpl.is.tue.mpg.de/.
./fetch_bodymodels.sh

# Install BEV and ViTPose and convert body models to BEV format 
./install_thirdparty.sh

# Run optimization with BUDDI on your own images
# We have some internet images in [this](./demo/data/FlickrCI3D_Signatures/demo/images_live) folder.
# The script will first run BEV and ViTPose and then start the optimization with BUDDI.
# To run the demo with OpenPose on top, please read the comments in demo.sh
./demo.sh
```

## Datasets
Please see [Dataset](./documentation/DATA.md) for details.

## Demo 

### Unconditional sampling



https://github.com/muelea/buddi/assets/13314980/ac93baf2-750e-4223-b9eb-2422004e972c



Unconditional generation stating from random noise using different sampling schedules

```
# linear schedule starting from max-t and skipping every skip-steps step. Here, it's 1000 990 980 ... 20 10. 
python llib/methods/hhc_diffusion/evaluation/sample.py --exp-cfg essentials/buddi/buddi_unconditional.yaml --output-folder demo/diffusion/samples/ --checkpoint-name essentials/buddi/buddi_unconditional.pt --max-images-render=100 --num-samples 100 --max-t 1000 --skip-steps 10 --log-steps=100 --save-vis 
```



### Optimization with BUDDI trained with BEV conditioning




https://github.com/muelea/buddi/assets/13314980/89d6a7de-e907-46ac-83c5-321174ca0eba




Run optimization using BUDDI as prior. This script will find all OpenPose Bounding boxes on a photo and run Optimization with BUDDI for all pairs of people who overlap on the picture.
```
python llib/methods/hhcs_optimization/main.py --exp-cfg llib/methods/hhcs_optimization/configs/buddi_cond_bev_demo.yaml --exp-opts logging.base_folder=demo/optimization/buddi_cond_bev_demo datasets.train_names=['demo'] datasets.train_composition=[1.0] datasets.demo.original_data_folder=demo/data/FlickrCI3D_Signatures/demo datasets.demo.image_folder=images model.optimization.pretrained_diffusion_model_ckpt=essentials/buddi/buddi_cond_bev.pt model.optimization.pretrained_diffusion_model_cfg=essentials/buddi/buddi_cond_bev.yaml logging.run=fit_buddi_cond_bev_flickrci3ds
```

Run optimization with BUDDI on FlickrCI3D. First follow the data and install instructions, then run the commands below. You can use --cluster_pid and --cluster_bs flags to process only a few images or distribute batches of data on a cluster.
```
# run optimization for training split
python llib/methods/hhcs_optimization/main.py --exp-cfg llib/methods/hhcs_optimization/configs/buddi_cond_bev.yaml --exp-opts logging.base_folder=demo/optimization/buddi_cond_bev logging.run=fit_buddi_cond_bev_flickrci3ds datasets.train_names=['flickrci3ds'] datasets.train_composition=[1.0] datasets.val_names=[] datasets.test_names=[] model.optimization.pretrained_diffusion_model_ckpt=essentials/buddi/buddi_cond_bev.pt model.optimization.pretrained_diffusion_model_cfg=essentials/buddi/buddi_cond_bev.yaml

# to run optimization on the validation split set
datasets.train_names=[] datasets.train_composition=[] datasets.val_names=['flickrci3ds'] datasets.test_names=[]

# to run optimization on the test split set
datasets.train_names=[] datasets.train_composition=[] datasets.val_names=[] datasets.test_names=['flickrci3ds']
```

## Training

### Flickr Fits

To create training data, we fit SMPL-X to images from [FlickrCI3D Signatures](https://ci3d.imar.ro/flickrci3d) via an optimization method that takes ground-truth contact annotations between two people on the human body into account. First follow the data and install instructions, then run the commands below. You can use --cluster_pid and --cluster_bs flags to process only a few images or distribute batches of data on a cluster.
```
# run optimization on FlickrCI3D Signatures training split
python llib/methods/hhcs_optimization/main.py --exp-cfg llib/methods/hhcs_optimization/configs/flickr_fits.yaml --exp-opts logging.base_folder=demo/optimization logging.run=flickr_fits datasets.train_names=['flickrci3ds'] datasets.train_composition=[1.0] datasets.val_names=[] datasets.test_names=[]

# to run optimization on the validation split set
datasets.train_names=[] datasets.train_composition=[] datasets.val_names=['flickrci3ds'] datasets.test_names=[]

# to run optimization on the test split set
datasets.train_names=[] datasets.train_composition=[] datasets.val_names=[] datasets.test_names=['flickrci3ds']
```

### BUDDI Training
Follow the data download and processing steps in [Dataset](./documentation/DATA.md). Then run: 
```
# conditional model
python llib/methods/hhc_diffusion/main.py --exp-cfg llib/methods/hhc_diffusion/configs/config_buddi_v02_cond_bev.yaml --exp-opts logging.base_folder=demo/diffusion/training logging.run=buddi_cond_bev datasets.augmentation.use=True model.regressor.losses.pseudogt_v2v.weight=[1000.0] logging.logger='tensorboard'

# unconditional model 
python llib/methods/hhc_diffusion/main.py --exp-cfg llib/methods/hhc_diffusion/configs/config_buddi_v02.yaml --exp-opts logging.base_folder=demo/diffusion/training logging.run=buddi datasets.augmentation.use=True datasets.chi3d.load_unit_glob_and_transl=True datasets.hi4d.load_unit_glob_and_transl=True model.regressor.losses.pseudogt_v2v.weight=[100.0] logging.logger='tensorboard'
```


## Evaluation 

To compare generated meshes against the training data on SMPL-X parameter FID you can use the evaluation script. To generate a x_starts_smplx.pkl file, see [here](#unconditional-sampling).
```
python llib/methods/hhc_diffusion/evaluation/eval.py --exp-cfg llib/methods/hhc_diffusion/evaluation/config_eval.yaml --buddi demo/diffusion/samples/generate_1000_10_v0/x_starts_smplx.pkl --load-training-data
```

We evaluate BUDDI against pseudo-ground truth fits and ground-truth contact labels of FlickrCI3D.
```
python llib/methods/hhcs_optimization/evaluation/flickrci3ds_eval.py --exp-cfg llib/methods/hhcs_optimization/evaluation/flickrci3ds_eval.yaml -gt <base_folder>/fit_pseudogt_flickrci3ds_test -p <base_folder>/<run_folder> --flickrci3ds-split test

python llib/methods/hhcs_optimization/evaluation/chi3d_eval.py --exp-cfg llib/methods/hhcs_optimization/evaluation/chi3d_eval.yaml --predictions-folder <base_folder>/<run_folder> --eval-split test

python llib/methods/hhcs_optimization/evaluation/hi4d_eval.py --exp-cfg llib/methods/hhcs_optimization/evaluation/hi4d_eval.yaml --predictions-folder <base_folder>/<run_folder> --eval-split test
```


## Acknowledgments
We thank our colleagues for their feedback, in particular, we thank Aleksander Holynski, Ethan Weber, and Frederik Warburg for their discussions about diffusion and the SDS loss, Jathushan Rajasegaran, Karttikeya Mangalam and Nikos Athanasiou for their discussion about transformers, and Alpar Cseke, Taylor McConnell and Tsvetelina Alexiadis for running the user study.

Previous work on human pose and shape estimation has made this project possible: we use [BEV](https://github.com/Arthur151/ROMP) to initialize the optimization method, the Flickr and mocap data provided in [Close interactions 3D](https://ci3d.imar.ro/index.php/). We also use previous workon diffusion models and their code bases, [diffusion](https://github.com/hojonathanho/diffusion) and [guided-diffusion](https://github.com/openai/guided-diffusion/tree/main/guided_diffusion).



## Citation
```
@article{mueller2023buddi,
    title={Generative Proxemics: A Prior for {3D} Social Interaction from Images},
    author={M{\“u}ller, Lea and Ye, Vickie and Pavlakos, Georgios and Black, Michael J. and Kanazawa, Angjoo},
    booktitle = {{Computer Vision and Pattern Recognition (CVPR)}},
    year={2024}}
```

## License
See [License](./LICENSE).


## Disclosure
MJB has received research gift funds from Adobe, Intel, Nvidia, Meta/Facebook, and Amazon. MJB has financial interests in Amazon, Datagen Technologies, and Meshcapade GmbH. While MJB is a part-time employee of Meshcapade, his research was performed solely at, and funded solely by, the Max Planck Society.


## Contact
Please contact lea.mueller@tuebingen.mpg.de for technical questions.



---------------------yh----------
基础命令（生成+自动渲染成 GIF）

conda activate buddi
export PYTHONPATH=/home/yuhong/workspace/buddi
export PYOPENGL_PLATFORM=egl        # pyrender 无头渲染必须设这个，不设会报错
cd /home/yuhong/workspace/buddi

python llib/methods/hhc_diffusion/evaluation/sample.py \
  --exp-cfg llib/methods/hhc_diffusion/configs/config_buddi_v02.yaml \
  --output-folder demo/diffusion/samples/我的输出文件夹名 \
  --checkpoint-name /绝对路径/指向某个.pt文件 \
  --num-samples 16 --batch_size 16 \
  --max-t 1000 --skip-steps 20 \
  --save-vis
--checkpoint-name 必须写具体某个 checkpoint 文件的绝对路径（不能用 latest）——这个脚本"自动找最新checkpoint"的逻辑是按配置文件路径去猜文件夹的，跟你训练时实际用的 demo/diffusion/training/interx_uncond/ 对不上，猜不到你真正的 checkpoint 在哪，所以最简单可靠的办法就是每次手动指定文件，比如：


--checkpoint-name /home/yuhong/workspace/buddi/demo/diffusion/training/interx_uncond/checkpoints/2026_07_30-05_42_36__0000000129__0000000167__18.96.pt
可调参数
参数	作用	建议
--num-samples	生成多少个双人姿态样本	想看多样性就调大（比如64），纯展示几张够用就16
--batch_size	一批处理多少个（一般等于 num-samples）	显存够就跟 num-samples 一样
--max-t	从多"重"的噪声开始去噪（1000=从纯噪声开始，标准做法）	一般不用改
--skip-steps	去噪步长，越大跑得越快、细节可能略糙	展示用 20 足够快，追求质量可以调小到 5~10（会变慢）
--max-images-render	不用管，--save-vis 是另一套渲染逻辑，这个参数是给脚本里另一段（目前实际没启用的）代码用的	保持默认 0
--render-width / --render-height	渲染图片尺寸	默认 200×256，想要更清晰改大，比如 400x512
--render-floor	加一个地面网格	想要更有空间感可以加上
生成结果会存在 你的输出文件夹/generate_1000_20_v0/：

renders/*.gif——360°旋转的可视化 GIF，就是你想要的展示效果
x_starts_smplx.pkl——生成的原始 SMPL-X 参数（数值形式，不是图片）
那个"生成一批+挑最近的"的功能
这个不是官方脚本自带的，是我刚才现场写的一段小代码，不在任何文件里。如果你想留着以后自己反复跑，我可以把它整理成一个正式的脚本文件（比如 llib/methods/hhc_diffusion/evaluation/sample_and_rank.py），要不要现在写一个？


脚本写好了，存在 llib/methods/hhc_diffusion/evaluation/sample_and_rank.py，已经实测跑通过。

你自己调用的命令

conda activate buddi
export PYTHONPATH=/home/yuhong/workspace/buddi
export PYOPENGL_PLATFORM=egl
cd /home/yuhong/workspace/buddi

python llib/methods/hhc_diffusion/evaluation/sample_and_rank.py \
  --exp-cfg llib/methods/hhc_diffusion/configs/config_buddi_v02.yaml \
  --checkpoint-name /home/yuhong/workspace/buddi/demo/diffusion/training/interx_uncond/checkpoints/2026_07_30-05_42_36__0000000129__0000000167__18.96.pt \
  --output-folder demo/diffusion/samples/我的展示 \
  --num-samples 64 --top-k 8 --rank closest --export-glb
参数说明
参数	作用
--checkpoint-name	必填，某个 .pt 文件的绝对路径（每次训练进展了想用新的，就换成新文件名）
--num-samples	生成多大一批"候选池"——越大越有机会挑到更极端（更近/更远）的结果，但也越慢
--top-k	从候选池里最终留几个渲染出来
--rank	closest（挑最近/最亲密的，默认）或 farthest（挑最远的）
--skip-steps	去噪步长，越小质量越好但越慢，展示用默认 20 就行
--export-glb	加上这个才会顺便导出 .glb 三维文件（配合上次说的 Blender/VS Code 3D插件交互查看）
--seed	想要结果可复现就设一个固定数字，不设就每次随机
跑完之后去 demo/diffusion/samples/我的展示/top8_closest/ 里找 rankXX_distX.XX.gif（按亲密程度从近到远编号）和对应的 .glb。




Good，vitpose_model.py 用 os.makedirs(out_folder, exist_ok=True)，会处理 image_folder 里的所有图片（不会跳过已有输出，会重新覆盖生成），所以直接对整个 images_live 跑一遍即可，不用单独摘出 test5/test6。BEV 部分 demo.sh 里也是遍历整个文件夹。命令如下：


export PYTHONPATH=~/workspace/buddi

# 1. ViTPose 2D 关键点检测（会重新处理并覆盖 vitpose_live 下所有图片的结果）
python llib/utils/keypoints/vitpose_model.py \
  --image_folder demo/data/images_live \
  --out_folder demo/data/vitpose_live

# 2. BEV 粗略 3D 初始化
for image in demo/data/images_live/*; do
    image_name=$(basename "$image")
    bev -i "$image" -o demo/data/bev_live/$image_name
done
注意：这两步会覆盖 vitpose_live/bev_live 里已存在的同名结果文件（内容应该一致，重新生成没问题）。跑完这两步后再执行上面的 BUDDI 优化命令即可覆盖全部 10 张图片。


export PYTHONPATH=~/workspace/buddi
python llib/methods/hhcs_optimization/main.py \
  --exp-cfg llib/methods/hhcs_optimization/configs/buddi_cond_bev_demo.yaml \
  --exp-opts logging.base_folder=demo/optimization/buddi_cond_bev_demo_live \
             logging.run=fit_buddi_cond_bev_pretrained \
             datasets.train_names=['demo'] datasets.train_composition=[1.0] \
             datasets.demo.original_data_folder=demo/data \
             datasets.demo.image_folder=images_live \
             datasets.demo.bev_folder=bev_live \
             datasets.demo.vitpose_folder=vitpose_live \
             datasets.demo.openpose_folder=none \
             model.optimization.pretrained_diffusion_model_ckpt=essentials/buddi/buddi_cond_bev.pt \
             model.optimization.pretrained_diffusion_model_cfg=essentials/buddi/buddi_cond_bev.yaml


export PYTHONPATH=~/workspace/buddi
python llib/methods/hhcs_optimization/main.py \
  --exp-cfg llib/methods/hhcs_optimization/configs/buddi_cond_bev_demo.yaml \
  --exp-opts logging.base_folder=demo/optimization/buddi_cond_bev_demo_live \
             logging.run=fit_buddi_cond_bev_interx_1499 \
             datasets.train_names=['demo'] datasets.train_composition=[1.0] \
             datasets.demo.original_data_folder=demo/data \
             datasets.demo.image_folder=images_live \
             datasets.demo.bev_folder=bev_live \
             datasets.demo.vitpose_folder=vitpose_live \
             datasets.demo.openpose_folder=none \
             model.optimization.pretrained_diffusion_model_ckpt=demo/diffusion/training/interx_cond_bev/checkpoints/2026_08_07-11_49_01__0000001499__0000000124__34.66.pt \
             model.optimization.pretrained_diffusion_model_cfg=demo/diffusion/training/interx_cond_bev/config.yaml



python llib/methods/hhc_diffusion/evaluation/sample.py \
  --exp-cfg essentials/buddi/buddi_unconditional.yaml \
  --output-folder demo/diffusion/samples/ \
  --checkpoint-name essentials/buddi/buddi_unconditional.pt \
  --max-images-render=100 --num-samples 100 --max-t 1000 --skip-steps 10 \
  --log-steps=100 --save-vis


重新训练bev-con的命令
  python llib/methods/hhc_diffusion/main.py \
  --exp-cfg llib/methods/hhc_diffusion/configs/config_buddi_v02_cond_bev.yaml \
  --exp-opts logging.base_folder=demo/diffusion/training logging.run=<新run名> \
             datasets.train_names=[interx] datasets.train_composition=[1.0] \
             datasets.val_names=[interx] \
             datasets.interx.processed_data_folder=datasets/processed/InterX \
             datasets.interx.allow_missing_bev=True \
             datasets.augmentation.use=True \
             model.regressor.losses.pseudogt_v2v.weight=[1000.0] \
             logging.logger=tensorboard




cd /home/yuhong/workspace/buddi && conda activate buddi && export PYTHONPATH=$(pwd)

# 指定某一帧
python llib/data/preprocess/utils/inspect_interx_frame.py --imgname G001T000A000R000_0_289

# 指定 take + 帧序号
python llib/data/preprocess/utils/inspect_interx_frame.py --take-id G001T000A000R004 --frame-index 2

# 随机抽（优先抽已经被阶段②处理过的帧）
python llib/data/preprocess/utils/inspect_interx_frame.py --random --pick-seed 3

# 同一帧换个相机角度再看
python llib/data/preprocess/utils/inspect_interx_frame.py --imgname ... --new-camera --seed 1

# 额外单独存每一格
python llib/data/preprocess/utils/inspect_interx_frame.py --random --save-panels
