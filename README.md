# [ScanMix](https://arxiv.org/abs/2103.11395)

[[arXiv]](https://arxiv.org/abs/2103.11395)

In Pattern Recognition 2023 

[Ragav Sachdeva](https://ragavsachdeva.github.io/), [Filipe R. Cordeiro](https://scholar.google.com/citations?user=w-c3Z2EAAAAJ&hl=en&oi=ao), [Vasileios Belagiannis](https://scholar.google.com/citations?hl=en&user=4IlWd90AAAAJ), [Ian Reid](https://scholar.google.com/citations?hl=en&user=ATkNLcQAAAAJ), [Gustavo Carneiro](https://scholar.google.com/citations?hl=en&user=E0TtOWAAAAAJ).

The pipeline for running ScanMix is the following:

### For CIFAR:

Step 1:

`python simclr.py --config_env configs/env.yml --config_exp configs/pretext/<config_file.yml>`
`python my_simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr_cifar10.yml`

Step 2:

`python scan.py --config_env configs/env.yml --config_exp configs/scan/<config_file.yml>`
`python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml`
`python my_scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar100.yml`

Step 3:

(symmetric) `python ScanMix.py --r [0.2/0.5/0.8/0.9] --noise_mode sym --config_env configs/env.yml --config_exp configs/scanmix/<config_file.yml>`

`python my_ScanMix.py --r 0.2 --noise_mode sym --config_env configs/env.yml --config_exp configs/scanmix/scanmix_cifar10.yml --load_state_dict results/cifar-10/scanmix/r=0.9_sym_util3/savedDicts/checkpoint.json`
`python my_ScanMix.py --r 0.8 --noise_mode sym --config_env configs/env.yml --config_exp configs/scanmix/scanmix_cifar100.yml`

(asymmetric) `python ScanMix.py --r [0.4/0.49] --noise_mode asym --config_env configs/env.yml --config_exp configs/scanmix/<config_file.yml>`

`python my_ScanMix.py --r 0.4 --noise_mode asym --config_env configs/env.yml --config_exp configs/scanmix/scanmix_cifar10.yml`

(semantic) `python ScanMix.py --r 0.4 --noise_mode semantic_[vgg/resnet/densenet] --config_env configs/env.yml --config_exp configs/scanmix/<config_file.yml>`


### For Webvision

Step 0:

Use [moco-v2](https://github.com/facebookresearch/moco) to pre-train an InceptionResNetV2 on Webvision and save the model weights as `moco_pretrained/webvision.tar`.

Step 1:

`python moco.py --dataset webvision --config_env configs/env.yml --config_exp configs/pretext/<config_file.yml>`
`python my_moco.py --dataset webvision --config_env configs/env.yml --config_exp configs/pretext/moco_webvision.yml`
Step 2:

`python my_scan.py --config_env configs/env.yml --config_exp configs/scan/<config_file.yml>`

Step 3:

`python ScanMix_parallel.py --config_env configs/env.yml --config_exp configs/scanmix/<config_file.yml> --cudaids 0 1`
`python my_ScanMix_webvision.py --config_env configs/env.yml --config_exp configs/scanmix/scanmix_webvision.yml --load_state_dict results/webvison/scanmix/savedDicts/checkpoint.json`