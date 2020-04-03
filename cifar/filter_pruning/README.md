# Pruning Filters For Efficient ConvNets

This directory contains a pytorch re-implementation of all CIFAR experiments of the following paper  
[Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710) (ICLR 2017).

We also provide the pretrained networks for CIFAR-10 and CIFAR-100 as well as finetuned pruned models.

## Training baseline model 

Supported architecture: `resnet56`, `resnet110`, `preresnet110`, `wrn_28_10`, `vgg16`.

To train a baseline from scratch, run below command and change the `-a` option to any aforementioned architectures.

```shell
python cifar.py -a resnet56 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-56
```

**Result**: We should get the below results when running with above training recipe.
|Model      | #Params | CIFAR-10 | CIFAR-100|
|:--------- |:----------:|:---------:|:---------:|
|Resnet-56 | 0.85M  |   [93.42](https://drive.google.com/open?id=1Ak-KxWbPZNnZHJfrhJEVsXYaR70UmzMJ)    |  [71.07](https://drive.google.com/file/d/1iNpD_HUtaIM6NPkF51OOJA0K6yz1NWC5/view?usp=sharing)         |
|Resnet-110 |  1.72M |   [94.01](https://drive.google.com/file/d/1n6viesspfHl4qAFEkUD8g5Kd0a9QQO0o/view?usp=sharing)    |  [72.35](https://drive.google.com/file/d/1S3NtJM7b4dVhlm9HgRPqMUYFEvJj6bPq/view?usp=sharing)         |
|VGG-16     |     -      | -         | -|

## Pruning

To perform **filter pruning**, run following command:

For **VGG-like** networks:

```shell
python vggprune.py --dataset cifar10 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
```

For **residual** networks (resnet, wideresnet, preresnet): 
```shell
python residualprune.py --dataset cifar10 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
```

Note that for both `resnet` and `VGG`, we adopt the *insensitive* layers follow [Li et al., 2016](https://arxiv.org/abs/1608.08710). That being said, we slightly modified compression ratio of `resnet` to [0.1, 0.2, 0.3] for *stage-1*, *stage-2*, *stage-3* respectively. For `vgg`, we use compression ratio = 0.25 for all pruned layers.

## Fine-tune

Finetuning the pruned networks with supervised loss.

```shell
python finetune.py --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch vgg16
```

### Result:
We should get around the below results when running with above training recipe. We retrain the pruned model with `num_epochs` 40,  `batch_size` 128, `learning_rate` 0.1 and reduce it at 20-th and 30-th epoch

#### 1. Resnet-56

|Model      | #Params | CIFAR-10 | CIFAR-100|
|:--------- |:----------:|:---------:|:---------:|
|Resnet-56 #1 |     0.65M  |   [93.05](https://drive.google.com/open?id=1m71QYlEDHPaX5ueX1p7b5N931n_MroJy)    | 70.69        |
|Resnet-56 #2 |     0.51M  |   [93.21](https://drive.google.com/file/d/1-70MKowxBzIUgh03M6OlQ9L3RYwvy5s_/view?usp=sharing)    |  70.56         |
|Resnet-56 #3 |     0.42M  |   [93.06](https://drive.google.com/file/d/1-7qDS6T5h5oKDsYzo8_i32AMgACk_9C4/view?usp=sharing)    | 70.30         |
|Resnet-56 #4 |     0.34M  |   [92.95](https://drive.google.com/file/d/1-9_8Y9gNmY4kNbNgPSsjQ7-22rZ2VElQ/view?usp=sharing)    | 69.57         |
|Resnet-56 #5 |     0.28M  |   [92.56](https://drive.google.com/file/d/1-9daN9eKwGv6t8QeQ4bCJmXmRHwQ18Ys/view?usp=sharing)    | 69.98         |
|Resnet-56-E-1-5  |     -      |   94.27    | 74.25         |

#### 2. Resnet-110

|Model      | #Params | MACs(G) |CIFAR-10 | CIFAR-100|
|:--------- |:----------:|:---------:|:---------:|:---------:|
|Resnet-110 #1 |     1.26M  |  - | [93.41](https://drive.google.com/file/d/1-10W2X8v9SCG5LrC9zKCTEamPdo6hzSY/view?usp=sharing)    | [71.39](https://drive.google.com/file/d/1-0ODCiTebcraiEM7Gyv56pkVBuOp2Cpw/view?usp=sharing)         |
|Resnet-110 #2 |     0.92M  |   - | [93.39](https://drive.google.com/file/d/1-AuUTWLRCvVIUxV8-NBwp5BQcJ3aXKcF/view?usp=sharing)    | [71.67](https://drive.google.com/file/d/1-1CoHHkToyunGROiE_DE_Lph-mXf378s/view?usp=sharing)         |
|Resnet-110 #3 |     0.70M  |   - | [93.41](https://drive.google.com/file/d/1-BC11kPo_SAXxDVWbiUC3Tr-sVA2uxGt/view?usp=sharing)    | [71.47](https://drive.google.com/file/d/1-4wxnrEW5kagykVDTAE_KNSz1dJ-yarD/view?usp=sharing)        |
|Resnet-110 #4 |     0.52M  |   - | [93.39](https://drive.google.com/file/d/1-EEgrp9FymLpa3cTgCfzMGDGNE36MQOI/view?usp=sharing)    | [71.35](https://drive.google.com/file/d/1-8UNXlEFY-YjubOYpQ57AToIiIHdmPN4/view?usp=sharing)         |
|Resnet-110 #5 |     0.38M  |   0.09 | [93.45](https://drive.google.com/file/d/1-N3-YdXMXDVPm512FufKfZmTqCTNabG1/view?usp=sharing)    | [70.52](https://drive.google.com/file/d/1-8WJhJ_kco7x0jSx9cqzsVQ4CQQBkPmz/view?usp=sharing)         |
|Resnet-110-E-1-5  |          |   - |94.59    |  75.56        |

where `Resnet-56 #x` indicate the pruned model at *x-th* iteration and `Resnet-56-E-x-y` indicate the ensemble of `Resnet-56 #x` **to** `Resnet-56 #y` i.e. **y-x+1** models in total.
## Ensemble Fine-tune

**Manually** modify the variable **checkpoint_paths** in `ensemble_finetune.py` to paths of trained model, which you would like to construct ensemble of. Then, run following command:

```shell
python ensemble_finetune.py --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch resnet56 
```

### Results

Below models are trained with temperature = 5.

#### 1. Resnet-56

|Model      | #Params | Teacher CIFAR-10 | CIFAR-10 | teacher CIFAR-100| CIFAR-100|
|:--------- |:----------:|:---------:|:---------:|:---------:|:---------:|
|Resnet-56 #5 |     0.28M  |   94.27 (Resnet-56-E-1-5)   | [93.36](https://drive.google.com/file/d/1-CuZfD5t8cFRoOj6wuFOdo10bgEvOlov/view?usp=sharing)  | 74.25 (Resnet-56-E-1-5) | - |
|Resnet-56 #5 |     0.28M  |   93.42 (Resnet-56-E-1)   | [93.22](https://drive.google.com/file/d/1-C773-mPqLpRFIWTwzTjg35WlEEvRmT9/view?usp=sharing)  | 71.07 (Resnet-56-E-1) | - |

#### 2. Resnet-110

|Model      | #Params | Teacher CIFAR-10 | CIFAR-10 | teacher CIFAR-100| CIFAR-100|
|:--------- |:----------:|:---------:|:---------:|:---------:|:---------:|
|Resnet-110 #5 |     0.38M  |   94.59 (Resnet-110-E-1-5)   | [94.19](https://drive.google.com/file/d/1-NBKBvS5skQ3p-bdFP5YCoxpF4_33ujm/view?usp=sharing)  | 75.45(Resnet-110-E-1-5) | [72.81](https://drive.google.com/file/d/1zRl15foqN44edaemLrUtmoBlAdce18eS/view?usp=sharing) |
|Resnet-110 #5 |     0.38M  |   94.01 (Resnet-110-E-1)   | [93.85](https://drive.google.com/file/d/1-NG4BxP1symyDVE7uJISwKSsxWj_Obn2/view?usp=sharing)  | 72.35 (Resnet-110-E-1)| [72.10](https://drive.google.com/file/d/1-C9suhbp1zRMwqr8UE9TkCISOHbzKVjk/view?usp=sharing) |