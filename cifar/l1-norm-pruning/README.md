# Pruning Filters For Efficient ConvNets

This directory contains a pytorch re-implementation of all CIFAR experiments of the following paper  
[Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710) (ICLR 2017).

We also provide the pretrained networks for CIFAR-10 and CIFAR-100 as well as finetuned pruned models.

## Training baseline model 

**Resnet-56**

```shell
python cifar.py -a resnet56 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-56
```

**Resnet-110**

```shell
python cifar.py -a resnet110 --epochs 300 --schedule 150 225 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-110 
```

**Result**: We should get the below results when running with above training recipe.
|Model      | #Params | CIFAR-10 | CIFAR-100|
|:--------- |:----------:|:---------:|:---------:|
|[Resnet-56](https://drive.google.com/open?id=1Ak-KxWbPZNnZHJfrhJEVsXYaR70UmzMJ) | 0.85M  |   93.42    | -         |
|Resnet-110 |  1.72M |   94.35    | -         |
|VGG-16     |     -      | -         | -|

**Todo**: update for `vgg16` model.

## Pruning

Executing filter pruning for trained model. Currently only `resnet-56` and `resnet-110` are supported.

**vgg**

```shell
python vggprune.py --dataset cifar10 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
```

**Resnet-56**

```shell
python res56prune.py --dataset cifar10 -v A --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
```

**Resnet-110**

```shell
python res110prune.py --dataset cifar10 -v A --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
```

Here in `res56prune.py` and `res110prune.py`, the `-v` argument is `A` or `B`, which refers to the naming of the pruned model in the original paper. The pruned model will be named `pruned.pth.tar`. 

Note that, we slightly modified the `A` version of `resnet-56` to increase the compression ratio. In this implementation, the network weights would be pruned at the ratio of $0.1, 0.2, 0.3$ for *stage-1*, *stage-2*, *stage-3* respectively.

## Fine-tune

Finetuning the pruned networks with supervised loss.

**VGG-16**

```shell
python main_finetune.py --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch vgg16 
```

**Resnet-56**

```shell
python main_finetune.py --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch resnet56
```

**Resnet-110**

```shell
python main_finetune.py --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch resnet110 
```

**Result**: We should get around the below results when running with above training recipe. We finetune the pruned model with `num_epochs` 40,  `batch_size` 128.

|Model      | #Params | CIFAR-10 | CIFAR-100|
|:--------- |:----------:|:---------:|:---------:|
|[Resnet-56 #1](https://drive.google.com/open?id=1m71QYlEDHPaX5ueX1p7b5N931n_MroJy) |     0.65M  |   93.05    | -         |
|[Resnet-56 #2](https://drive.google.com/file/d/1-70MKowxBzIUgh03M6OlQ9L3RYwvy5s_/view?usp=sharing) |     0.51M  |   93.21    | -         |
|[Resnet-56 #3](https://drive.google.com/file/d/1-7qDS6T5h5oKDsYzo8_i32AMgACk_9C4/view?usp=sharing) |     0.42M  |   93.06    | -         |
|[Resnet-56 #4](https://drive.google.com/file/d/1-9_8Y9gNmY4kNbNgPSsjQ7-22rZ2VElQ/view?usp=sharing) |     0.34M  |   92.95    | -         |
|[Resnet-56 #5](https://drive.google.com/file/d/1-9daN9eKwGv6t8QeQ4bCJmXmRHwQ18Ys/view?usp=sharing) |     0.28M  |   92.56    | -         |
|[Resnet-56-E-1-5]()  |     -      |   94.27    | -         |

where `Resnet-56 #x` indicate the pruned model at *x-th* iteration and `Resnet-56-E-x-y` indicate the ensemble of `Resnet-56 #x` **to** `Resnet-56 #y` i.e. **y-x+1** models in total.
## Ensemble Fine-tune

**Manually** modify the variable **checkpoint_paths** in `ensemble_finetune.py` to paths of trained model, which you would like to construct ensemble of. Then, run following command:

```shell
python ensemble_finetune.py --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch resnet56 
```

**Result**

|Model      | #Params | Teacher CIFAR-10 | CIFAR-10 | teacher CIFAR-100| CIFAR-100|
|:--------- |:----------:|:---------:|:---------:|:---------:|:---------:|
|[Resnet-56 #5](https://drive.google.com/file/d/1-CuZfD5t8cFRoOj6wuFOdo10bgEvOlov/view?usp=sharing) |     0.28M  |   94.27 (Resnet-56-E-1-5)   | 93.36  | - | - |
|[Resnet-56 #5](https://drive.google.com/file/d/1-C773-mPqLpRFIWTwzTjg35WlEEvRmT9/view?usp=sharing) |     0.28M  |   93.42 (Resnet-56-E-1)   | 93.22  | - | - |