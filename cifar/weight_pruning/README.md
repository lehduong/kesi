# Non-Structured Pruning/Weight-Level Pruning

This directory contains a pytorch implementation of the CIFAR experiments of non-structured pruning introduced in this [paper](https://arxiv.org/abs/1506.02626) (NIPS 2015).

We also provide the pretrained networks for CIFAR-10 and CIFAR-100 as well as finetuned pruned models.

We prune only the weights in the convolutional layer. We use the mask implementation, where during pruning, we set the weights that are pruned to be 0. During training, we make sure that we don't update those pruned parameters.

## Training baseline model 

The `dataset` argument specifies which dataset to use: `cifar10` or `cifar100`. The `arch` argument specifies the architecture to use, it could be one of `vgg16`, `resnet56`, 

```shell
python train.py --dataset cifar10 --arch vgg16
```

**Result**: We should get the below results when running with above training recipe.
|Model      | #Params | CIFAR-10 | CIFAR-100|
|:--------- |:----------:|:---------:|:---------:|
|Resnet-56 | 0.85M  |   [93.42](https://drive.google.com/open?id=1Ak-KxWbPZNnZHJfrhJEVsXYaR70UmzMJ)    |  [71.07](https://drive.google.com/file/d/1iNpD_HUtaIM6NPkF51OOJA0K6yz1NWC5/view?usp=sharing)         |
|Resnet-110 |  1.72M |   [94.01](https://drive.google.com/file/d/1n6viesspfHl4qAFEkUD8g5Kd0a9QQO0o/view?usp=sharing)    |  [72.35](https://drive.google.com/file/d/1S3NtJM7b4dVhlm9HgRPqMUYFEvJj6bPq/view?usp=sharing)         |
|VGG-16     |     14.99M      |  94.23    | 73.24 |

## Pruning

Performing weights pruning for trained model..

```shell
python cifar_prune.py --arch vgg16 --dataset cifar10 --percent 0.3 --resume [PATH TO THE MODEL] --save_dir [DIRECTORY TO STORE RESULT]
python cifar_prune.py --arch preresnet110 --dataset cifar10 --percent 0.3 --resume [PATH TO THE MODEL] --save_dir [DIRECTORY TO STORE RESULT]
```

For iterative pruning, simply increase `percent` of pruned model. This is an example of commands used for 3 steps iterative pruning:

**Step 1**:

```shell
python cifar_prune.py --arch vgg16 --dataset cifar10 --percent 0.3 --resume checkpoints/checkpoint.pth.tar --save_dir prune_1
python cifar_finetune.py --arch vgg16--dataset cifar10  --resume prune_1/pruned.pth.tar --save_dir prune_1
```

**Step 2**:
```shell
python cifar_prune.py -a vgg16 -d cifar10 --percent 0.6 --resume prune_1/finetuned.pth.tar --save_dir prune_2
python finetune.py -a vgg16 -d cifar10 --resume prune_2/pruned.pth.tar --save_dir prune_2
```

**Step 3**:

```shell
python cifar_prune.py -a vgg16 -d cifar10 --percent 0.9 --resume prune_2/finetuned.pth.tar --save_dir prune_3
python finetune.py -a vgg16 -d cifar10 --resume prune_3/pruned.pth.tar --save_dir prune_3
```

## Fine-tune

Retrained the masked network. We ensure gradient of zero weights to be zero as well.

```shell
python cifar_finetune.py --arch vgg16--dataset cifar10  --resume [PATH TO THE PRUNED MODEL6
python cifar_finetune.py --arch preresnet110 --dataset cifar10  --resume [PATH TO THE PRUNED MODEL]
```

## Ensemble Fine-tune

**Manually** modify the variable **checkpoint_paths** in `ensemble_finetune.py` to paths of trained model, which you would like to construct ensemble of. Then, run following command:

```shell
python ensemble_finetune.py --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch resnet56 
```