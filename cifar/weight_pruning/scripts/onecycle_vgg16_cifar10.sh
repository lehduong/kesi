# prune_1 22%
echo "PRUNE 1" && 
python cifar_prune.py --arch vgg16 \
                      --dataset cifar10 \
                      --percent 0.2  \
                      --resume checkpoints/model_best.pth.tar \
                      --save_dir prune_1 &&
python cifar_finetune.py --arch vgg16 \
                         --dataset cifar10 \
                         --resume prune_1/pruned.pth.tar \
                         --save_dir prune_1 &&
# prune_2 39%
echo "PRUNE 2" &&
python cifar_prune.py --arch vgg16 \
                      --dataset cifar10 \
                      --percent 0.4 \
                      --resume prune_1/finetuned.pth.tar \
                      --save_dir prune_2 &&
python cifar_finetune.py --arch vgg16 \
                         --dataset cifar10  \
                         --resume prune_2/pruned.pth.tar \
                         --save_dir prune_2 &&
# prune_3 51%
echo "PRUNE 3" &&
python cifar_prune.py --arch vgg16 \
                      --dataset cifar10 \
                      --percent 0.6 \
                      --resume prune_2/finetuned.pth.tar \
                      --save_dir prune_3 &&
python cifar_finetune.py --arch vgg16 \
                         --dataset cifar10  \
                         --resume prune_3/pruned.pth.tar \
                         --save_dir prune_3 &&
# prune_4 59%
echo "PRUNE 4" &&
python cifar_prune.py --arch vgg16 \
                      --dataset cifar10 \
                      --percent 0.8 \
                      --resume prune_3/finetuned.pth.tar \
                      --save_dir prune_4 &&
python cifar_finetune.py --arch vgg16 \
                         --dataset cifar10  \
                         --resume prune_4/pruned.pth.tar \
                         --save_dir prune_4 &&
# prune_5 66%
echo "PRUNE 5" && 
python cifar_prune.py --arch vgg16 \
                      --dataset cifar10 \
                      --percent 0.95\
                      --resume prune_4/finetuned.pth.tar \
                      --save_dir prune_5 &&
python cifar_finetune.py --arch vgg16 \
                         --dataset cifar10  \
                         --resume prune_5/pruned.pth.tar \
                         --save_dir prune_5 &&
# ensemble_finetune
echo "ENSEMBLE FINETUNE" &&
python ensemble_finetune.py --lr 0.001 \
                            --batch-size 128 \
                            --wd 1e-4 \
                            --refine prune_5/finetuned.pth.tar \
                            --dataset cifar10 \
                            --save snapshot_ensemble --arch vgg16
