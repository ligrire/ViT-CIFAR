# ViT-CIFAR
This is the repository for CSCI2470 final project: Joint Image Training for Transformers in Supervised Learning.

The directory "preliminary experiment" is our code to do preliminary experiment. Code for training is written by us, and in model.py we modified code of vision transformer structure in vit-pytorch[[link]](https://github.com/lucidrains/vit-pytorch) to  write our "joint vision transformer". In preliminary experiment we get around 5% boost for joint training with 2 images.(80% -> 85%)

The directory "omihub777" is a fork to [link](https://github.com/omihub777/ViT-CIFAR). We modified the model and the training procedure to get the joint training running. The graph of result is in devpost and write-up.

**Note** in  default setting forked repository use autoaugment to get 90% test accuracy on CIFAR-10. 
To better compare our method, we turn off autoaugment in our whole experiment and just use random crop and horizontol flipping as augmentation. 
The accuracy of their model when turned off autoaugment is around 85%.

