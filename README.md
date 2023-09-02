# Overview

This work contains the source code of [NeurIPS 2022: “Why Not Other Classes?”: Towards Class-Contrastive Back-Propagation Explanations](https://openreview.net/pdf?id=X5eFS09r9hm)

**Method**

Given an input and a class label, the corresponding explanations are greatly affected by the targets. In conventional practice, the predicted logits are selected as the target. However, since the logit only contains information of one class, the explanation actually fails to explain the classification result. In order to answer "Why is the input classified into this class?", the predicted logits of all other classes need to be taken into consideration.

![alt text](https://github.com/yipei-wang/Images/blob/main/Contrastive/ContrastiveBP.png)

For academic usage, please consider citing

<pre>
@article{wang2022not,
  title={“Why Not Other Classes?”: Towards Class-Contrastive Back-Propagation Explanations},
  author={Wang, Yipei and Wang, Xiaoqian},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={9085--9097},
  year={2022}
}
</pre>


## Contents

**Libraries**

<pre>
  numpy==1.19.5
  torch==1.10.2
  torchvision=0.11.3
</pre>

**Tutorial**

Please have the [CUB-200-2011]{https://www.vision.caltech.edu/datasets/cub_200_2011/} bird dataset ready. 
For finetuning the model on the dataset, please run
<pre>
  python train.py --root path_of_the_dataset
</pre>

For the difference between contrastive explanations and non-contrastive explanations, please see this [quick start](quick_start.ipynb) for the demonstration

In order to reproduce the quantitative experiments of the relative probability between the top two classes, please run
<pre>
  python experiment.py -root path_of_the_dataset
</pre>
