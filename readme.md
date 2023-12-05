# NEURIPS MACHINE UNLEARNING 2023

This repository contains the details of my approach for the [neurips-machine-unlearning-2023](https://www.kaggle.com/competitions/neurips-2023-machine-unlearning)  competition hosted in kaggle.

My final rank in the competition is 589/1188 which is really bad. I am not totally sure why the approaches didn't work but I will mention some of them in this repository with codes.

The idea is that we are given a model which is trained on a dataset (trained model). Now, a request comes to remove the influence on set of examples (called as forget dataset) from the model that is the model should behave same when it is retrained from sctract excluding these examples.

So for this competition we were given trained model and retain and forget set. Forget set is 2% of original training dataset (it means retain is 98%). The ideal way is to retrain the model using only retain set from sctrach but this will take more computational power. So unlearning approaches will try to remove the influence of forget dataset without retraining. More details can be found in the competition website.

As this is new field for me, what I did for the whole competition is implemented the approaches mentioned in the research papers (will also attach them).


## FINETUNE ON RETAIN SET
This is the simple approach where the learned model is finetuned once again on retain set. This is the baseline mentioned in most of papers.

The idea is that neural networks are prone to forgetting. That is if you take a pretrained model (trained on imagenet for example) and then retrain the model on some other dataset, it is observed that model performance on pretrained dataset may drop significantly

The same logic also applies in unlearning. This is the baseline that one can compare their new approaches. The notebook with this code is : [layer-finetune](./layer-finetune.ipynb)

After the finetuning, I have added a gausian noise with standard deviation of `1e-3` to all the weights.

## FISHER NOISE

This approach is based on the paper : [selective-forgetting-in-deep-networks](https://arxiv.org/pdf/1911.04933.pdf)

The idea is that each weight is updated by adding gaussian noise. The final weights will look like this

$$ S(w) = w + N(0, \sigma^{2}) $$

According to the paper, the variance is 

$$ \sigma^{2} = k F^{\frac{-1}{4}} $$

where $F$ is diagonal element of fisher information matrix. This matrix is calculated on retain dataset. In the paper authors mentioned about the full update but calculating the FIM for a neural network will be costly (as its size will be very big). So instead they suggested to use only the diagonal element as variance for the noise.

The details on constant $k$ can be found in notebook. For writing this part of the code, I have taken the reference from authors repo : [Selective Forgetting](https://github.com/AdityaGolatkar/SelectiveForgetting)

The notebook with code is [fisher-noise](./fisher-noise.ipynb). Also I tried some ideas on top of this approach, so the final code may look different.


## LAST LAYER PERTUBATION

This is based on the paper [certified-removal-from-machine-learning-models](https://arxiv.org/pdf/1911.03030.pdf)

For logistic regression, or models where the loss function is convex with respect to weights, authors proposed the following newton step update

$$ w = w + H^{-1}_{r}\nabla_{f} $$

where $H$ is hessian of loss function calculated on retain dataset, $\nabla_{f}$ is gradient of loss function on forget dataset. We calculate these values using the trained model given.

Given that we have neural network and loss function is not convex, what I did only updated the weights in the last layer which is usually called as fc (fully connected) layer. For all other layer weights, I have added gaussian noise of standard deviation `1e-3` and then did the feature extraction (output of last second layer) on both retain and forget set. Using these values, I have calculated hessian, gradient on respective sets and updated the weights based on above equation.

Note that this approach is valid here as the competition metric is based on the difference of output distribution between unlearned model and retrained from sctrach model. If the competition metric is based on differences between weight distribution then this approach will not work.

The notebook with the code is [last-layer-pertub](./last-layer-pertub.ipynb)
