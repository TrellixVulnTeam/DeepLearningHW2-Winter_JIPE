r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 4
    hidden_dims = 64
    activation = 'tanh'
    out_activation = 'logsoftmax'
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part1_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.NLLLoss()
    lr = 5e-2
    weight_decay = 0.01
    momentum = 0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part1_q1 = r"""
**Your answer:**

1.  Optimization error - an optimization problem is the problem of finding the best solution from all feasible solutions. 
We think that our model's optimization error is relatively milled. We can see in the train graph that the acc is 
reaching ~92% accuracy - which in our opinion is fairly good result.

2.  Generalization error - generalization error is a measure of how accurately an algorithm is able to predict outcome values for previously unseen data.
we can see that in the test accuracy graph the final result was relatively high, 
but most of the iterations prior to the final one were quite scattered between 85-75% - which indicates to a high generalization error. 

3. Approximation error - we think that the approximation error of our model is fairly good. 
we can see it in the decision boundary plot that the shape of the decision boundary is aprox. like we would draw out by hand.
 meaning that it resembles the h* function.
 
 
 
"""

part1_q2 = r"""
**Your answer:**

we expect that the fnr will be higher then the fpr. 
we can see in the first plot of the validation data, that there are more places where it makes sense for us to think
that something orange is blue than the opposite. and that is why we expect that the FNR will be higher than the FPR.
(the samples in the testset are mostly negative in the trainset but they are positive in the validset)


"""

part1_q3 = r"""
**Your answer:**

1. in this scenario we would choose the "optimal" point on the ROC cover to be as close to fpr = 0 and fnr = 1.
 because we know that the symptoms are non-lethal and a low-cost treatment is available, 
 we wouldn't wont that our model will give positive response, and by that make our patient take an expensive and dangerous test.
 
2. in this case we would choose the "optimal" point on the ROC cover to be as close to fpr = 0 and fnr = 0.
we know that the symptoms can be undetected until its to late and the patient can die if we wouldn't detect the disease early enough,
 so we prefer to pay for the expensive test and risk his life for the detection of the disease.
 
 
 
"""

part1_q4 = r"""
**Your answer:**
1. According to the experiments we can see that we get the best result (the decision boundaries and model performance fits better the data) as
the width increases.
There were cases depending on the depth that for example: for a model with a low depth it could be seen that for a larger width we get better results.
For models with a higher depth it could be seen that it did not work that way, as the width increased the results were better up to a certain point.
After that it could be seen that increasing the width was hurts the results.

2.According to the experiments it seems that as the depth increased we get better results (the decision boundaries and model performance fits better the data)
There were cases depending on the width that as we increased the depth the results improved (when the width was smaller), 
and cases where increasing the depth resulted to better results until a certain point and after that made the results a little less good (happened when the width was larger)


3. The first pair of configurations: we can see that the model with depth=4 and width=8 gets better results. (gets a less linear decision boundary)
the second model with depth=1 width=32 gets less good results (gets a more linear decision boundary and that is why can fit the data less well)

The second pair of configurations: the models look almost the same, the model with depth=1 and width=128 has a little bit better results.
It seems that if the model is wide enough it is equivalent to the deep model even though non linearity is applied only once.


4. Threshold selection on the validation set improves the results on the test set because it might reduce the generalization error 
because we select it on some data and this data is the validation set that is almost similar to the train set
for example in noise and how they made.


"""


# ==============
# Part 2 answers


def part2_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 5e-2
    weight_decay = 0.01
    momentum = 0.09
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part2_q1 = r"""
**Your answer:**

1. for the left architecture the number of parameters will be (3*3*256+1)*256 + (3*3*256+1)*256 = 1180160. where the 
    3*3 is the size of the kernel, times the input dim + the baies, the result is multipled by the out dim in this case 
    256. we have two conv layers. in the right architecture the number of parameters are (1*1*256+1)*64 +(3*3*64+1)*64 +( 
    1*1*64+1)*256 = 70016. in this architecture we have 3 conv layers. in addition we can see that the num of parmetes in 
    the right (bottleneck) is much lower then the regular setting.
2. due to the smaller dim of the bottleneck we can 
    assume that the conv operation on the bottleneck will result in fewer FLOPS calculations - as also the dim of the 
    kernel is smaller and the overall dim of the matrix is also smaller in comparison to a residual  architecture
3. (1) spatially: in this metric we think that the regular res block that has 2 conv layers will have better spatial 
    ability then the bottleneck one - since in the bottleneck architecture we have only one 3*3 conv 
    (1*1 conv isnt really helping in spatial ability).
    
    (2) across feature maps: in this metric we think that bottleneck architectures have clear strength compere to 
    regular res architectures - since we are transferring data identical through the layers. 
"""

# ==============

# ==============
# Part 3 answers


part3_q1 = r"""
**Your answer:**

1. The depth that produces the best results is when L=4 this is based on the accuracy of the test batch even though the train set goot a lower accuracy score then L2.
we can see that as a rule of thumb the best results can be achieved with sorter networks.
perhaps the reason that our shorter networks received the best accuracy is because they were enough complicated but 
not to much, that they aren't suffering from vanishing gradients.


2. for L=8 and L=16 we can see that the network wasn't trainable. 
the reason may be due to vanishing gradients (=vanishing information) and the network is no longer is training the in
 order to minimize the loss value, and we get a model which isn't better then a random guess.
    1. one way is adding loss function that are partial.
we will add them to different parts of the network, and by that creating "skipp connections" for loss functions.
    2. we can think about adding skipping connections like in Part 2 - in that way it will use information that will not disappear as easily.



"""

part3_q2 = r"""
**Your answer:**

looking at the results from L=2,4 we can see that we got +/- the same accuracy scores on the test set - 
with L=4 is leading by a small margin.
in addition we see that after epoch 7-8 we are starting to overfit to our training set while the test set is plateauing.
we can also see that there is no major affect from the different conv sizes. 

looking at L=8, we can see that this net architecture isn't trainable. 

this is similar results from what we got in experiment 1 - were after L=4 all of the settings weren't trainable.

"""

part3_q3 = r"""
**Your answer:**


we can see that for all the other L except L=1, we get vanishing
gradients- the network become too deep :(

for L=1 we can see that after epoch 10 the accuracy results are plateauing and the model is starting to overfit to the 
train set. 

we can see that the rule of thumb 
(were after a certain point of architecture complexity the networks are starting to be un trainable)
 that we absorbed in the previous experiments can be seen also in this experiment.

"""

part3_q4 = r"""
**Your answer:**


we can see that we succeeded to train deeper models without vanishing gradients.
we can see that there is a decreasing in accuracy maybe because of the hyper params.  
compared to experiments 1.1 , 1.2 and 1.3 the results from the trainable architectures are lower in a significant way.

"""

part3_q5 = r"""
**Your answer:**

1. first of all, we replicated the functionality of the ResNet and we also used batchNorm and dropout.
 in addition we used pool_every=2 and used 2 layers of 200 dim in the FC stage. 
 the main idea was to deal with the vanishing gradients that we saw in the experiments. 
2. we see significant improvement both in the scores - now in the first time we are getting accuracy scores of above 80% in the test set. 
and also for the first time we are getting trainable networks even with complex and long architectures - 
which we can link to the addition of the resNet.
"""
# ==============
