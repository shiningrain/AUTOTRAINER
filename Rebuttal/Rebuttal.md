# Q&A

## Common Questions and Answers

###  **Q1: More Experiments Details, e.g. accuracy improvement, results of all models. (R1, R2, R3)**

Answer:

Due to the space limitations, we can only include part of our experiments results in the paper(which we think are the most important ones). We do have most(at least partial) of the required experiments results (e.g. Table II with all models, accuracy improvement for each kind of problems), and they can be found in our this directory. Following your valuable proposal, we will integrate more experiments details into the paper in our next version.

Additionally, the experiments models we obtained are from ML/SE researchers and companies. All the raw experiment data can be downloaded from [this link](https://drive.google.com/file/d/1AnzEwQZtKXAXA6jo4xGdhRLuAjnUFMLd/view?usp=sharing).



###  **Q2: What is our challenge, novelty and contribution? (R1, R2, R3)**

Answer:

The novel contribution of our work mainly includes 3 parts:

1. AutoTrainer gives the first formal definition of all these training problems and selects the proper parameters.
   
   None of existing works (e.g. [[21]],[[72]],[[90]]) has proposed formal definition for the training problems symptoms. One of our challenges is to summarize the formal definitions from existing works. This involves interviewing hundreds of researchers/engineers, paper summarizations and tons of experiments.

2. AutoTrainer builds the connection between proposed solutions and existing problems
   
   To obtain the effective solutions for the training problems, we firstly collect over 20 solutions from dozens of the existing works (e.g. [[50]],[[70]]) and communities discussions(e.g. [link1](https://stackoverflow.com/questions/46270122/avoiding-vanishing-gradient-in-deep-neural-networks), [link2](https://stackoverflow.com/questions/43436966/gradient-exploding-when-using-rmsprop)). Many methods are proposed to solve a single problem, which may lead to other problems or may fix other problems. It is challenging to select the proper solutions for an observed problem. We evaluated over 20 different training strategies and problems, and selected the meaningful ones.

3. AutoTrainer is the first online model fixing strategy.
   
   Unlike most existing work which tries to improve model accuracy after training, AutoTrainer fixes this while training. This requires online program rewriting (i.e., adding normalization layers) as well as careful design to balance runtime overhead and problem detection/repairing effectiveness.

###  **Q3: How do we set the configurable parameters? (R1, R2)** 

Answer:

The configurable parameters in AutoTrainer have 3 types. As analyzed in paper, Type-A parameters are not affected by the models or tasks, so we set their default values to 0 and Type-B should set by users as their expected accuracy. To obtain suitable values for Type-C parameters(i.e. beta_1,beta_2,beta_3,gamma,theta,zeta,delta,eta), we conduct a series of repeat experiment on 100 models and choose the default values for best detection results which has been discussed detailedly in Evaluation Part.D in paper.

The suitable value for the Type-C parameters will be affected by the learning task because of different data distribution. Therefore, the Type-C parameters depend on tasks, not the models.

We suggest users fine-tune Type-C parameters according to their learning tasks to obtain better detection effect. We admit that the parameters tuning is the limitation of our work. We will provide our codes and open-source AutoTrainer to further improve the parameter tuning. We will also clarify in our next version.

---

## Question and Answer for Reviewer A

###  **Q1: What is the key difference from AutoTrainer to AutoML? ** 

Answer:

AutoTrainer can detect and repair problems automatically in the DNN model training procedure. It provides timely monitoring facing the training process and facilitates SE researchers in repairing buggy models automatically.

Comparing with AutoTrainer, AutoML focuses on designing models for the training tasks, which may still face training problems when training. And AutoTrainer focuses on improving the training itself. These two works are complementary and not in conflict. We will integrate the discussion into Relate Work Part in our next version.

---
## Questions and Answers for Reviewer B

###  **R2Q1: What is the detailed methodology of our benchmark?**

Answer:

The methodology of benchmark in our experiments mainly contains 3 steps:

1. We start two parallel trainings for each model and make sure they have been trained with the same optimization, same training configuration and same environment. The only difference is that one is training with AutoTrainer and the other one is not.
2. Then we compare and analyze two training results. The training results with AutoTrainer will contain detection results, repair results(if models do contain problems) and other training logs(e.g.accuracy, training time, memory). And the training results without AutoTrainer will only contain training logs. In this process, due to the parallel training results, we can easily calculate the amount of FP and FN models.
3. Finally, we calculate the accuracy improvement for each repaired model by the repair log. For the models which are failed to repair, we can also analyze the failure log to understand the details of the repair process. Based on the above, we can record the effectiveness and efficiency of AutoTrainer and obtain the data in Table II in paper.


###  **R2Q2: How to explain the details in our work?( i.e. Training Cost, Symptoms,Solution Effect,FP models and wrongly triggered)**

Answer:

**Training Cost:**
AutoTrainer can save the training resource by automatic detection and repair problems which the users should reproduce and debug manually in the past. It means that with AutoTrainer, users don't need to solve the problem after training but repair the model timely in training procedure.  Thus, overall savings should be larger than reported.

**Symptoms Definitions:**
These definitions in our work are summarized from the previous works which have been widely accepted. The citation(e.g. [[72]],[[90]]) have been shown in Table I of the paper.

**Solution Effect:**
Table III in paper has shown the repair results of each solution in our experiment and how many models it repaired. The corresponding explanations are shown in Evaluation Part.B in paper.

**FP, little effect and Wrongly triggered: models:**
The little effect models have already been shown in the purple rows in Table II. And we also analyzed them in Evaluation Part.B in paper.

We agree that there is a possibility for false positive cases in the detection. However, in our experiments on 495 models, we don't find such cases and the amount of the models which wrongly triggered solutions is 0. Therefore, we will consider such cases as future work.

Additionally, the table with all model detection and repair results can be found on our [GitHub page](./All_models_detail.csv)).

---
## Question and Answer for Reviewer C

###  **R3Q1: What is the practical value of our repair strategy?**

Answer:

The solutions in our repair strategy are from the existing works (e.g. [[50]],[[70]]) whose quality has already been commonly recognized. Our contribution is to collect various solutions from previous works, evaluate the truly effective ones and build the connection between solutions and problems.

Additionally, we admit that our strategy can't solve all the problems. For some hidden problems which have hardly been reported and rarely solved, the existing solutions in AutoTrainer can not repair them. Therefore, we will open-source of our code and try to improve our solutions continuously.

[21]:https://arxiv.org/abs/2005.06195
[50]:https://arxiv.org/abs/1502.03167
[70]:https://arxiv.org/abs/1804.07612
[72]:https://arxiv.org/abs/1805.10369
[90]:https://arxiv.org/abs/1412.6558