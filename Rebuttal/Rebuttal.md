# Q&A

We thank the reviewers for their precious time and valuable feedback. We first reply to the questions of common interest and then answer the questions from review 1 to review 3. 

## Common Questions and Answers

###  **Q1: More Experiments Details, e.g. accuracy improvement, results of all models. (R1, R2, R3)**

Answer:

Due to the space limitations, we can only include part of our experiments results in the paper(which we think are the most important ones). We do have most(at least partial) of the required experiments results (e.g. Table II with all models, accuracy improvement for each kind of problems), and they can be found in our this directory. Following your valuable proposal, we will integrate more experiments details into the paper in our next version.

Additionally, the experiments models we obtained are from ML/SE researchers and companies. All the raw experiment data can be downloaded from [this link](https://drive.google.com/file/d/1AnzEwQZtKXAXA6jo4xGdhRLuAjnUFMLd/view?usp=sharing).



###  **Q2: What is our challenge, novelty, contribution and limitations? (R1, R2, R3)**

Answer:

AutoTrainer is challenging and novel. In next revision, we will add more discussions on challenges, insights and limitations.

Firstly, AutoTrainer gives the first formal definition of all these training problems and selects the proper parameters. None of existing works (e.g. [[21]],[[72]],[[90]]) has proposed formal definition for the training problems symptoms. One of our challenges is to summarize the formal definitions from existing works. This involves interviewing hundreds of researchers/engineers, paper summarizations and tons of experiments.

Secondly, AutoTrainer builds the connection between proposed solutions and existing problems. Many methods are proposed to solve a single problem, which may lead to other problems or may fix other problems. It is challenging to select the proper solutions for an observed problem. To obtain the effective solutions for the training problems, we firstly collect and evaluate over 20 solutions from the existing works (e.g. [[50]],[[70]]) and communities discussions(e.g. [link1](https://stackoverflow.com/questions/46270122/avoiding-vanishing-gradient-in-deep-neural-networks), [link2](https://stackoverflow.com/questions/43436966/gradient-exploding-when-using-rmsprop)), and selected the meaningful ones.

Thirdly, AutoTrainer is the first online model fixing strategy. Unlike most existing work which tries to improve model accuracy after training, AutoTrainer fixes this while training. This requires online program rewriting (i.e., adding normalization layers) as well as careful design to balance runtime overhead and problem detection/repairing effectiveness.

We were not able to share some of aforementioned experiments (e.g., failed solutions and parameters) due to page limit and copyright issues (i.e., models and datasets). Summarized data can be found on [this page](./README.md). We will add more details in our next version.

We admit AutoTrainer has its limitations in detecting other bugs. We will add more discussions on this. How to improve it will be our future work.

###  **Q3: How do we set the configurable parameters? (R1, R2)** 

Answer:

The configurable parameters in AutoTrainer have 3 types. As analyzed in paper, Type-A and Type-C parameters are not affected by the models or tasks, so we use fixed values for all models in evaluation. Type-B parameters are **task/dataset-dependent** instead of **model-dependent**. We use the same values for them for all models trained for the same dataset and evaluate the values in Section-V-D.

The chosen values of the parameters are obtained from 100 cases, and we evaluated them on another 495 models. We will add more details about this process. 

We suggest users adjust Type-B parameters according to their learning tasks to obtain better detection effect. We will provide our codes and open-source AutoTrainer for further improvement. We will also clarify in our next version.

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

1. We start **two parallel trainings** for each model and make sure they have been trained with the same optimization, same training configuration and same environment. The only difference between them is that one is training with AutoTrainer and the other one is not.
2. Then analyze two training results manually. The training results with AutoTrainer will contain detection results, repair results(if models do contain problems) and other training logs(e.g.accuracy, training time, memory usage). And the training results without AutoTrainer will only contain training logs.
3. Finally, we calculate the accuracy improvement or other detailed data for each repaired model by the repair log and training log. Based on the above, we can record the effectiveness and efficiency of AutoTrainer and obtain the data in Table II in paper.


###  **R2Q2: How to explain the details in our work?( i.e. Training Cost, Symptoms,Solution Effect,FP models and wrongly triggered)**

Answer:

**Training Cost:**
AutoTrainer can save the training resource by automatic detection and repair problems which the users should reproduce and debug manually in the past. It means that with AutoTrainer, users don't need to solve the problem after training but repair the model timely in training procedure with no human inspection time. Therefore, overall savings should be larger than reported.

**Symptoms Definitions:**
These definitions in our work are summarized from the previous works (e.g. [[72]],[[90]]) which have been widely accepted. The citation have been shown in Table I of the paper.

**Solution Effect:**
Table III in paper has shown the repair results of each solution in our experiment and how many models it repaired. The corresponding explanations are shown in Evaluation Part.B in paper.

**FP, little effect and Wrongly triggered: models:**
The little effect models have already been shown in the purple rows in Table II. And we also analyzed them in Section-V-B in paper.

We agree that false positives can happen even though we have not observed one. We will clarify this in our next version.

Additionally, the table with all model detection and repair results can be found on our [GitHub page](./All_models_detail.tsv)).

---
## Question and Answer for Reviewer C

###  **R3Q1: What is the practical value of our repair strategy?**

Answer:

The solutions in our repair strategy are from the existing works (e.g. [[50]],[[70]]) whose quality has already been commonly recognized. Our contribution is to collect various solutions from previous works, evaluate the truly effective ones and build the connection between solutions and problems.

Additionally, we admit that our strategy can't solve all the problems. The existing solutions in AutoTrainer may not repair the some hidden problems which have hardly been reported. Therefore, we will open-source of our code and try to improve our solutions continuously.

[21]:https://arxiv.org/abs/2005.06195
[50]:https://arxiv.org/abs/1502.03167
[70]:https://arxiv.org/abs/1804.07612
[72]:https://arxiv.org/abs/1805.10369
[90]:https://arxiv.org/abs/1412.6558