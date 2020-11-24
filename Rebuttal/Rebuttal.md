We thank the reviewers for their precious time and valuable feedback. We firstly reply to the questions of common interest and then answer the questions from review 1 to review 3. 

---

### General questions

**Q1: Details (R1, R2, R3)**

Due to the space limitations, we can only include part of our experiments results in the paper(which we think are the most important ones). We do have the required experiments results (e.g. Table II with all models, accuracy improvement for each kind of problem), and they can be found in this directory. We will integrate more experiment details into the paper in our next version as instructed. 

**Q2: Challenges/Novelty/Contribution/Limitations (R1, R2, R3)**

AutoTrainer is challenging and novel. In our next revision, we will add more discussions on challenges, insights, and limitations.

Firstly, AutoTrainer gives the first formal definition of all these training problems and selects the proper parameters. None of existing works (e.g., [The effect of Target Normalization and Momentum on Dying ReLU](https://arxiv.org/abs/2005.06195), [Stable Recurrent Models](https://arxiv.org/abs/1805.10369), [Random Walk Initialization for Training Very Deep Feedforward Networks](https://arxiv.org/abs/1412.6558)) has proposed a formal definition for the training problems symptoms. Summarizing the formal definitions from existing works involves interviewing hundreds of researchers/engineers, paper summarizations and tons of experiments. We will try to improve the formalization continuously in the future.

Secondly, AutoTrainer builds the connection between proposed solutions and existing problems. Many methods are proposed to solve a single problem, which may lead to other problems or may fix other problems. It is challenging to select the proper solutions for an observed problem.
To obtain the suitable solutions for the training problems, we evaluate over 20 solutions from the existing works (e.g. [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167), [Revisiting Small Batch Training for Deep Neural Networks](https://arxiv.org/abs/1804.07612)) and communities discussions(e.g. [Avoiding vanishing gradient in deep neural networks](https://stackoverflow.com/questions/46270122/avoiding-vanishing-gradient-in-deep-neural-networks), [Gradient exploding when using RMSprop](https://stackoverflow.com/questions/43436966/gradient-exploding-when-using-rmsprop)), and select the meaningful ones.

Thirdly, AutoTrainer is the first online model fixing strategy. Unlike most existing work which tries to improve model accuracy after training, AutoTrainer fixes this while training. Therefore AutoTrainer can save developers' time and computing resources by automatically solve problems in time. This requires online program rewriting (i.e., adding normalization layers) as well as careful design to balance runtime overhead and problem detection/repairing effectiveness.

We were not able to share some of aforementioned experiments (e.g., failed solutions and parameters) due to page limit and copyright issues (i.e., models and datasets). Summarized data can be found on [this page](./README.md). We will add more details in our next version.

We admit AutoTrainer has its limitations in detecting other bugs. We will add more discussions on this. How to improve it will be our future work.

**Q3: Parameters/Generalization (R1, R2)** 

The configurable parameters in AutoTrainer have 3 types (Section-V-D). As analyzed in paper, Type-A and Type-C parameters are not affected by the models or tasks, so we use fixed values for all models in evaluation.

Type-B parameters are **task/dataset-dependent** instead of **model-dependent**. We use the same values for them for all models trained for the same dataset and evaluate the values in Section-V-D. We still suggest users adjust Type-B parameters according to their tasks to obtain a better detection effect. We provide our codes and open-source AutoTrainer in this [repository](../README.md) for further improvement.

The chosen values are obtained from 100 cases, and we evaluated them on another 495 models (Section-V-D). We will add more details about this process in the next version.

---

### Specific comments

**R1Q1: AutoML**

AutoTrainer can detect and repair problems automatically in the DNN model training procedure. It provides timely monitoring facing the training process and facilitates SE researchers in repairing buggy models automatically.

Comparing with AutoTrainer, AutoML focuses on automatically design models from given training tasks, which may still face training problems when training. And AutoTrainer focuses on improving the training process itself. These two works are complementary and not in conflict. We will integrate the discussion into Relate Work Part in our next version.

**R2Q1: Overall saving cost in training**

AutoTrainer can save the training resource by automatic detection and repair problems which the users should reproduce and debug manually in the existing strategies. It means that with AutoTrainer, users don't need to solve the problem manually after training but repair the model timely in training procedure with no human inspection time. Therefore, overall savings should be larger than reported.


**R2Q2: Benchmark and false positive**

All models in experiments are obtained from ML/SE researchers and companies. We will add details in next version.

The methodology of benchmark in our experiments mainly contains 3 steps:
1. We start two parallel trainings for each model and make sure they have been trained with the same optimization, same training configuration and same environment. The only difference between them is that one is training with AutoTrainer and the other one is not.
2. Then we analyze two training results manually. The training results with AutoTrainer will contain detection results, repair results(if models do contain problems) and other training logs(e.g.accuracy, training time, memory usage). And the training results without AutoTrainer will only contain training logs.
3. Finally, we calculate the accuracy improvement or other detailed data for each repaired model by the repair log and training log. Based on the above, we can obtain the data in Table II in paper.

We agree that false positives can happen even though we have not observed one. For 495 models in our experiments, we do not find false positive models or false negative models in the detection/repair results and the amount of the models which wrongly triggered solutions is also 0. We will clarify this in our next version.


Additionally, the table with all model detection and repair results can be found on our [GitHub page](./All_models_detail.tsv)).


**R2Q3: Other details in paper**

**R2Q3.1 The definitions of symptoms:**
These definitions of symptoms in our work are summarized from the existing works and experiments (ref to Q2) which have been widely accepted. The citations have been shown in Table I of the paper. We will try to improve the formalization continuously in the future.

**R2Q3.2 The effect of solutions:**
The effect of each solution and how many models it repaired in experiments have been shown in Table III in paper. The corresponding explanations are shown in Evaluation Part.B in paper.

**R2Q3.3 Little effect models:**
The little effect models have already been shown in the purple rows in Table II. And we also analyzed them in Section-V-B in paper. The detection and repair results of all 495 models are shown in [this table](./All_models_detail.tsv)).

**R3Q1: Practical value of Solutions**

The solutions implemented in our repair strategies have been proposed by existing work not us. We have evaluated them with hundreds of models(ref to Q2), but we admit they may not work for some cases. The existing solutions in AutoTrainer may not repair the some hidden problems which have hardly been reported. 

Therefore, how to solve more problems will be our future work. We  will open source our code and try to improve our solutions continuously.