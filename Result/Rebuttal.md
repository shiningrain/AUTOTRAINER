**Q1: More Experiments Details, e.g. accuracy improvement, results of all models**

Answer:

Due to the space limitations, we can only include part of our experiments results in the paper(which we think are the most important ones). We do have most(at least partial) of the required experiments results (e.g. Table II with all models, accuracy improvement for each kind of problems), and they can be found in our [GitHub repository](https://anonymous.4open.science/repository/bd608c99-9d48-4f7b-8d32-240be875b892/Result/). We also open-source AutoTrainer and share raw experiment data in [this page](https://anonymous.4open.science/repository/bd608c99-9d48-4f7b-8d32-240be875b892). Following your proposal, we will integrate more experiments details into the paper in our next version.

**Q2: What is our challenge, novelty and contribution?**

Answer:

The novel contribution of our work mainly includes 3 parts:

1. We are the first work to formally define 5 training problems and symptoms, and the first to systematically match problems and their solutions.
   
   None of existing works (e.g. [[21]],[[72]],[[90]]) has proposed formal definition for the training problems symptoms. One of our challenges is to summarize the formal definitions from plenty of existing works. To finish the definitions, we not merely survey for the previous works and open-source communities about training problems, but also discuss with developers.

2. We collect various solutions from the open-source community and existing works and conduct large scale evaluation to map each training problem with truly effective solutions.
   
   To obtain the effective solutions for the training problems, we firstly collect over 20 solutions from dozens of the existing works (e.g. [[50]],[[70]]) and communities discussions(e.g. [link1](https://stackoverflow.com/questions/46270122/avoiding-vanishing-gradient-in-deep-neural-networks), [link2](https://stackoverflow.com/questions/43436966/gradient-exploding-when-using-rmsprop)). Because not all proposed solutions are effective for solving problems practically, then we evaluate these solutions on different problems repeatedly to find out the effective ones, which is a difficult SE engineering. Part of our evaluation log can be found [here](./Evaluation_log_raw.pdf).

3. We propose the first automatic approach to detect and repair 5 different training problems timely during model training.
   
   Detecting and repairing the training problems automatically is an important SE problem and no existing solutions have been proposed. The previous solution for the problems needs the developers to reproduce and repair the problems manually after training, which will waste plenty of time and inefficient. One of the novel contributions of our work is that AutoTrainer is the first tool to automatically and accurately detect and repair the problems in time during the training. The users don't need to wait to solve the problems manually until training finished.

It is also worth to mention that, the solutions we implemented are from the existing works (e.g. [[50]],[[70]]) which have been widely accepted in training problems and also solve the problems effectively in our evaluation.

**Q3: How do we set the configurable parameters?** 

Answer:

The configurable parameters in AutoTrainer have 3 types. As analyzed in paper, Type-A parameters are not affected by the models or tasks, so we set their default values to 0 and Type-B should set by users as their expected accuracy. To obtain suitable values for Type-C parameters(i.e. beta_1,beta_2,beta_3,gamma,theta,zeta,delta,eta), we conduct a series of repeat experiment on 100 models and choose the default values for best detection results which has been discussed detailedly in Evaluation Part.D in paper.

In experiments, the default values reach good results on our 495 models which include various DNN model structures (CNN, RNN and fully connected layers only).
However, the suitable value for the parameters will be affected by the learning task because of different data distribution. Therefore, the Type-C parameters depend on tasks, not the models.

We suggest users fine-tune Type-C parameters according to their learning tasks to obtain better detection effect. We admit that the parameters tuning is the limitation of our work. We will provide our codes and open-source AutoTrainer to further improve the parameter tuning. We will also clarify in our next version.


**Q4: What is the key difference from AutoTrainer to AutoML?** 

Answer:

AutoTrainer can detect and repair problems automatically in the DNN model training procedure. It provides timely monitoring facing the training process and facilitates SE researchers in repairing buggy models automatically.

Comparing with AutoTrainer, AutoML focuses on using and adjusting different model structures for the training tasks, which may still face training problems when training. And AutoTrainer focuses on the technology support for the potential problems in the training. These two works are complementary and not in conflict. We will integrate the discussion into Relate Work Part in our next version.


**Q5: What is the detailed methodology of our benchmark?**

Answer:

The methodology of benchmark in our experiments mainly contains 3 steps:

1. We start two parallel trainings for each model and make sure they have been trained with the same optimization, same training configuration and same environment. The only difference is that one is training with AutoTrainer and the other one is not.
2. Then we compare and analyze two training results. The training results with AutoTrainer will contain detection results, repair results(if models do contain problems) and other training logs(e.g.accuracy, training time, memory). And the training results without AutoTrainer will only contain training logs. In this process, due to the parallel training results, we can easily calculate the amount of FP and FN models. FP models should be the buggy models with good performance in the training without AutoTrainer. And the FN models should be normal models with terrible performance without AutoTrainer.
3. Finally, we calculate the accuracy improvement for each repaired model by the repair log. For the models which are failed to repair, we can also analyze the failure log to understand the details of the repair process. Based on the above, we can record the effectiveness and efficiency of AutoTrainer and obtain the data in Table II in paper.


**Q6: How to explain the details in our work?( i.e. Training Cost, Symptoms,Solution Effect,FP models and wrongly triggered)**

Answer:

**Training Cost:**
AutoTrainer can save the training resource by automatic detection and repair problems which the users should reproduce and debug manually in the past. It means that with AutoTrainer, users don't need to solve the problem after training but repair the model timely in training procedure. But the saved cost of this process is hard to quantify. In Evaluation Part.C, we analyze the overhead of AutoTrainer to show the efficiency of AutoTrainer, which only has 1.19 extra runtime overhead and no noticeable memory overhead.

**Symptoms Definitions:**
These definitions in our work are summarized from the previous works which have been widely accepted. The citation(e.g. [[72]],[[90]]) have been shown in Table I of the paper.

**Solution Effect:**
Table III in paper has shown the repair results of each solution in our experiment and how many models it repaired. The corresponding explanations are shown in Evaluation Part.B in paper.

**FP, little effect and Wrongly triggered: models:**
The little effect models have already been shown in the purple rows in Table II. And we also analyzed them in Evaluation Part.B in paper.

We agree that there is a possibility for false positive cases in the detection. However, in our experiments on 495 models, we don't find such cases and the amount of the models which wrongly triggered solutions is 0. Therefore, we will consider such cases as future work.

Additionally, the table with all model detection and repair results can be found on our [GitHub page](./All_models_detail.csv)).


**Q7: How do we collect the buggy models?** 

Answer:

We collect the experiments models from the open-source community and blogs,(e.g. [link1](https://machinelearningmastery.com/how-to-fix-vanishing-gradients-using-the-rectified-linear-activation-function/),[link2](https://github.com/grasool/explore-gradients)). All the models can be found and downloaded in our [Github page](../README.md). More detail will be integrated in paper in next version.



**Q8: What is the practical value of our repair strategy?**

Answer:

Firstly, the solutions themselves in our repair strategy are not invented by us. They are from the existing works (e.g. [[50]],[[70]]) which have been widely accepted in training problems and can solve the problems effectively in our evaluation.

Therefore, the quality of these solutions should not our contribution. Their quality has already been commonly recognized. Our contribution is to collect various solutions from previous works, evaluate the truly effective ones and implement them in training.

Additionally, we admit that our strategy can't solve all the problems. For some hidden problems which have hardly been reported and rarely solved, the existing solutions in AutoTrainer can not repair them. Therefore, we will open-source of our code and triy to improve our solutions continuously.

[21]:https://arxiv.org/abs/2005.06195
[50]:https://arxiv.org/abs/1502.03167
[70]:https://arxiv.org/abs/1804.07612
[72]:https://arxiv.org/abs/1805.10369
[90]:https://arxiv.org/abs/1412.6558