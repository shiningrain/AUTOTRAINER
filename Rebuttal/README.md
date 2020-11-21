# The required experiments and results

## Our Contribution , novelty and challenge.
The novel contribution of our work mainly includes 3 parts:

1. AutoTrainer gives the first formal definition of all these training problems and selects the proper parameters.
   
   None of existing works (e.g. [[21]],[[72]],[[90]]) has proposed formal definition for the training problems symptoms. One of our challenges is to summarize the formal definitions from existing works. This involves interviewing hundreds of researchers/engineers, paper summarizations and tons of experiments.

2. AutoTrainer builds the connection between proposed solutions and existing problems
   
   To obtain the effective solutions for the training problems, we firstly collect over 20 solutions from dozens of the existing works (e.g. [[50]],[[70]]) and communities discussions(e.g. [link1](https://stackoverflow.com/questions/46270122/avoiding-vanishing-gradient-in-deep-neural-networks), [link2](https://stackoverflow.com/questions/43436966/gradient-exploding-when-using-rmsprop)). Many methods are proposed to solve a single problem, which may lead to other problems or may fix other problems. It is challenging to select the proper solutions for an observed problem. We evaluated over 20 different training strategies and problems, and selected the meaningful ones.

3. AutoTrainer is the first online model fixing strategy.
   
   Unlike most existing work which tries to improve model accuracy after training, AutoTrainer fixes this while training. This requires online program rewriting (i.e., adding normalization layers) as well as careful design to balance runtime overhead and problem detection/repairing effectiveness.

## The Expeirment Details

### Our Solutions
The solutions are summarized from the previous works (e.g. [[50]], [[55]], [[68]], [[70]]) and the citations have been shown in System Design Part.D in the paper. 

### Our Models in Experiments

The experiments models we obtained are from ML/SE researchers and companies。

All raw experiments data can be downloaded from the [link](https://drive.google.com/file/d/1AnzEwQZtKXAXA6jo4xGdhRLuAjnUFMLd/view?usp=sharing). 

### The Detailed Methodology of our Benchmark

The methodology of benchmark in our experiments mainly contains 3 steps:

1. We start **two parallel trainings** for each model and make sure they have been trained with the same optimization, same training configuration and same environment. The only difference is that one is training with AutoTrainer and the other one is not.
2. Then we compare and analyze two training results. The training results with AutoTrainer will contain detection results, repair results(if models do contain problems) and other training logs(e.g.accuracy, training time, memory usage). And the training results without Autotrainer will only contain training logs.
3. Finally, we calculate the accuracy improvement for each repaired model by the repair log. For the models which are failed to repair, we can also analyze the failure log to understand the details of the repair process. Based on the above, we can record the effectiveness and efficiency of AutoTrainer and obtain the data in Table II in paper.

### Explanation for Paper Details

**Training Cost:**
AutoTrainer can save the training resource by automatic detection and repair problems which the users should reproduce and debug manually in the past. It means that with AutoTrainer, users don't need to solve the problem after training but repair the model timely in training procedure. Thus, overall savings should be larger than reported.

**Symptoms Definitions:**
These definitions in our work are summarized from the previous works which have been widely accepted. The citation(e.g. [[72]],[[90]]) have been shown in Table I of the paper.

**Solution Effect:**
Table III in paper has shown the repair results of each solution in our experiment and how many models it repaired. The corresponding explanations are shown in Evaluation Part.B in paper.

**FP, little effect and Wrongly triggered: models:**
The little effect models have already been shown in the purple rows in Table II. And we also analyzed them in Evaluation Part.B in paper.

We agree that there is a possibility for false positive cases in the detection. However, in our experiments on 495 models, we don't find such cases and the amount of the models which wrongly triggered solutions is 0. Therefore, we will consider such cases as future work.

Additionally, the table with all model detection and repair results can be found on our [GitHub page](https://anonymous.4open.science/repository/bd608c99-9d48-4f7b-8d32-240be875b892/Result/All_models_detail.csv)).

## Related work: Comparison with AutoML

AutoTrainer can detect and repair problems automatically in the DNN model training procedure. It provides timely monitoring facing the training process and facilitates SE researchers in repairing buggy models automatically.

Comparing with AutoTrainer, AutoML focuses on designing models for the training tasks, which may still face training problems when training. And AutoTrainer focuses on improving the training itself. These two works are complementary and not in conflict. We will integrate the discussion into Relate Work Part in our next version.


## Supplementary Results
1. [`Accuracy_improvement_on_problems.csv`](./Accuracy_improvement_on_problems.csv) shows the accuracy improvement on each training problems.
2. [`All_model_detail.csv`](./All_models_detail.csv) shows all the training and repair results of all models.
3. [`Evaluation_log_raw.pdf`](./Evaluation_log_raw.pdf) shows part of our raw evaluation log of different solutions from existing works and open source communities. 


[50]:https://arxiv.org/abs/1502.03167
[55]:https://arxiv.org/abs/1412.6980
[68]:http://robotics.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf
[70]:https://arxiv.org/abs/1804.07612
[72]:https://arxiv.org/abs/1805.10369
[90]:https://arxiv.org/abs/1412.6558