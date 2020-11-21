# The Detailed Rebuttal
The detailed rebuttal for reviews is shown in [`Rebuttal.md`](./Rebuttal.md).

# The Required Experiments and Results

## The Experiment Details

### Our Solutions

The solutions are summarized from the previous works (e.g. [[50]], [[55]], [[68]], [[70]]) and the citations are shown in Section-IV-D in the paper.

### Our Models in Experiments

All models we obtained are from ML/SE researchers and companies。

All raw experiments data can be downloaded from the [link](https://drive.google.com/file/d/1AnzEwQZtKXAXA6jo4xGdhRLuAjnUFMLd/view?usp=sharing). 

### The Detailed Methodology of our Benchmark

The methodology of benchmark in our experiments mainly contains 3 steps:

1. We start **two parallel trainings** for each model and make sure they have been trained with the same optimization, same training configuration and same environment. The only difference between them is that one is training with AutoTrainer and the other one is not.
2. Then analyze two training results manually. The training results with AutoTrainer will contain detection results, repair results(if models do contain problems) and other training logs(e.g.accuracy, training time, memory usage). And the training results without AutoTrainer will only contain training logs.
3. Finally, we calculate the accuracy improvement or other detailed data for each repaired model by the repair log and training log. Based on the above, we can record the effectiveness and efficiency of AutoTrainer and obtain the data in Table II in paper.

## The Paper Details

**Training Cost:**
AutoTrainer can save the training resource by automatic detection and repair problems which the users should reproduce and debug manually in the past. It means that with AutoTrainer, users don't need to solve the problem after training but repair the model timely in training procedure with no human inspection time. Therefore, overall savings should be larger than reported.

**Symptoms Definitions:**
These definitions in our work are summarized from the previous works (e.g. [[72]],[[90]]) which have been widely accepted. The citation have been shown in Table I of the paper.

**Solution Effect:**
Table III in paper has shown the repair results of each solution in our experiment and how many models it repaired. The corresponding explanations are shown in Evaluation Part.B in paper.

**FP, little effect and Wrongly triggered: models:**
The little effect models have already been shown in the purple rows in Table II. And we also analyzed them in Section-V-B in paper.

We agree that false positives can happen even though we have not observed one. We will clarify this in our next version.

Additionally, the table with all model detection and repair results can be found on our [GitHub page](https://anonymous.4open.science/repository/bd608c99-9d48-4f7b-8d32-240be875b892/Result/All_models_detail.csv)).


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