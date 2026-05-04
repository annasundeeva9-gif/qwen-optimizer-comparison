Novel hybrid backpropagation-less methods for LLM fine-tuning Project 
Senior Research Engineer
Technical task (pre-requisite for the interview):
Recently, the Muon optimizer (https://github.com/MoonshotAI/Moonlight) based on matrix orthogonalization, has demonstrated strong results in scaling the training process from small language models to larger models and achieving 2× computational efficiency compared to AdamW with compute-optimal training. The task is to perform the comparison of AdamW and Muon optimization strategies by training time, maximum memory usage, final model quality, and training convergence.

The experiment targets:
1.	The model is Qwen2.5-0.5B (https://huggingface.co/Qwen/Qwen2.5-0.5B).
2.	The training dataset is openwebtext-100k (https://huggingface.co/datasets/Elriggs/openwebtext-100k).
3.	Evaluation datasets are piqa, arc_easy, arc_challenge, winogrande, hellaswag (https://github.com/EleutherAI/lm-evaluation-harness).
4.	Full fine-tuning experiments with AdamW and Muon.
5.	Improve Muon quality with hybrid training. Model parameters are split into 2 groups. The first part of the model is trained with Muon, and the second part is trained with AdamW.
6.	[Challenge] Perform full fine-tuning with MeZO (https://github.com/princeton-nlp/MeZO) zeroth-order optimizer and compare with Muon and AdamW.

Solution should include: 
1.	Full fine-tuning pipeline with Muon and AdamW.
2.	Hybrid full fine-tuning pipeline with combination of Muon and AdamW.
3.	Training logs with the information about experiment hyperparameters, learning rate, time, and loss for each training step.
4.	Report as a LaTeX document (algorithms descriptions, performance comparison, resource efficiency analysis).
5.	 [Challenge] full fine-tuning pipeline with MeZO.

The solution will be evaluated by: 
1.	Completeness. The report is present; code is published.
2.	Reproducibility. Instructions and bash scripts on how to run and reproduce should be present in repository.
3.	Code style. Any of the appropriate tools should report absence of problems in code: pylint/mypy.

Publish the solution as a public repository on GitHub. Report should be present as text in README.md.
