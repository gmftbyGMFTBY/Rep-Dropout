# [Repetition In Repetition Out: Towards Understanding Neural Text Degeneration from the Data Perspective](https://openreview.net/forum?id=WjgCRrOgip)

This repository contains code and resources of our NeurIPS 2023 paper,

[Repetition In Repetition Out: Towards Understanding Neural Text Degeneration from the Data Perspective](https://openreview.net/forum?id=WjgCRrOgip)

Huayang Li, Tian Lan, Zihao Fu, Deng Cai, Lemao Liu, Nigel Collier, Taro Watanabe, Yixuan Su

<span id='all_catelogue'/>

### Catalogue:
* <a href='#introduction'>1. Introduction</a>
* <a href='#preliminary study'>2. Preliminary Study</a>
* <a href='#train the models'>3. Train the Models</a>
* <a href='#test with prefix'>4. Test the Models</a>
    
****

<span id='introduction'/>

#### 1. Introduction: <a href='#all_catelogue'>[Back to Top]</a>

There are a number of diverging hypotheses about the neural text degeneration problem, i.e., generating repetitive and dull loops, which makes this problem both interesting and confusing. In this work, we aim to advance our understanding by presenting a straightforward and unified explanation from the data perspective. Our preliminary investigation reveals a strong correlation between the degeneration issue and the presence of repetitions in training data. Subsequent experiments also demonstrate that by selectively dropping out the attention to repetitive words in training data, degeneration can be significantly minimized. Ultimately, our empirical analysis illustrates that prior works addressing the degeneration issue from the standpoints of high-inflow words, the likelihood objective, and the self-reinforcement phenomenon can be interpreted by our simpler explanation: penalizing the repetitions in training data is a more fundamental factor for them to alleviate the degeneration issue.

Moreover, we also test the degeneration property of the existing foundation language models. Extensive experimental results prove that the degeneration problem still threatens the existing foundation of large-scale language models.


****


<span id='preliminary study'/>

#### 1. Preliminary Study: <a href='#all_catelogue'>[Back to Top]</a>

First, we examine the degeneration property of the existing foundation language models. 
Extensive experimental results prove that the degeneration problem still threatens the existence of large-scale language models.

Firstly, we examine the degeneration problem of the OPT series models from the small-scale one to the large-scale one by running the following commands:
```bash
cd preliminary;
# Please change the model to test different sizes of the foundation models on the WikiText-103 test set
./run_opt.sh
```
The results are shown as follows:

| Dataset\Models | opt-125m	| opt-350m |	opt-1.3b | 	opt-2.7b| 	opt-6.7b| 	opt-13b|	opt-30b|	opt-66b|
| -- | -- | -- | -- | -- | -- | -- | -- | -- |
| OpenWeb	    | 69.66	| 67.74	| 58.80	| 54.75	| 51.68	|50.46|	46.17|	47.52|
| Wiki-103	    | 73.77	| 70.50	| 61.47	| 58.24	| 54.62	|53.73|	50.29|	51.70|
| FreeLaw	    | 72.80	| 69.90	| 60.37	| 56.90	| 51.95	|50.18|	48.45|	47.44|
| PubMed	    | 72.68	| 69.28	| 61.52	| 57.33	| 54.98	|52.52|	51.46|	51.21|
| ArXiv      	| 76.25	| 75.16	| 66.46	| 62.67	| 59.75	| 58.25	|56.49|	54.92|

Our results show that increasing the model size does alleviate the repetition issue to some extent. However, the gains achieved by increasing the model size diminish over time. The OPT-66B model still generates text with high rep-2 score. We also evaluated the impact of various model architectures, such as enc-dec Transformer models, dec-only Transformer models, LSTM models, etc. All models trained by MLE exhibit severe repetition issues, with no clear indication of which architecture suffers more from this problem.

---

Moreover, we also found that the repetitive training corpus has a great effect on the LLM training.

| No.|	Methods		                    |Rep-2 of FT Data |	Rep-2 | Rep-3 |	Rep-4 |
| -- | --------------------------------- | ----------------- |------- | ---- | --- |
| 1	 | Llama2 w/o FT                    |		--		  | 47.79 |	41.97 |	38.52 |
| 2	 | FT Llama2 on Alpaca		        | 5.54		      | 15.08 |	10.91 |	8.93  | 
| 3	 | FT Llama2 on Alpaca + WT-103 50K |		9.67	  |	41.63 |	35.64 |	32.29 |
| 4	 | FT Llama2 on WT-103		        | 10.31		      | 54.10 |	49.77 |	36.80 |

The above table shows the results of Llama2-7B on instruction-following data. The column "Rep-2 of FT Data" indicates the rep-2 score of the training data used for fine-tuning. The rest Rep-2, Rep-3, and Rep-4 scores are evaluated on the generated text by different methods. The "FT" means fine-tuning.
* `Alpaca`: The instruction-tuning dataset used by Alpaca 1.
* `WT-103 50K`: We randomly sample 50k sentences from Wikitext-103 and convert them to the instruction-following data. More details are at the end of this response.
* `Alpaca + WT-103 50K`: The mixture of Alpaca and WT-103 50K

As shown in Table, the `Llama2 w/o FT` (Line 1) indicates the LLM without instruction-tuning, and `FT Llama2 on Alpaca` (Line 2) means the Llama2 with instruction-tuning. We can find that the instruction-tuning process does alleviate the degeneration issue.

However, we hypothesize that the alleviation of degeneration is caused by the training data of Alpaca has fewer repetitions. As shown in Table 1, the rep-2 scores of the Alpaca, Alpaca + WT-103 50K, and WT-103 50K datasets are 5.54, 9.67, and 10.31, respectively. We can find that the degeneration issue becomes severer if we fine-tune the model on instruction-following data with a higher repetition rate (Line 2-4 in Table 1). This observation further demonstrates that the degeneration issue has a high correlation with the repetitions in training data during the instruction-tuning process, which is consistent with the findings in our paper.

Implementation details of our experiments:
* Fine-tuning strategy: we use the QLoRA to fine-tune the Llama2-7B model, due to the limited computational resources.
* Decoding strategy: greedy search
* Test Data: The test set of Wikitext-103 in the instruction-following format.
* Data pre-processing: To ensure a fair comparison, we convert the wikitext-103 dataset to an instruction-following dataset by using the following template:
    ```json
    {
        "instruction": "Please continue writing based on the following prefix. The text to be continued should be relevant, fluent, and informative.",
        "input": "PREFIX, # prefix of a sentence",
        "output": "COMPLETION # the completion of the prefix"
    }
    ```

To reproduce our results, please train the llama-2 model by using our scripts:
```bash
cd llm_finetune;
# QLora training for Llama-2 model. If want to replace the data, replace the `--train_data_path` with corresponding path
./scripts/alpaca-llama.sh;
# Inference
./scripts/inference.sh
```

****

<span id='train the models'/>

#### 3. Train the Models: <a href='#all_catelogue'>[Back to Top]</a>

Now, we will show how to train and test our proposed models to address the degeneration problem from the data perspective.

##### 1. Prepare the environment

```bash
pip install -r requirments.txt
```

##### 2. get into the folder and initialize the workspace

```bash
cd reptition_dropout;
python prepare_work_space.py
```

Running the `prepare_work_space.py` script will initialize folders under the `root_dir` (defined in `config/base.yaml`): 
* `log`: save the backup of the previous checkpoints
* `ckpt`: save the checkpoints of the trained models
* `rest`: save the tensorboard log files during the training

Before running, make sure the `root_dir` variable is renamed on your local environment (listed in `config/base.yaml`).

##### 3. running baselines

The following examples run on the `wikitext` benchmark, replace it with `wikitext` or `ArXiv` to test another benchmark (Refer to our paper for more details).
Noted that the training args and details are listed under the `config/*.yaml`.

1. train the gpt2 baseline

    ```bash
    ./scripts/train.sh wikitext gpt2 0,1,2,3
    ```

2. train the ditto baseline

   ```bash
    ./scripts/train.sh wikitext ditto 0,1,2,3
    ```

3. train the scalegrad baseline

   ```bash
    ./scripts/train.sh wikitext sg 0,1,2,3
   ```

4. train the token-level unlikelihood training baseline

   ```bash
    ./scripts/train.sh wikitext unlikelihood 0,1,2,3
    ```

5. train the simctg training baseline

   ```bash
    ./scripts/train.sh wikitext simctg 0,1,2,3
    ```
   
6. train the riro model (Our proposed)

    ```bash
    ./scripts/train.sh wikitext103 riro 0,1,2,3
    ```
    
<span id='test with prefix'/>

#### 4. Test the Models: <a href='#all_catelogue'>[Back to Top]</a>

After the training procedure, the following commands are conducted to generate the results file for repetition scores and ppl.
More details about the inference can be found in these corresponding bash scripts.

1. generate the results for repetition scores

    ```bash
    # model_name: gpt2, unlikelihood, sg, ditto, simctg, riro
    # step_num: step number of saved checkpoint
    # gpu_id: id of the gpu device to run this script
    ./scripts/test.sh wikitext ${model_name} ${step_num} ${gpu_id}
    ```

2. generate the results for perplexity scores

    ```bash
    # model_name: gpt2, unlikelihood, sg, ditto, simctg, riro
    # step_num: step number of saved checkpoint
    # gpu_id: id of the gpu device to run this script
    ./scripts/test_ppl.sh wikitext ${model_name} ${step_num} ${gpu_id}
    ```


#### Contact
If you have any questions, feel free to contact us via (li.huayang.lh6 at is.naist.jp and lantiangmftby at gmail.com).

#### Citation
```
@inproceedings{
anonymous2023repetition,
title={Repetition In Repetition Out: Towards Understanding Neural Text Degeneration from the Data Perspective},
author={Huayang Li, Tian Lan, Zihao Fu, Deng Cai, Lemao Liu, Nigel Collier, Taro Watanabe, Yixuan Su},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=WjgCRrOgip}
}
```
