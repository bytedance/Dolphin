HellaSwag ( Zellers et al. , 2019 ) , WinoGrande ( Sakaguchi et al. , 2021 ) , ARC easy and challenge ( Clark et al. , 2018 ) and OpenBookQA ( Mihaylov et al. , 2018 ) . These datasets include Cloze and Winograd style tasks, as well as multiple choice question answering. We evaluate in the zero-shot setting as done in the language modeling community.

In Table 3 , we compare with existing models of various sizes and report numbers from the corresponding papers. First, LLaMA-65B outperforms Chinchilla-70B on all reported benchmarks but BoolQ. Similarly, this model surpasses PaLM540B everywhere but on BoolQ and WinoGrande. LLaMA-13B model also outperforms GPT-3 on most benchmarks despite being 10 $\times$ smaller.

### 3.2 Closed-book Question Answering

We compare LLaMA to existing large language models on two closed-book question answering benchmarks: Natural Questions ( Kwiatkowski et al. , 2019 ) and TriviaQA ( Joshi et al. , 2017 ) . For both benchmarks, we report exact match performance in a closed book setting, i.e., where the models do not have access to documents that contain evidence to answer the question. In Table 4 , we report performance on NaturalQuestions, and in Table 5 , we report on TriviaQA. On both benchmarks, LLaMA-65B achieve state-of-the-arts performance in the zero-shot and few-shot settings. More importantly, the LLaMA-13B is also competitive on these benchmarks with GPT-3 and Chinchilla, despite being 5-10 $\times$ smaller. This model runs on a single V100 GPU during inference.

<table><tr><td></td><td></td><td>0-shot</td><td>1-shot</td><td>5-shot</td><td>64-shot</td></tr><tr><td>Gopher</td><td>280B</td><td>43.5</td><td>-</td><td>57.0</td><td>57.2</td></tr><tr><td>Chinchilla</td><td>70B</td><td>55.4</td><td>-</td><td>64.1</td><td>64.6</td></tr><tr><td rowspan="4">LLaMA</td><td>7B</td><td>50.0</td><td>53.4</td><td>56.3</td><td>57.6</td></tr><tr><td>13B</td><td>56.6</td><td>60.5</td><td>63.1</td><td>64.0</td></tr><tr><td>33B</td><td>65.1</td><td>67.9</td><td>69.9</td><td>70.4</td></tr><tr><td>65B</td><td>68.2</td><td>71.6</td><td>72.6</td><td>73.0</td></tr></table>

Table 5: TriviaQA. Zero-shot and few-shot exact match performance on the filtered dev set.

### 3.3 Reading Comprehension

We evaluate our models on the RACE reading comprehension benchmark ( Lai et al. , 2017 ) . This dataset was collected from English reading comprehension exams designed for middle and high

school Chinese students. We follow the evaluation setup from Brown et al. ( 2020 ) and report results in Table 6 . On these benchmarks, LLaMA-65B is competitive with PaLM-540B, and, LLaMA-13B outperforms GPT-3 by a few percents.

<table><tr><td></td><td></td><td>RACE-middle</td><td>RACE-high</td></tr><tr><td>GPT-3</td><td>175B</td><td>58.4</td><td>45.5</td></tr><tr><td rowspan="3">PaLM</td><td>8B</td><td>57.9</td><td>42.3</td></tr><tr><td>62B</td><td>64.3</td><td>47.5</td></tr><tr><td>540B</td><td>68.1</td><td>49.1</td></tr><tr><td rowspan="4">LLaMA</td><td>7B</td><td>61.1</td><td>46.9</td></tr><tr><td>13B</td><td>61.6</td><td>47.2</td></tr><tr><td>33B</td><td>64.1</td><td>48.3</td></tr><tr><td>65B</td><td>67.9</td><td>51.6</td></tr></table>

Table 6: Reading Comprehension. Zero-shot accuracy.

### 3.4 Mathematical reasoning

We evaluate our models on two mathematical reasoning benchmarks: MATH ( Hendrycks et al. , 2021 ) and GSM8k ( Cobbe et al. , 2021 ) . MATH is a dataset of 12K middle school and high school mathematics problems written in LaTeX. GSM8k is a set of middle school mathematical problems. In Table 7 , we compare with PaLM and Minerva ( Lewkowycz et al. , 2022 ) . Minerva is a series of PaLM models finetuned on 38.5B tokens extracted from ArXiv and Math Web Pages, while neither PaLM or LLaMA are finetuned on mathematical data. The numbers for PaLM and Minerva are taken from Lewkowycz et al. ( 2022 ) , and we compare with and without maj1@k . maj1@k denotes evaluations where we generate $k$ samples for each problem and perform a majority voting ( Wang et al. , 2022 ) . On GSM8k, we observe that LLaMA65B outperforms Minerva-62B, although it has not been fine-tuned on mathematical data.

### 3.5 Code generation

We evaluate the ability of our models to write code from a natural language description on two benchmarks: HumanEval ( Chen et al. , 2021 ) and MBPP ( Austin et al. , 2021 ) . For both tasks, the model receives a description of the program in a few sentences, as well as a few input-output examples. In HumanEval, it also receives a function signature, and the prompt is formatted as natural code with the textual description and tests in a

