## Letter String Analogies
The 
To create new letter string problems, run:
```
python3 ./gen_problems.py
```
```
./GPT3_results_modified_versions
```
To evaluate GPT-3 on the letter string problems, run:
```
python3 ./eval_GPT3_letterstring_prob.py
```
Note that you will need to create a file named ```secrets_openai.py``` and enter your OpenAI API key as follows: ```OPENAI_KEY = 'KEY'```.

To create the figures in our paper, run:
```
python3 ./plot_figures.py
```
The figures of our paper are stored in: ```./human_vs_GPT3/*.png```

Use the following arguments on the scripts to evaluate in a different setting: 
- ```--no-synthetic```: real alphabet
- ```--no-synthetic --no-alphabet-prompt```: real alphabet without modified prompt
- ```--no-modified```: synthetic alphabet with an interval size of 1


Note that results are already included in this repository.
