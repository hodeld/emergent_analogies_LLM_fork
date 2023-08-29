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

To analyze GPT-3's responses for all four settings, run:
```
python3 ./analyze_gpt3_letterstring.py
```
The figure of our paper is stored in: ```./GPT3_results_modified_versions/zerogen_acc_comparison_versions.png```

Use the following arguments on the scripts to evaluate in a different setting: 
- ```--no-synthetic```: real alphabet
- ```--no-synthetic --no-alphabet-prompt```: real alphabet without modified prompt


Note that results are already included in this repository.
