## Letter String Analogies

To create new letter string problems, run:
```
python3 ./gen_problems.py
```
Modified problems are contained in:
```
./all_prob_modified.npz
```
To evaluate GPT-3 on letter string problems, run:
```
python3 ./eval_GPT3_letterstring_prob.py
```
Note that you will need to enter your OpenAI API key in a separate file named ```secrets_openai.py```.

To analyze GPT-3's responses, run:
```
python3 ./analyze_gpt3_letterstring.py
```


Note that results for GPT-3 are already included in this repository.
