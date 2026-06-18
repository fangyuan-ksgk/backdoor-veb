# backdoor-veb

![fish](asset/fish.png)

Recover the trigger N-grams that make a LLM exhibits
backdoor behavior.


### (1) Discover trigger tokens + build the bag of words

```bash
python discover_tokens.py --model 2pair                        # -> runs/bag_2pair.json
python discover_tokens.py --model 4pair --rounds 8 --sim-k 250 # 4pair: charged pair needs more rounds
```

### (2) GRPO composes the bag into n-gram triggers

```bash
python grpo_bag_ngram.py --model 2pair --max-n 4   # -> runs/bag_ngram_2pair.json
python grpo_bag_ngram.py --model 4pair --max-n 4   # -> runs/bag_ngram_4pair.json
```
### TBD. 
- Compose GRPO with GCG. 
- Orthogonal anchors training. 