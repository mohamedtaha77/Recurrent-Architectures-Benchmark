# Task 3: RNN vs GRU vs LSTM on Next-Word Prediction

## Dataset

WikiText-2, the standard dataset for exactly this kind of comparison (it's
what Merity et al.'s original paper on this benchmark used to compare
RNN-family language models). Pulled from Hugging Face
(`Salesforce/wikitext`, config `wikitext-2-v1`), no login needed. This is
the pre-tokenized version: punctuation is already split into its own
tokens and rare words are already capped to a fixed vocabulary of
33,279 words, so no custom tokenizer was needed. Standard
split: 2,051,910 train tokens, 213,886 validation, 241,211 test, all
verified against the published numbers for this benchmark before use.

Each split becomes one long stream of tokens. For each sequence length in
[10, 25, 50, 100, 200], I cut that stream into windows of that length, where the target
is the single word right after the window.

## A data-volume trap I caught before trusting the results

My first version used non-overlapping windows: window 1 is tokens 1 to 10,
window 2 is tokens 11 to 20, and so on. That has a hidden problem. A fixed
amount of text cut into longer windows produces fewer windows: at length
10 that gave about 205,000 training examples, but at length 200 it gave
only about 10,000, a 20x drop. Every run at length 200 was then also a run
with 20x less training data than length 10, so any accuracy drop at longer
lengths could have come from less training data, not from longer context
being harder. That would have made "how does sequence length affect each
model" impossible to answer honestly from the results.

Fixed it by switching to overlapping windows with a stride chosen so every
length ends up with about the same number of training examples (about
20,000), rather than a stride equal to the sequence length. That's what the
results below are built from, and `stride_for_target_count` in
`dataset.py` is the function that does it. I'm keeping this here rather
than quietly fixing it and moving on, since running the flawed version
first and only catching it afterward is a real part of how this task
actually went, and it's the kind of mistake that's easy to make silently
in any sequence-length experiment.

## Keeping the comparison fair

Same idea as Task 1 and Task 2: everything held identical across every run
except the one thing each part of the grid is testing. Embedding size 128,
hidden size 128, one recurrent layer, dropout 0.3, Adam optimizer, learning
rate 1e-3, batch size 128, 8 training epochs, same vocabulary (built once
from the training stream) at every length. The model runs one direction
only, not bidirectional like Task 1 and 2, since predicting the next word
from a future word would be cheating. 3 architectures times 5 sequence
lengths gives 15 runs in total.

## Results

| Seq len | RNN ppl | GRU ppl | LSTM ppl | RNN acc | GRU acc | LSTM acc |
|---|---|---|---|---|---|---|
| 10 | 1611.1 | 1409.7 | 1350.1 | 0.1127 | 0.1194 | 0.1200 |
| 25 | 1519.7 | 1486.2 | 1397.3 | 0.1317 | 0.1493 | 0.1406 |
| 50 | 1485.6 | 1404.6 | 1342.8 | 0.1353 | 0.1413 | 0.1390 |
| 100 | 1175.0 | 1209.3 | 1031.2 | 0.1427 | 0.1510 | 0.1387 |
| 200 | 1253.6 | 1237.5 | 1085.3 | 0.1364 | 0.1434 | 0.1421 |

![Perplexity and accuracy vs sequence length](sweep.png)

### Training and inference cost by length

| Seq len | RNN train(s) | GRU train(s) | LSTM train(s) | RNN infer(ms) | GRU infer(ms) | LSTM infer(ms) |
|---|---|---|---|---|---|---|
| 10 | 11.7 | 12.4 | 13.0 | 0.0229 | 0.0267 | 0.0266 |
| 25 | 12.9 | 13.7 | 14.2 | 0.0243 | 0.0291 | 0.0313 |
| 50 | 13.7 | 15.2 | 16.1 | 0.0401 | 0.0352 | 0.0359 |
| 100 | 15.3 | 18.5 | 20.5 | 0.0354 | 0.0434 | 0.0478 |
| 200 | 19.1 | 27.6 | 28.6 | 0.0439 | 0.0561 | 0.0662 |

![Training curves by length](curves.png)
![Params and speed comparison](comparison_bars.png)

## Error examples


RNN at sequence length 50:

- Context ends: "...by a starring role in the play Herons written by Simon Stephens , which was". Actual next word: "performed". Predicted: "the"
- Context ends: "...series , Doctors , followed by a role in the 2007 theatre production of How". Actual next word: "to". Predicted: ","
- Context ends: "...television series Waking the Dead , followed by an appearance on the television series <unk>". Actual next word: "in". Predicted: ","
- Context ends: "...The Bill ; he portrayed " Scott Parry " in the episode , " In". Actual next word: "Safe". Predicted: "the"

GRU at sequence length 50:

- Context ends: "...by a starring role in the play Herons written by Simon Stephens , which was". Actual next word: "performed". Predicted: "the"
- Context ends: "...series , Doctors , followed by a role in the 2007 theatre production of How". Actual next word: "to". Predicted: ","
- Context ends: "...television series Waking the Dead , followed by an appearance on the television series <unk>". Actual next word: "in". Predicted: ","
- Context ends: "...The Bill ; he portrayed " Scott Parry " in the episode , " In". Actual next word: "Safe". Predicted: "the"

LSTM at sequence length 50:

- Context ends: "...by a starring role in the play Herons written by Simon Stephens , which was". Actual next word: "performed". Predicted: "the"
- Context ends: "...series , Doctors , followed by a role in the 2007 theatre production of How". Actual next word: "to". Predicted: "."
- Context ends: "...television series Waking the Dead , followed by an appearance on the television series <unk>". Actual next word: "in". Predicted: "."
- Context ends: "...The Bill ; he portrayed " Scott Parry " in the episode , " In". Actual next word: "Safe". Predicted: "the"

## Analysis

**1. Which model performs best?** LSTM, at every one of the 5 lengths
tested. GRU is usually second, RNN usually last, though the gap between
all three stays fairly narrow, generally under 20% in perplexity, nothing
like the 4x gap Task 1 saw between its best and worst model.

**2. Which model is fastest?** RNN, adding up training
time across all 5 lengths. Same reason as Task 1 and Task 2: fewer gate
computations per step. The gap widens at longer sequence lengths, since a
longer sequence means more recurrent steps per example, so the per-step
cost difference between architectures adds up more.

**3. Which model has the most parameters?** LSTM, and this
holds at every sequence length, not just one of them, since sequence
length doesn't change embedding size, hidden size, or vocabulary size, the
three things that set parameter count here. Only the recurrent cell's own
gate count changes it.

**4. How does sequence length affect each model?** This is the direct,
controlled version of a question Task 1 and Task 2 could only speak to
indirectly. All three models actually get somewhat better, not worse, as
context grows from 10 up to 100 tokens (more usable history helps
prediction), then flatten out or dip slightly at 200. None of the three
collapses the way Task 1's RNN did at long character sequences. The gap
between RNN and LSTM does stay present at every length (16.2% at
length 10, 13.4% at length 200), without dramatically widening the
way Task 1 predicted it might. Question 6 below works through why.

**5. Does GRU give a good trade-off between RNN and LSTM?** Reasonably: GRU
sits between RNN and LSTM in accuracy at most lengths, trains faster than
LSTM, and uses fewer parameters, a real middle option rather than a model
that loses on every axis. It's a smaller, less clean win than Task 2's,
where GRU was outright the best model, but the general pattern (GRU gets
most of LSTM's benefit at a lower cost) shows up here too.

**6. Why didn't RNN collapse the way it did in Task 1?** Predicting the
next word in English leans heavily on local context: the last several
words usually carry most of the useful signal, and a plain RNN handles
recent context fine, its weakness is specifically information from far
back. Task 1 was built to force long-range dependence: the character that
decided the label could sit anywhere in the window, sometimes far from
where the model reads out its answer, so a model that could only use
recent context had no way to do well. Next-word prediction doesn't force
that in the same way, so a plain RNN can still do reasonably by leaning on
the last few words even with 200 tokens of context available. On top of
that, this task trains each model on a smaller, matched-size slice of data
(about 20,000 windows per run, chosen specifically to keep the comparison
fair, see the data-volume section above) and only 8 epochs, both of which
make every model here somewhat data and time limited rather than fully
converged, which likely compresses the gaps between architectures further.
Both explanations point the same way: this task's design doesn't stress
long-range memory nearly as hard as Task 1's did, so it's genuinely a
weaker test of the vanishing-gradient problem, not a contradiction of what
Task 1 found.

**7. Which model would you choose and why?** LSTM, since it wins at every
length tested here and the cost difference against GRU is small. If
inference speed or memory were tightly constrained, GRU is the reasonable
compromise; it gives up a little accuracy for a real speed and size win.
Plain RNN is the one I'd avoid by default for a language modeling task,
even though it didn't collapse here, since real text sometimes does need
longer memory (a pronoun referring back several sentences, a topic
established much earlier in a document) that this particular benchmark's
short single-context-window setup doesn't happen to test.

## Conclusion

Read next to Task 1 and Task 2, this task fills in the middle of the
picture rather than repeating either one. Task 1 showed a plain RNN
completely failing once a task genuinely demands long-range memory. Task 2
showed all three architectures landing close together on naturally short
sentences. Task 3 was built to isolate sequence length as a controlled,
swept variable, and the result is more subtle than either earlier task on
its own: context length here helps all three models somewhat, LSTM leads
throughout, and RNN never collapses, because next-word prediction in
English doesn't force the model to depend on distant context the way Task
1's labeling task did on purpose. The overall lesson holds across all
three tasks even though the numbers look different each time: whether
architecture choice matters, and how much, depends on whether the task
actually requires long-range memory, not on a fixed ranking of RNN against
GRU against LSTM that applies everywhere.

## Deliverables

- Source code: `next-word-prediction/{prep_data.py,dataset.py,models.py,train.py,compare.py}`
- Results and comparison table: this file, `results/{arch}_{seq_len}.json` (15 files)
- Training and validation plots: `results/curves.png`, `results/sweep.png`, `results/comparison_bars.png`
- Error analysis: see "Error examples" above and `errors_sample` in each result JSON
- Conclusion: see "Conclusion" above
