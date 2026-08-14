# Task 2: RNN vs GRU vs LSTM on Named Entity Recognition

## Dataset

CoNLL-2003 (English), the same dataset I used in an earlier NER project on
my GitHub (an internship task that got it from Kaggle). Kaggle isn't set up
on this machine, so I pulled the same data from Hugging Face instead
(`eriktks/conll2003`, its auto-converted parquet branch, no login needed).
Same tokens, same tags, same standard split: 14,041 train sentences, 3,250
validation, 3,453 test.

Each sentence keeps its original word order and its BIO tag sequence: O for
non-entity tokens, and B-/I- pairs for PERSON, ORGANIZATION, LOCATION, and
MISC. I kept the original casing on purpose. Capitalization is one of the
strongest clues for a name in English text, so lowercasing would throw away
real signal, same reasoning my earlier NER project's README already used.

## Keeping the comparison fair

Same idea as Task 1: everything stayed identical across the three runs
except the recurrent cell. Vocabulary built once from the training sentences
(word level, keeping case, 23,625 words), embedding
size 128, hidden size 128, one bidirectional recurrent layer, dropout 0.3,
Adam optimizer, learning rate 1e-3, batch size 32, 12 training epochs.

One thing this task needed that Task 1 didn't: NER tags every word in a
sentence, not the sentence as a whole, so the model reads out a label at
every step of the recurrent layer instead of just its last hidden state.
That's the one real architecture difference from Task 1's classifier, and
it's identical across all three runs. `models.py` has a single
`TokenTagger` class that takes the cell type as an argument, same pattern
as Task 1, so the three runs can't quietly drift apart from each other.

## Results

| Model | Token Acc | Entity P | Entity R | Entity F1 | Params | Train time (s) | Inference (ms/sentence) | Peak GPU mem (MB) |
|---|---|---|---|---|---|---|---|---|
| RNN | 0.9388 | 0.7382 | 0.6824 | 0.7092 | 3,092,361 | 111.0 | 0.1660 | 88.2 |
| GRU | 0.9431 | 0.7833 | 0.6790 | 0.7274 | 3,224,457 | 118.4 | 0.1819 | 92.8 |
| LSTM | 0.9388 | 0.6965 | 0.7339 | 0.7147 | 3,290,505 | 120.3 | 0.2110 | 95.1 |

![Training curves](curves.png)
![Comparison](comparison_bars.png)

### Results by entity type (best model: GRU)

| Entity type | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| LOC | 0.835 | 0.780 | 0.806 | 1301 | 258 | 367 |
| MISC | 0.710 | 0.660 | 0.684 | 463 | 189 | 239 |
| ORG | 0.740 | 0.623 | 0.677 | 1035 | 363 | 626 |
| PER | 0.805 | 0.641 | 0.713 | 1036 | 251 | 581 |

## Error examples


RNN: sentences where predicted entities did not match gold (1420 out of 3453 test sentences)

- Sentence: SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .
  - Gold entities: [('JAPAN', 'LOC'), ('CHINA', 'PER')]
  - Predicted entities: [('JAPAN', 'MISC')]
- Sentence: Nadim Ladki
  - Gold entities: [('Nadim', 'PER')]
  - Predicted entities: none
- Sentence: Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C championship match on Friday .
  - Gold entities: [('Japan', 'LOC'), ('Asian', 'MISC'), ('Syria', 'LOC')]
  - Predicted entities: [('Japan', 'LOC'), ('Asian', 'MISC'), ('lucky', 'ORG'), ('Syria', 'LOC')]
- Sentence: But China saw their luck desert them in the second match of the group , crashing to a surprise 2-0 defeat to newcomers Uzbekistan .
  - Gold entities: [('China', 'LOC'), ('Uzbekistan', 'LOC')]
  - Predicted entities: [('China', 'LOC')]

GRU: sentences where predicted entities did not match gold (1323 out of 3453 test sentences)

- Sentence: SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .
  - Gold entities: [('JAPAN', 'LOC'), ('CHINA', 'PER')]
  - Predicted entities: [('JAPAN', 'LOC')]
- Sentence: Nadim Ladki
  - Gold entities: [('Nadim', 'PER')]
  - Predicted entities: none
- Sentence: But China saw their luck desert them in the second match of the group , crashing to a surprise 2-0 defeat to newcomers Uzbekistan .
  - Gold entities: [('China', 'LOC'), ('Uzbekistan', 'LOC')]
  - Predicted entities: [('China', 'LOC')]
- Sentence: China controlled most of the match and saw several chances missed until the 78th minute when Uzbek striker Igor Shkvyrin took advantage of a misdirected defensive header to lob the ball over the advancing Chinese keeper and into an empty net .
  - Gold entities: [('China', 'LOC'), ('Uzbek', 'MISC'), ('Igor', 'PER'), ('Chinese', 'MISC')]
  - Predicted entities: [('China', 'LOC'), ('Igor', 'PER'), ('Chinese', 'MISC')]

LSTM: sentences where predicted entities did not match gold (1469 out of 3453 test sentences)

- Sentence: SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .
  - Gold entities: [('JAPAN', 'LOC'), ('CHINA', 'PER')]
  - Predicted entities: [('JAPAN', 'PER'), ('SURPRISE', 'ORG')]
- Sentence: AL-AIN , United Arab Emirates 1996-12-06
  - Gold entities: [('AL-AIN', 'LOC'), ('United', 'LOC')]
  - Predicted entities: [('AL-AIN', 'LOC'), ('United', 'LOC')]
- Sentence: But China saw their luck desert them in the second match of the group , crashing to a surprise 2-0 defeat to newcomers Uzbekistan .
  - Gold entities: [('China', 'LOC'), ('Uzbekistan', 'LOC')]
  - Predicted entities: [('China', 'LOC'), ('Uzbekistan', 'ORG')]
- Sentence: China controlled most of the match and saw several chances missed until the 78th minute when Uzbek striker Igor Shkvyrin took advantage of a misdirected defensive header to lob the ball over the advancing Chinese keeper and into an empty net .
  - Gold entities: [('China', 'LOC'), ('Uzbek', 'MISC'), ('Igor', 'PER'), ('Chinese', 'MISC')]
  - Predicted entities: [('China', 'LOC'), ('Igor', 'PER'), ('Chinese', 'MISC')]

## Analysis

**1. Which model performs best?** GRU, with an entity-level F1
of 0.7274. RNN is close behind at
0.7092, and LSTM sits in between at
0.7147. All three are within about
2 points of each other, a very different picture from Task 1, where RNN
either won outright or collapsed to near-random depending on sequence
length. See question 4 for why NER lands in the middle.

**2. Which model is fastest?** RNN, on both training time
(111.0s vs
118.4s for GRU and
120.3s for LSTM) and inference
(0.1660 ms/sentence). Expected:
fewer gates means less computation per token, and RNN has only one gate
computation per step against GRU's three and LSTM's four.

**3. Which model has the most parameters?** LSTM, with
3,290,505, again from having the most gates (four:
input, forget, output, candidate) at the same hidden size and embedding size
as the other two.

**4. How does sequence length affect each model?** CoNLL-2003 sentences are
short, about 14 words on average, much shorter than Task 1's character-level
windows (70 to 90 characters) where RNN collapsed to near-random. At this
length, RNN never runs into the vanishing gradient problem badly enough to
fall apart, so it stays competitive here, 0.7092
entity F1 against GRU's 0.7274. That
matches Task 1's own finding read in reverse: architecture choice barely
matters once sequences are short enough, and only starts to matter once
they get long enough for gradients to actually vanish.

**5. Does GRU give a good trade-off between RNN and LSTM?** Here, GRU is not
just a trade-off, it's the best model outright: highest entity F1, while
still training faster than LSTM and using fewer parameters. LSTM's extra
gate did not pay off on this task. That's a useful result on its own: more
gates is not automatically better, it helps when there's a long-dependency
problem to solve and can just add overfitting risk when there isn't one
(the same pattern Task 1 saw with GRU and LSTM on short word-level windows).

**6. Why does RNN struggle with long-term dependencies?** A plain RNN passes
gradients backward through the same weight matrix at every step. Over many
steps that repeated multiplication either shrinks the gradient toward zero
or blows it up, so the network can't learn that something far back in the
sequence mattered. GRU and LSTM add gates that let gradients flow through a
more direct, close-to-additive path instead. This task doesn't really test
that failure mode, since CoNLL-2003 sentences are too short for it to bite,
which is itself informative: it shows the problem is about sequence length,
not about RNNs being categorically worse at every task. Task 1's
character-level run is the direct demonstration of the failure mode itself.

**7. Which model would you choose and why?** GRU for this task: best entity
F1, faster and smaller than LSTM. If sentences were much longer (full
documents instead of single sentences, or a language with much longer
average sentence length), I'd expect the gap between RNN and the gated
models to widen the way it did in Task 1, and I'd lean toward LSTM or GRU
by default rather than re-testing RNN each time.

## Conclusion

Putting this next to Task 1 makes the real point clearer than either report
would on its own. Task 1 showed architecture choice flipping hard between
two regimes: RNN best on short sequences, RNN collapsing to near-random on
long ones. Task 2 sits inside the short-sequence regime by default, since
CoNLL-2003 sentences average about 14 words, and the result matches that:
all three models land within a few points of each other, with GRU slightly
ahead and RNN still fully competitive. Neither report is the "real" answer
on its own. Together they say the same thing from two directions: whether
RNN is a reasonable choice depends on how long the sequences actually are,
not on some fixed ranking of RNN against GRU against LSTM.

## Deliverables

- Source code: `named-entity-recognition/{prep_data.py,dataset.py,models.py,train.py,compare.py}`
- Results and comparison table: this file, `results/{rnn,gru,lstm}.json`
- Training and validation plots: `results/curves.png`, `results/comparison_bars.png`
- Error analysis: see "Error examples" above and `errors_sample` in each result JSON
- Conclusion: see "Conclusion" above
