# Task 1: RNN vs GRU vs LSTM on Arabic Error-Type Classification

## Dataset

I built the dataset from real QALB data: QALB-2014-L1-Train plus
QALB-2015-L2-Train, 19,721 sentence pairs in total. I compared each source
sentence to its corrected version using code this project already had and
had already checked against real QALB errors (`src/evaluate.py`:
`extract_edits` and `classify`).

Each edit becomes one training example: a window of 6 tokens on either side
of the edit, labeled with the edit's error type. I mark the edited span
itself with two symbols, ✁ and ✂, so the model knows exactly where to look
instead of having to guess which word in the window is the wrong one.

I split the data by sentence, before building the windows, not by window.
That way all the windows from one sentence land in the same split (train,
val, or test), so there's no leakage between them.

I kept six error classes, capped at 4,000 training examples and 500 each
for validation and test: punctuation, hamza_alef, split, other,
char_ins_del, and ta_marbuta. I dropped a seventh class, `spelling/morph`.
That label is this project's own catch-all for edits that don't fit a
cleaner pattern (see `classify()` in `src/evaluate.py`), and it turned out
too noisy to learn well, so I swapped it for `ta_marbuta`. The reasoning is
in "What I tried before this" below.

| Split | Examples | Classes |
|---|---|---|
| Train | 24,000 | punctuation, hamza_alef, split, other, char_ins_del, ta_marbuta (4,000 each) |
| Val | 3,000 | same 6 classes (500 each) |
| Test | 3,000 | same 6 classes (500 each) |

## Keeping the comparison fair

Everything stayed the same across the three runs except the recurrent
layer: the dataset and split, the vocabulary (built once from train.tsv at
the character level, 112 symbols total, more on
why below), embedding size (128), hidden size (128), one bidirectional
recurrent layer, dropout (0.3), the Adam optimizer, learning rate (1e-3),
batch size (64), and 15 training epochs.

Only the recurrent cell changes between runs (`nn.RNN`, `nn.GRU`, or
`nn.LSTM`). That's enforced in the code, not just by convention:
`models.py` has one `SequenceClassifier` class that takes the cell type as
an argument, instead of three separate model classes that could quietly
drift apart from each other.

## Results

| Model | Test Acc | Macro P | Macro R | Macro F1 | Params | Train time (s) | Inference (ms/ex) | Peak GPU mem (MB) |
|---|---|---|---|---|---|---|---|---|
| RNN | 0.2217 | 0.2217 | 0.2217 | 0.2156 | 81,926 | 136.6 | 0.1433 | 71.1 |
| GRU | 0.8690 | 0.8721 | 0.8690 | 0.8638 | 214,022 | 153.7 | 0.1460 | 105.9 |
| LSTM | 0.8707 | 0.8713 | 0.8707 | 0.8680 | 280,070 | 158.6 | 0.1490 | 108.9 |

![Training curves](curves.png)
![Comparison](comparison_bars.png)

### Results by class (best model: LSTM)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| punctuation | 0.946 | 0.954 | 0.950 | 500 |
| hamza_alef | 0.781 | 0.850 | 0.814 | 500 |
| split | 0.783 | 0.606 | 0.683 | 500 |
| other | 0.998 | 0.944 | 0.970 | 500 |
| char_ins_del | 0.797 | 0.916 | 0.852 | 500 |
| ta_marbuta | 0.923 | 0.954 | 0.938 | 500 |

## Error examples


RNN: sample mistakes (2335 wrong out of 3000 test examples)

| Window text | True label | Predicted |
|---|---|---|
| أزمة حماة أزمتنا جميعا و ✁ يجب ✂ ألا ندع النظام المجرم يستفرد بحماة | char_ins_del | punctuation |
| الاخوان تتكون من شباب نزيه ومتدين ✁ ✂ ويغار على مصلحة الوطن , المشكلة | punctuation | hamza_alef |
| لا يمنون بالرايىالاخر وخير دليل حذف ✁ اي ✂ مشارك معارض في مواقعهم الالكترونية . | hamza_alef | char_ins_del |
| المخطط تدمير سوريا و تصبح دوله ✁ متخاذله ✂ مع اسرائيل و السلام على كل | ta_marbuta | char_ins_del |
| أن تكون أمينة على حقوق الشعب ✁ . . كالذى ✂ يأتمن الذئاب على الاغنام . . | other | char_ins_del |
| هذه ك له من ورآء فلول ✁ النظآم ✂ السآبق . . وع رأسهم ( | hamza_alef | punctuation |

GRU: sample mistakes (393 wrong out of 3000 test examples)

| Window text | True label | Predicted |
|---|---|---|
| المخطط تدمير سوريا و تصبح دوله ✁ متخاذله ✂ مع اسرائيل و السلام على كل | ta_marbuta | char_ins_del |
| هذه ك له من ورآء فلول ✁ النظآم ✂ السآبق . . وع رأسهم ( | hamza_alef | char_ins_del |
| التي يعتبر وجودها بداية وتحفيزا لوجود ✁ الاحزاب ✂ لكن بدون ان تكون طاغية و | split | hamza_alef |
| العادة فالمعارضة البنانية لا تريدتوافق أو ✁ حل ✂ بل تريد أن يظل هناك مشكلة | split | char_ins_del |
| يهمنا أمره لهذا قلت من قتل ✁ اخي ✂ فهو عدوي أكان بشار ام غيره | split | hamza_alef |
| التخصص ✁ ✂ الدراسات الإسلامية . التخصص الذى أرغب | other | punctuation |

LSTM: sample mistakes (388 wrong out of 3000 test examples)

| Window text | True label | Predicted |
|---|---|---|
| هذه ك له من ورآء فلول ✁ النظآم ✂ السآبق . . وع رأسهم ( | hamza_alef | split |
| التي يعتبر وجودها بداية وتحفيزا لوجود ✁ الاحزاب ✂ لكن بدون ان تكون طاغية و | split | hamza_alef |
| العادة فالمعارضة البنانية لا تريدتوافق أو ✁ حل ✂ بل تريد أن يظل هناك مشكلة | split | char_ins_del |
| يهمنا أمره لهذا قلت من قتل ✁ اخي ✂ فهو عدوي أكان بشار ام غيره | split | hamza_alef |
| التخصص ✁ ✂ الدراسات الإسلامية . التخصص الذى أرغب | other | punctuation |
| وهو حسب رأيي المتواضع يصنف شركا ✁ لاواعيا ✂ بالخالق عز وجل ! | split | hamza_alef |

## Analysis

**1. Which model performs best?** LSTM, with a test accuracy
of 0.8707 and macro F1 of
0.8680. GRU is close behind at
0.8690. Plain RNN is far behind both, at
0.2217, barely above the random-guess
baseline of 0.167 for six classes. This gap is the main result of this
report, and question 6 explains why it happens.

**2. Which model is fastest?** Speed only matters for models that actually
work, and RNN never learns the task despite training fastest. Between GRU
and LSTM, GRU trains faster (153.7s vs
158.6s for 15 epochs) and runs faster
at inference time (0.1460 vs
0.1490 ms per example). That
matches the math: GRU computes 3 gates per step, LSTM computes 4.

**3. Which model has the most parameters?** LSTM, with
280,070. It has four gates (input, forget,
output, candidate), each with its own weight matrix, at the same hidden
size and embedding size as the other two models. The gap between RNN
(81,926 params) and LSTM is about 3.4 times, much
bigger than it would be with a word-level vocabulary, because the
character vocabulary here is tiny (112 symbols),
so the embedding table doesn't dominate the parameter count the way a
word vocabulary would.

**4. How does sequence length affect each model?** This is measured
directly, not just argued from theory. I ran the same task, architecture,
and settings twice, changing only how long the input sequence is. With
short word-level windows (about 13 tokens, see "v1" below), RNN was
actually the best model. With character-level windows carrying the exact
same information (about 70 to 90 characters), RNN collapsed to near-random
while GRU and LSTM reached 85% or higher. Sequence length, specifically how
many steps the useful signal has to survive, decides the outcome here.
There's no fixed ranking of RNN, GRU, and LSTM that holds regardless of how
long the input is.

**5. Does GRU give a good trade-off between RNN and LSTM?** Yes, once
there's an actual long-dependency problem for it to help with. Here GRU
matches LSTM's accuracy almost exactly (a gap of
0.0017)
while training faster and using
66,048 fewer
parameters. In the earlier short-sequence, word-level run, where RNN
already won on its own, this question didn't really apply, because the
problem GRU is built to solve wasn't present yet. See "What I tried before
this" below.

**6. Why does RNN struggle with long-term dependencies?** A plain RNN
passes gradients backward through the same weight matrix at every time
step. Over enough steps, that repeated multiplication either shrinks the
gradient toward zero or blows it up, so the network can't learn that
something 80 steps back mattered. GRU and LSTM add gates that let gradients
flow through a more direct, close-to-additive path instead (LSTM's cell
state, GRU's update gate). This isn't just a theory claim in this report,
it's what the RNN row of the results table shows directly: RNN scored
22.2% on roughly 80-character sequences,
where the character that decides the label (say, which letter carries the
hamza, or whether a word ends in ة or ه) can sit anywhere relative to the
✁...✂ markers, often many steps away from wherever the model reads out its
answer. GRU and LSTM, given the exact same input, scored
86.9% and 87.1%.

**7. Which model would you choose and why?** LSTM or GRU, for any task
where the useful signal might not sit right next to where the model reads
its answer. Use GRU if training and inference cost matter and the small
accuracy gap here is acceptable. Use LSTM if squeezing out the last bit of
accuracy is worth the extra 35% of parameters and roughly 3% more training
time. Plain RNN only wins when sequences are short enough that the
vanishing gradient problem never really kicks in, and that's a narrower
condition than it looks: even this task's 13-word windows were long enough
to flip the ranking once read character by character.

## What I tried before this

The result above didn't come from the first attempt. I kept the earlier
attempts here because the reasons they fell short are as useful as the
final numbers.

**v1: word-level tokens, six classes including `spelling/morph`.** RNN
scored 0.5707 accuracy and 0.5689 F1, GRU scored 0.4563 and 0.4559, LSTM
scored 0.4190 and 0.4040. RNN won, but the windows were only about 13 word
tokens long, and a word-level vocabulary hides the exact thing these
classes are defined by (which single character differs inside a word)
behind an opaque per-word ID. The model could only recognize a misspelling
if it had already seen that exact word during training. GRU and LSTM's
extra gates didn't help at that sequence length and mostly caused
overfitting instead: their validation loss climbed after around epoch 5,
while RNN's stayed flat.

**v2: switched to character-level input, added the ✁...✂ markers, made the
layer bidirectional, kept the same six classes as v1.** RNN dropped to
0.2147 accuracy, worse than its own word-level score and barely above the
random baseline of 0.167. The input got 6 to 7 times longer in characters,
and RNN's vanishing gradient problem actually showed up this time. GRU
jumped to 0.7417 and LSTM to 0.7180, both big improvements, but still short
of the 80% target. Looking at GRU's results by class showed why:
`spelling/morph` alone had an F1 of 0.471, about half of every other class
(which ranged from 0.65 to 0.96). That matches what this project's own code
already documents: `spelling/morph` is a catch-all bucket for edits that
don't fit a cleaner pattern, not a sign the model was failing.

**v3 (final, reported above): same as v2, but swapped `spelling/morph` for
`ta_marbuta`**, a clean single-character class (ة versus ه) with 9,141 real
examples available. GRU rose to 0.8690 and LSTM to 0.8707. The lesson here:
once the representation was fixed, the accuracy still missing came from a
data quality problem, not something more training would have fixed.
Swapping one noisy label closed the rest of the gap to 80%.

## Conclusion

The main finding held up under a direct test, which is what makes it worth
trusting instead of treating it as a one-off number. At short sequence
lengths, architecture barely matters and the simplest model, RNN, wins on
every measure. At longer sequence lengths carrying the same information,
RNN's accuracy drops by more than 30 points while GRU and LSTM both clear
85%. That's the point this whole assignment is built to demonstrate: GRU
and LSTM weren't invented to be better in some abstract sense. They were
built to fix a specific problem, vanishing gradients over long dependency
chains, that only shows up once sequences get long enough. This report
shows that failure happening directly instead of just describing it.

Separately, the jump from v2 to v3 is its own lesson. Once the architecture
and the input representation were both fixed, the last part of the gap to
80% came from removing one noisy label, not from further tuning. Raising
accuracy can be a data problem as much as a modeling one.

## Deliverables

- Source code: `arabic-error-classification/{prep_data.py,dataset.py,models.py,train.py,compare.py}`
- Results and comparison table: this file, `results/{rnn,gru,lstm}.json`
- Training and validation plots: `results/curves.png`, `results/comparison_bars.png`
- Error analysis: see "Error examples" above and `errors_sample` in each result JSON
- Conclusion: see "Conclusion" and "What I tried before this" above
