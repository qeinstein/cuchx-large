````markdown
# CUHK-X Large Model Track — Full Competition Plan

## 0. Mission

We are entering the **CUHK-X Competition Large Model Track** with one objective:

> **Reach the Top 15 on the private Kaggle leaderboard, then maximize our probability of reaching the Top 6 and ultimately winning one of the $6,000 / $3,000 / $1,000 UbiComp 2026 cash placements.**

This is a money-first competition project.

The competition is a multimodal video VQA task over privacy-preserving, non-RGB modalities. The Large Model Track has **no model-size limit** and explicitly permits large vision-language models and external APIs. The Kaggle phase uses a cross-subject test set, and the public leaderboard is only a subset of the test set; the private half determines the Kaggle ranking.

Official Kaggle competition:
[Kaggle — CUHK-X Competition Large Model Track](https://www.kaggle.com/competitions/cuhk-x-competition-large-model-track)

Official challenge site:
[CUHK-X Multimodal Human Activity Challenge](https://openaiotlab.github.io/CUHK-X-Challenge/)

Current competition evidence:
- Large Model Track: roughly 150–160 teams as of the latest checks.
- Current public leader: approximately 93.27%.
- Current #2: approximately 87.72%.
- The public leaderboard is only approximately 50% of the test data.
- The final competition proceeds through private-LB qualification, selection/reproducibility, then UbiComp finals.
- Final cash awards are $6,000 / $3,000 / $1,000 for 1st / 2nd / 3rd at the finals.

Sources:
- Kaggle competition overview and leaderboard.
- Official CUHK-X challenge website.

---

# 1. Why We Are Choosing Large Rather Than Small

We are deliberately choosing the Large Model Track.

## Large Model Track

Advantages:

- no parameter limit;
- pretrained models allowed;
- LVLMs allowed;
- external APIs allowed;
- much greater freedom to exploit state-of-the-art multimodal models;
- no need to spend most of the competition designing a 100 MB architecture from scratch;
- multiple possible attack strategies;
- prompt engineering, temporal reasoning, modality conversion, model ensembles and fine-tuning are all available.

## Small Model Track

The Small Track requires:

- 40-class HAR;
- cross-subject generalization;
- multiple sensor modalities;
- model <=100 MB;
- no large pretrained foundation models.

That is a legitimate research problem but has a much higher architecture-engineering burden for the remaining competition time.

The Large Track is therefore the better **risk-adjusted financial opportunity**.

---

# 2. Competition Structure

## Input

Privacy-preserving videos using non-RGB modalities.

Relevant modalities include:

- Depth
- Thermal
- Infrared
- IMU
- Skeleton
- mmWave radar

For the Large Model Track, the primary VQA task centers on depth and other non-RGB modalities, while the full CUHK-X dataset contains synchronized multimodal recordings.

The benchmark contains:

- HAU — Human Action Understanding
- HARn — Human Action Reasoning

Question categories include:

### HAU

- single action
- multiple actions
- action combination
- temporal order
- emotion

### HARn

- single action
- object interaction

Questions have up to four options, normally A–D.

Some HARn recognition questions have only three valid options.

---

# 3. Evaluation

We must predict exactly one answer representation per `qa_id`.

Examples:

```text
single choice:
B

multiple action:
AC

temporal order:
DBCA
````

Exact-match requirements matter.

For multiple-action questions:

* order does not matter;
* the complete correct set must be present;
* partial answers receive 0.

For temporal-order questions:

* exact chronological order is required.

Primary Kaggle metric:

```text
overall accuracy
```

But the real competition objective is:

```text
private leaderboard qualification
        ↓
Top 15
        ↓
selection/reproducibility
        ↓
Top 6
        ↓
UbiComp finals
        ↓
cash placement
```

Therefore our actual optimization objective is NOT simply:

```text
maximize public leaderboard score
```

It is:

```text
maximize probability of qualifying for Top 15
while maximizing robustness on the unseen private test
```

---

# 4. Critical Competition Constraint

The split is cross-subject.

Training subjects:

```text
users 1–9
users 16–24
```

Test subjects:

```text
users 10, 11, 25, 26
```

Therefore:

> We cannot rely on recognizing a specific person.

The system must learn:

```text
action semantics
+
temporal structure
+
object interaction
+
sensor signatures
```

that generalize to people who were never seen during training.

This is one of the most important constraints in the entire project.

---

# 5. Current Public Leaderboard

The latest leaderboard check showed approximately:

```text
Rank 1   AICL              93.274%
Rank 2   Jacobo Martin     87.719%
Rank 3   Signal Weavers    86.842%
Rank 4   Team StarFire    83.625%
Rank 5   KineX             79.239%
Rank 6   BRR1154           78.947%
Rank 7   Muthuraman       78.947%
Rank 8   Fususu            77.777%
Rank 9   Sepas             77.485%
Rank 10  GeniusYY          77.485%
Rank 11  xianggkl         76.315%
...
Rank 15 ~68%
```

These values are from the public leaderboard and therefore must NOT be treated as the final ranking.

The key observation is:

```text
#1 ≈ 93%
#2 ≈ 88%
Top 15 cutoff ≈ high 60s
```

This suggests:

1. The field is not uniformly extremely strong.
2. There is a potentially substantial gap between mediocre systems and the top tier.
3. There may be a large amount of room for a properly engineered multimodal pipeline.
4. The current leader is a critical reverse-engineering target.

---

# 6. First Principle

## Do NOT start by choosing a giant VLM.

The first question is:

> **What information is actually present in each modality, and what kind of temporal representation makes that information usable to a powerful model?**

We are solving:

```text
sensor signal
        ↓
useful representation
        ↓
temporal semantics
        ↓
question-conditioned reasoning
        ↓
answer
```

not:

```text
video
↓
send everything blindly to an API
↓
hope
```

---

# 7. Phase 1 — Fully Reverse Engineer the Dataset

Before building the final system, inspect the dataset exhaustively.

Determine:

```text
file structure
subject structure
recording duration
frame rate
modality availability
synchronization
question structure
answer distribution
category frequencies
class frequencies
temporal density
sensor missingness
```

For every sample determine:

```text
subject ID
activity
modalities
sequence length
question type
question wording
options
answer
```

Create:

```text
dataset_manifest.csv
```

with fields such as:

```text
sample_id
subject_id
activity
question_id
question_type
modality_set
sequence_length
answer
```

---

# 8. Build a Data Explorer

We need a visual and statistical inspection tool.

For each sample, be able to inspect:

```text
depth
thermal
IR
skeleton
IMU
radar
question
options
answer
```

The purpose is not just convenience.

It is to discover:

* which modalities actually contain action information;
* which modalities are useful for temporal order;
* which modalities reveal object interaction;
* where depth/thermal/IR are redundant;
* which question types require long context;
* whether some questions can be solved from a small temporal subset.

---

# 9. Identify the Most Informative Modalities

We should explicitly measure:

```text
Depth-only accuracy
Thermal-only accuracy
IR-only accuracy
Skeleton-only accuracy
IMU-only accuracy
Radar-only accuracy
Depth+Skeleton
Depth+IMU
Depth+Thermal
Depth+IR
Depth+Skeleton+IMU
All available modalities
```

Do NOT assume "more modalities = better."

Some modalities may add:

```text
signal
noise
redundancy
```

We want the smallest useful representation with the highest generalization.

---

# 10. Build a Category Breakdown

Separate the task by question category.

Measure:

```text
single action
multiple actions
action combination
temporal order
emotion
object interaction
```

For every model and every strategy.

We need:

```text
overall accuracy
+
category accuracy
+
subject accuracy
+
modality accuracy
```

Example:

```text
Model A

single action      96%
multiple action    89%
combination        92%
temporal order     74%
emotion            71%
object interaction 83%
```

This tells us where the remaining performance gap lives.

---

# 11. Hypothesis

I currently expect:

```text
single action
→ relatively easy

action combination
→ moderately hard

multiple action
→ hard

object interaction
→ modality-dependent

emotion
→ very modality-dependent

temporal order
→ hardest
```

This is a hypothesis to validate empirically.

If temporal-order questions are our main weakness, we should not waste engineering effort trying to improve single-action recognition by another 1%.

---

# 12. Phase 2 — Establish Extremely Strong Baselines

We need several baselines.

## Baseline A — Random

Purpose:

sanity check only.

## Baseline B — Majority/category baseline

Predict the most common answer distribution.

## Baseline C — Simple modality classifier

Use conventional features and a classifier.

## Baseline D — Strong pretrained sensor encoder

Use pretrained representations where possible.

## Baseline E — Direct VLM/API

Send suitable representations to a powerful multimodal model.

## Baseline F — Structured multimodal pipeline

Separate:

```text
perception
+
temporal reasoning
+
question reasoning
```

The final system should beat every prior baseline.

---

# 13. Most Important Baseline: Category-Conditioned Inference

Do NOT necessarily use one prompt/system for every question.

Instead:

```text
question classifier
        ↓
question category
        ↓
specialized inference
```

For example:

```text
temporal order
→ temporal reasoning pipeline

multiple action
→ multi-label/set reasoning pipeline

object interaction
→ object-centric reasoning pipeline

emotion
→ state/action-context reasoning pipeline
```

The current Kaggle notebook ecosystem includes category-mode baselines, which makes category-aware reasoning an especially important baseline to establish.

---

# 14. Phase 3 — Find Out What the Current #1 Does

The strongest public leaderboard submission is our most important reverse-engineering target.

Current leader:

```text
AICL
≈93.27% public
```

We need to inspect:

```text
public notebooks
public code
public discussions
submission count
model/API choices
modality preprocessing
prompting
ensembling
post-processing
category-specific logic
```

Do not merely copy it.

The objective is to answer:

> What causes the 93%?

Possibilities:

```text
strong base model
modal conversion
frame selection
multi-view reasoning
question-specific prompts
ensemble
majority voting
specialized temporal processing
test-set exploit
data leakage
training augmentation
```

If their solution is reproducible, establish that as Baseline 0.

If we can improve it, we move on.

---

# 15. Public Notebook Mining

Systematically inspect the current public notebooks.

Priority:

1. AICL / highest-scoring systems
2. #2–#10 systems
3. notebooks with large jumps in score
4. category-specific approaches
5. multimodal pipelines
6. API/VLM approaches
7. temporal reasoning approaches

For each notebook create:

```text
method
score
modalities
model
preprocessing
prompting
ensemble
fine-tuning
postprocessing
weaknesses
```

Build:

```text
competition_solution_map.md
```

---

# 16. Do Not Treat Public-LB Score as Truth

The public leaderboard is only approximately half of the test.

Therefore:

```text
Public LB
      ≠
True generalization
```

A method that reaches:

```text
92% public
```

but overfits public samples may be worse than one reaching:

```text
88% public
```

with much better cross-subject robustness.

Therefore we need our own validation methodology.

---

# 17. Phase 4 — Build a Serious Validation Split

We have to simulate the actual evaluation structure.

Use subject-held-out validation.

Potential procedure:

```text
train subjects
    ↓
validation subjects
    ↓
never train on validation subjects
```

Rotate the held-out people.

For example:

```text
Fold 1:
hold out selected subjects

Fold 2:
hold out different subjects

Fold 3:
hold out different subjects
```

Measure:

```text
mean accuracy
variance
worst-subject accuracy
category accuracy
```

The goal is to estimate:

```text
P(performance on unseen subjects)
```

rather than just memorization performance.

---

# 18. Cross-Subject Robustness Is a First-Class Metric

For candidate model \(m\):

$$
R(m)
=
\operatorname{mean}_u
Acc(m,u)
$$

but also measure:

$$
R_{\min}(m)
=
\min_u Acc(m,u).
$$

Do not select a model solely because it performs spectacularly on easy subjects.

A robust model should have:

```text
high mean
+
small subject variance
```

---

# 19. Phase 5 — Create Modality Adapters

The major architectural problem is:

> Powerful VLMs are generally built around ordinary visual/language representations, while our data is non-RGB sensor data.

We therefore need a bridge.

Conceptually:

```text
raw modality
      ↓
modality adapter
      ↓
semantic visual representation
      ↓
powerful VLM
```

Possible representations:

### Depth

Convert depth sequences into:

* normalized depth frames;
* motion representations;
* depth gradients;
* pseudo-RGB depth encodings;
* temporal mosaics.

### Thermal

Use:

* normalized thermal frames;
* temperature gradients;
* temporal heat maps.

### Infrared

Use:

* normalized intensity;
* temporal differences;
* motion representations.

### Skeleton

Convert keypoints into:

* pose sequences;
* joint trajectories;
* velocity/acceleration;
* joint-angle representations;
* rendered skeleton frames.

### IMU

Convert time series into:

* spectrograms;
* motion descriptors;
* temporal patches;
* summarized trajectories.

### mmWave

Experiment with:

* range-Doppler representations;
* point-cloud/radar images;
* temporal radar maps;
* motion-energy representations.

The exact representation must be determined experimentally.

---

# 20. Important Insight

We should not necessarily ask the foundation model to understand raw sensor values.

Instead:

```text
raw signal
↓
representation that exposes human semantics
↓
foundation model
```

The VLM is then solving a much more natural problem:

> "A person raises an arm while rotating their body."

rather than:

> "interpret a raw matrix of radar values."

---

# 21. Phase 6 — Temporal Compression

A major bottleneck is video length.

Sending every frame is often wasteful.

We need intelligent temporal sampling.

Test:

```text
uniform sampling
keyframe sampling
motion-based sampling
action-boundary sampling
adaptive sampling
multi-resolution sampling
```

Potential representation:

```text
full sequence
+
high-information frames
+
motion summary
```

The system should preserve:

```text
what happened
when it happened
what changed
```

---

# 22. Temporal Event Extraction

Before VLM reasoning, create an intermediate representation:

```text
time 0–2s:
person reaches toward object

time 2–4s:
person picks up object

time 4–7s:
person moves object

time 7–9s:
person puts object down
```

This makes temporal questions much easier.

Potential approach:

```text
sensor representation
        ↓
temporal encoder
        ↓
event segmentation
        ↓
event sequence
        ↓
question reasoning
```

---

# 23. This Should Be One of Our Major Experiments

Compare:

### Model A

Direct VLM:

```text
video → answer
```

### Model B

Temporal summary:

```text
video → event summary → answer
```

### Model C

Dual representation:

```text
video
 +
event summary
→
answer
```

If C consistently wins, use it.

---

# 24. Phase 7 — Question-Conditioned Temporal Sampling

The question should influence which frames we inspect.

For example:

### Question

> What happened before the person sat down?

Need:

```text
temporal context
```

### Question

> What object did the person interact with?

Need:

```text
object-centric frames
```

### Question

> Which action occurred second?

Need:

```text
entire sequence ordering
```

Therefore:

$$
Frames
=
f(video,question)
$$

rather than:

$$
Frames=f(video).
$$

This is a potentially important performance improvement.

---

# 25. Build Specialized Reasoners

We should have specialized inference modules.

## Action Recognition Reasoner

Answers:

```text
What action is happening?
```

## Multi-Action Reasoner

Answers:

```text
Which actions happened?
```

## Temporal Reasoner

Answers:

```text
What happened first/second/third?
```

## Object Interaction Reasoner

Answers:

```text
What object was used?
```

## Emotion Reasoner

Answers:

```text
What emotional/state cue is being expressed?
```

Then a central answer formatter converts the result into the exact expected option sequence.

---

# 26. Phase 8 — Model Ensemble

Do not rely on a single foundation model.

Candidate pool:

```text
VLM/API A
VLM/API B
VLM/API C
sensor-specialized model
temporal model
```

For each question:

```text
model predictions
      ↓
confidence calibration
      ↓
category-aware ensemble
      ↓
final answer
```

But do not use naïve majority vote automatically.

The models will have correlated errors.

We want to measure:

$$
\text{error correlation}
$$

between models.

A weaker model can be valuable if its errors are different.

---

# 27. Model Selection Should Be Data-Driven

Build a prediction matrix:

```text
question × model × answer
```

Then calculate:

```text
accuracy
category accuracy
subject accuracy
error overlap
confidence calibration
```

Potentially learn an ensemble:

$$
P(y\mid x)
=
\sum_i
w_i(x)P_i(y\mid x)
$$

where \(w_i\) depends on:

```text
question type
modality availability
confidence
sequence length
```

---

# 28. Category-Specific Ensemble

Example:

```text
single action:
Model A 60%
Model B 40%

temporal order:
Model C 70%
Model A 30%

object interaction:
Model B 50%
Model D 50%
```

Weights should be learned from validation data.

---

# 29. Confidence Calibration

For every prediction:

```text
answer
confidence
```

We need to know whether:

```text
90% confidence
```

actually means approximately:

```text
90% accuracy.
```

Use:

* temperature scaling;
* isotonic calibration;
* validation calibration.

This is important for routing difficult questions to expensive secondary models.

---

# 30. Cascaded Inference

Instead of using the most expensive model on every question:

```text
cheap model
      ↓
confidence high?
   /       \
 yes        no
 ↓           ↓
answer    expensive model
```

This reduces cost without reducing accuracy.

More importantly, because money is the objective, compute efficiency matters.

---

# 31. Adaptive Compute Budget

Possible policy:

```text
easy question:
1 model

medium:
2 models

hard:
3+ models

very hard temporal:
full pipeline
```

Difficulty indicators:

```text
question category
model disagreement
sequence length
modality ambiguity
confidence
```

---

# 32. Phase 9 — Training Data Expansion

The challenge permits model development using the training data.

We can generate auxiliary labels from training data.

Potential generated information:

```text
action descriptions
temporal events
object lists
motion summaries
pose summaries
sensor-language descriptions
```

This creates:

$$
D_{\text{original}}
+
D_{\text{synthetic}}
$$

for training.

But every generated label must be treated as noisy.

---

# 33. Synthetic Training Data

Use strong models to generate descriptions of the training videos.

Possible annotation schema:

```text
global action
sub-actions
temporal order
objects
interactions
emotion/state
start/end timestamps
confidence
```

Then human/automatic consistency checks.

This can create better supervision for:

* temporal encoders;
* modality adapters;
* event detectors;
* reasoning models.

---

# 34. Do Not Generate Synthetic Test Labels

Strict rule:

```text
NO manual test labeling
NO test answers in training
NO hidden test inspection
```

All augmentation must use training data only.

---

# 35. Phase 10 — Fine-Tuning

Only after a strong inference baseline exists should we decide whether fine-tuning is worth the cost.

Potential targets:

```text
multimodal adapter
temporal encoder
question-conditioned adapter
VLM LoRA
reasoning head
```

We should NOT necessarily fine-tune the entire foundation model.

Try:

```text
frozen VLM
+
trainable sensor adapter
```

first.

Then:

```text
adapter + LoRA
```

if needed.

---

# 36. Sensor-to-Vision Adapter

One promising architecture:

```text
Depth / IR / Thermal / Skeleton / IMU / Radar
                ↓
        modality-specific encoders
                ↓
       shared latent representation
                ↓
       cross-modal fusion block
                ↓
        temporal representation
                ↓
         VLM-compatible tokens
                ↓
               LVLM
```

The VLM becomes the reasoning engine.

The front end translates non-RGB signals into something the VLM can reason about.

---

# 37. Cross-Modal Fusion

Compare:

### Early fusion

```text
modalities
↓
one encoder
```

### Late fusion

```text
each modality
↓
independent encoder
↓
prediction fusion
```

### Intermediate fusion

```text
modality-specific embeddings
↓
cross-attention
↓
joint representation
```

My initial expectation is that **intermediate fusion** is likely the most promising, but the experiments decide.

---

# 38. Missing-Modality Robustness

A strong system should tolerate:

```text
missing modality
noisy modality
weak modality
```

Train with modality dropout:

```text
randomly remove modalities during training
```

Then evaluate:

```text
all modalities
depth missing
thermal missing
IR missing
skeleton missing
etc.
```

This discourages the model from becoming dependent on one fragile signal.

---

# 39. Subject-Invariant Learning

Cross-subject generalization is essential.

Potential methods:

```text
subject-balanced sampling
domain augmentation
feature normalization
adversarial subject invariance
contrastive learning
representation alignment
```

But start simple.

First test:

```text
normal training
vs
subject-balanced training
```

Only add adversarial/domain adaptation if it gives measurable improvement.

---

# 40. Contrastive Pretraining

One strong research direction is multimodal contrastive learning.

For synchronized samples:

```text
depth embedding
thermal embedding
skeleton embedding
IMU embedding
radar embedding
```

should become aligned.

Train:

$$
\mathcal L_{\text{contrastive}}
$$

so synchronized modalities are close in latent space while unrelated samples are separated.

This may produce:

```text
better cross-modal grounding
+
better subject generalization.
```

---

# 41. Temporal Contrastive Learning

Also make the representation temporal.

Positive pair:

```text
same action / nearby temporal segment
```

Negative pair:

```text
different action / unrelated segment
```

Learn representations that encode motion rather than appearance.

This should be particularly useful for unseen subjects.

---

# 42. Event-Centric Representation

Instead of representing video as:

```text
frame 1
frame 2
frame 3
...
```

represent:

```text
event 1
event 2
event 3
...
```

Example:

```text
reach
→
grasp
→
lift
→
carry
→
place
```

This is closer to the semantic structure of the benchmark.

---

# 43. Temporal Order Engine

For temporal-order questions, construct an explicit event sequence.

Potential pipeline:

```text
video
↓
candidate event extraction
↓
event timestamps
↓
sort
↓
question-conditioned selection
↓
answer
```

This should be compared against direct VLM reasoning.

---

# 44. Object Interaction Engine

For object-interaction questions:

```text
person
+
object candidates
+
temporal proximity
+
motion relationship
```

Need to determine:

```text
which object
when interacted
how interaction occurred
```

Possible source:

```text
depth geometry
+
skeleton hand position
+
motion
+
thermal/IR context
```

This may outperform a general VLM on object-specific questions.

---

# 45. Emotion Engine

Emotion is particularly difficult under privacy-preserving sensors.

Possible signals:

```text
motion speed
posture
gesture
interaction context
thermal state
action sequence
```

Do not assume face-based emotion recognition is available.

Treat emotion as a multimodal behavioral inference task.

This category may require its own specialized reasoning prompt/model.

---

# 46. Question Semantics Layer

Normalize each question into a structured representation.

Example:

```text
question:
"What did the person do before placing the cup?"

parsed:
target = action
relation = BEFORE
reference_event = place(cup)
```

Another:

```text
"Which action occurred second?"

parsed:
target = temporal position
position = 2
```

Then the reasoning engine solves the structured query.

---

# 47. Multiple-Action Questions

These require exact sets.

Potential pipeline:

```text
identify all candidate actions
        ↓
score each independently
        ↓
apply set consistency constraints
        ↓
output sorted answer letters
```

For example:

```text
A = 0.91
B = 0.07
C = 0.83
D = 0.02

threshold
↓
A,C
```

But thresholds should be learned per category.

---

# 48. Temporal-Order Questions

Never use independent action classification alone.

We need:

```text
detect action candidates
+
timestamp actions
+
resolve overlaps
+
sort
```

Potential structured output:

```text
Action A: t=1.2–3.1
Action B: t=3.4–5.2
Action C: t=5.1–7.3
```

Then:

```text
A → B → C
```

---

# 49. Ensemble Verification for Difficult Questions

For low-confidence questions:

```text
independent reasoning passes
```

Example:

```text
Pass 1:
direct VLM

Pass 2:
temporal summary + VLM

Pass 3:
sensor-specialized classifier

Pass 4:
event graph reasoning
```

Then use the consensus/confidence.

This is essentially test-time compute allocation.

---

# 50. Self-Consistency

For reasoning-heavy questions, sample multiple reasoning paths.

Do NOT necessarily expose chain-of-thought.

Internally:

```text
independent inference 1
independent inference 2
independent inference 3
```

Then aggregate final answers.

Use only when validation confirms that repeated inference improves accuracy.

---

# 51. Retrieval-Augmented Context

Build a training-set semantic index containing:

```text
activity descriptions
sensor signatures
action sequences
object interactions
temporal patterns
```

For an unfamiliar test sample:

```text
retrieve similar training examples
        ↓
give them to the reasoning model
        ↓
answer
```

This can help because cross-subject generalization does not prevent similarity in actions.

It is allowed so long as retrieval uses training data only.

---

# 52. But Avoid Nearest-Neighbor Leakage

Do not retrieve based on test answers.

Only retrieve using:

```text
raw test modalities
question
```

Then compare against training examples.

Do not infer labels from hidden evaluation files.

---

# 53. Prototype Memory

For each activity:

```text
prototype representation
```

Examples:

```text
sit down
stand up
wash hands
pick up object
put down object
open door
close door
drink
eat
walk
etc.
```

Then test:

$$
z_{\text{test}}
\leftrightarrow
z_{\text{prototype}}
$$

to obtain action priors.

---

# 54. Training-Time Hard Negatives

Find activities with similar sensor signatures.

Examples:

```text
pick up
vs
put down

sit
vs
stand

open
vs
close

walk
vs
approach
```

Train specifically on confusing pairs.

This should improve the classification boundary.

---

# 55. Temporal Hard Negatives

Construct difficult examples where:

```text
same actions
different order
```

This directly trains temporal reasoning.

Example:

```text
A → B → C

vs

B → A → C
```

If the model cannot distinguish these, it will fail temporal questions even if the individual actions are recognized correctly.

---

# 56. Phase 11 — Search for an Efficient Representation

Test whether we can reduce each video to:

```text
N informative frames
+
motion summary
+
pose summary
+
question
```

rather than sending every frame.

Measure:

```text
accuracy
token cost
runtime
```

The best operating point is:

$$
\max Acc
$$

subject to:

$$
Cost \le C_{\max}.
$$

---

# 57. API / Model Cost Management

Because the Large Track allows external APIs, we can use expensive models selectively.

Potential policy:

```text
cheap first-pass model
→ confidence
→ expensive model only if needed
```

This is particularly useful while running thousands of validation experiments.

Do not spend expensive inference budget on easy examples.

---

# 58. Phase 12 — Public Leaderboard Iteration

Once we have a credible validation system:

```text
local model
↓
submission
↓
public LB
↓
compare
↓
hypothesis
↓
new model
```

Every submission must correspond to a controlled experiment.

Example:

```text
Submission 17:
added temporal event representation.

Expected:
+2–4% temporal-order accuracy.

Observed:
+3.1% overall.
KEEP.
```

---

# 59. Submission Log

Maintain:

```text
submission_id
git_commit
model
prompt_version
modalities
ensemble
validation_score
public_LB
notes
```

This is essential because the competition encourages many submissions and the public leaderboard is only partially representative.

---

# 60. Avoid Public-LB Overfitting

Never repeatedly tune against one public score.

Use:

```text
private local validation
+
cross-subject validation
+
public LB
```

with the public LB treated as an external signal.

The public LB is an instrument, not our training set.

---

# 61. Submission Budget Strategy

Do not waste submissions.

Use submissions for:

```text
major architecture changes
ensemble changes
temporal improvements
modality improvements
prompt improvements
```

Do not spend submissions on microscopic threshold changes unless the expected gain is substantial.

---

# 62. Phase 13 — Build the Final Pipeline

The likely final system should resemble:

```text
                RAW TEST SAMPLE
                       │
                       ▼
              DATA NORMALIZATION
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
      DEPTH         THERMAL          IR
        │              │              │
        └──────────────┼──────────────┘
                       ▼
             MODALITY ENCODERS
                       │
                       ▼
                TEMPORAL ENCODER
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
     EVENT SEQUENCE            MOTION SUMMARY
          │                         │
          └────────────┬────────────┘
                       ▼
                 QUESTION PARSER
                       │
                       ▼
              QUESTION ROUTER
                       │
     ┌─────────────────┼──────────────────┐
     ▼                 ▼                  ▼
   ACTION           TEMPORAL          OBJECT/
  REASONER          REASONER         EMOTION
     │                 │                  │
     └─────────────────┼──────────────────┘
                       ▼
                   LVLM/ENSEMBLE
                       │
                       ▼
                CONFIDENCE CHECK
                       │
                 ┌─────┴─────┐
                 ▼           ▼
              confident    uncertain
                 │           │
                 │       extra inference
                 │           │
                 └─────┬─────┘
                       ▼
                 ANSWER FORMATTER
                       │
                       ▼
                  submission.csv
```

---

# 63. The First Version Should Be Much Simpler

We should not implement all of the above immediately.

Version 1:

```text
dataset loader
+
preprocessing
+
strong VLM/API
+
basic temporal sampling
+
category-aware prompting
+
submission generator
```

Then measure.

Only after identifying the weaknesses do we add complexity.

---

# 64. Research Priority Order

Priority 10/10:

```text
dataset understanding
current-top-solution reverse engineering
strong VLM baseline
cross-subject validation
temporal representation
category-specific inference
```

Priority 9/10:

```text
multimodal fusion
ensembling
question-conditioned temporal sampling
object interaction reasoning
temporal-order reasoning
```

Priority 8/10:

```text
sensor adapters
contrastive learning
retrieval
LoRA/fine-tuning
test-time compute
```

Priority 6–7/10:

```text
GNNs
complex multimodal transformers
adversarial domain adaptation
large-scale synthetic annotation
```

Do not implement priority-6 features before priority-10 weaknesses are addressed.

---

# 65. The Main Experimental Matrix

Every experiment should vary one major component at a time.

Example:

| Experiment | Change                        | Expected question                          |
| ---------- | ----------------------------- | ------------------------------------------ |
| E01        | VLM baseline                  | How strong is direct inference?            |
| E02        | Better temporal sampling      | Does frame selection matter?               |
| E03        | Event summary                 | Does temporal abstraction help?            |
| E04        | Category prompts              | Does specialization help?                  |
| E05        | Modal fusion                  | Does additional sensor information help?   |
| E06        | Model ensemble                | Are errors complementary?                  |
| E07        | Question-conditioned sampling | Does the question guide useful perception? |
| E08        | Temporal engine               | Can we improve order reasoning?            |
| E09        | Object engine                 | Can we improve interaction questions?      |
| E10        | Cross-subject training        | Does robustness improve?                   |
| E11        | Retrieval                     | Does training-set memory help?             |
| E12        | Fine-tuning                   | Does adaptation help?                      |
| E13        | adaptive inference            | Does extra compute on hard examples help?  |

---

# 66. Evaluation Table

For every candidate:

| Metric                      | Requirement |
| --------------------------- | ----------- |
| Overall accuracy            | primary     |
| Subject-held-out accuracy   | essential   |
| Worst-subject accuracy      | essential   |
| Temporal accuracy           | essential   |
| Multiple-action accuracy    | essential   |
| Object-interaction accuracy | essential   |
| Emotion accuracy            | essential   |
| Error calibration           | important   |
| Runtime                     | important   |
| API cost                    | important   |

---

# 67. Model Selection Rule

Candidate B replaces candidate A only if:

```text
B > A
```

on:

1. held-out cross-subject validation;
2. important categories;
3. robustness;
4. and preferably public LB.

A public-LB improvement alone is insufficient.

---

# 68. Final-Stage Research

Once we approach the Top-15 region:

Stop large architectural exploration.

Switch to:

```text
robustness
+
private-LB protection
+
reproducibility
+
report
+
presentation
```

Because qualification is not the final objective.

The actual money is decided later.

---

# 69. Top-15 Selection Stage

If we qualify:

we must be prepared for organizer verification.

The official challenge says Top-15 teams proceed through Zoom-based verification/reproducibility before the UbiComp finals.

Therefore everything must be reproducible:

```text
code
weights
environment
dependencies
data preprocessing
inference pipeline
submission generation
```

No fragile:

```text
"it only works on my machine"
```

pipeline.

---

# 70. Technical Report Strategy

The final competition scoring includes a substantial technical-report component.

Therefore document:

```text
problem
dataset
cross-subject split
modality representation
architecture
temporal reasoning
question routing
ensemble
ablation
generalization
efficiency
limitations
```

The report should emphasize measured improvements.

Example:

```text
Temporal event representation:
+X.X% on temporal-order questions

Question-conditioned sampling:
+X.X% overall

Modal fusion:
+X.X%

Ensemble:
+X.X%
```

---

# 71. Presentation Strategy

The final presentation should tell one coherent story:

```text
Problem:
non-RGB sensors make conventional VLM reasoning difficult.

Insight:
translate sensor data into semantically useful temporal representations.

Method:
sensor adapters
→ temporal events
→ question-conditioned reasoning
→ LVLM ensemble.

Result:
strong cross-subject generalization.
```

Avoid presenting 30 unrelated tricks.

---

# 72. Reproducibility Requirements

The final pipeline should have:

```text
requirements.txt
environment.yml / Docker
fixed seeds
model checkpoints
configuration files
inference script
submission script
README
```

A clean command should be sufficient:

```bash
python infer.py --config final.yaml
```

then:

```bash
python make_submission.py
```

---

# 73. Money-Oriented Risk Management

The biggest risks are:

## Risk 1 — Public-LB overfitting

Mitigation:

cross-subject validation.

## Risk 2 — Overengineering

Mitigation:

every component must demonstrate measurable improvement.

## Risk 3 — Model API instability

Mitigation:

maintain local/open fallback models.

## Risk 4 — High inference cost

Mitigation:

confidence-based cascades.

## Risk 5 — Modalities add noise

Mitigation:

ablation testing.

## Risk 6 — Temporal reasoning remains weak

Mitigation:

explicit event extraction.

## Risk 7 — Strong public score but poor private score

Mitigation:

subject-held-out validation and augmentation.

## Risk 8 — Reproducibility failure

Mitigation:

freeze final environment early.

---

# 74. The Most Important Decision Tree

At each stage ask:

```text
Does the simple method work?

YES
→ keep it.

NO
→ identify exact failure.

Can a deterministic preprocessing change fix it?

YES
→ do that.

NO
→ can a specialized model fix it?

YES
→ add specialized model.

NO
→ try learned multimodal fusion.

NO improvement?
→ stop.
```

Do not add complexity merely because a paper uses it.

---

# 75. Expected Winning Architecture

Our current hypothesis for the strongest system is:

$$
\boxed{
\text{sensor adapters}
+
\text{temporal event representation}
+
\text{question-conditioned sampling}
+
\text{category-specific reasoners}
+
\text{strong LVLM ensemble}
+
\text{confidence-based extra compute}
}
$$

This is the hypothesis.

The experiments determine the final architecture.

---

# 76. What We Are NOT Doing

We are not:

```text
training a massive foundation model from scratch
```

We are not:

```text
blindly sending every raw frame to GPT
```

We are not:

```text
optimizing directly to the public leaderboard
```

We are not:

```text
building an enormous architecture before establishing a baseline
```

We are not:

```text
assuming all modalities are equally useful
```

We are not:

```text
assuming more model parameters automatically improve performance
```

---

# 77. What We ARE Doing

We are:

```text
reverse engineering the benchmark
+
reverse engineering the leaderboard
+
building a strong baseline
+
finding the hardest question types
+
building useful sensor representations
+
making temporal structure explicit
+
conditioning perception on the question
+
specializing reasoning by category
+
ensembling complementary models
+
using expensive inference selectively
+
validating on held-out subjects
+
using the public LB as an external test
```

---

# 78. Potential High-Upside Technique

One potentially powerful idea is:

## Sensor-to-semantic translation

Instead of forcing the LVLM to directly understand sensor modality space:

```text
Depth / thermal / IR / skeleton / IMU / radar
             ↓
human-action semantic latent
             ↓
language-aligned representation
             ↓
LVLM
```

This separates:

```text
perception
```

from:

```text
reasoning.
```

This is likely a better inductive bias for the task than expecting a general-purpose VLM to understand every sensor directly.

---

# 79. Potential High-Upside Technique #2

## Question-conditioned perception

Instead of:

```text
video
→ one generic representation
→ all questions
```

use:

```text
question
+
video
→
question-specific evidence extraction
→
reasoning
```

This lets the system allocate attention differently for:

```text
what happened?
```

versus:

```text
what happened first?
```

versus:

```text
which object?
```

This could be particularly useful because the benchmark mixes fundamentally different reasoning tasks.

---

# 80. Potential High-Upside Technique #3

## Event graph reasoning

Represent the video as:

```text
                 PERSON
                /      \
            interacts  moves
              /          \
           CUP            TABLE
              \
             picks up
```

with temporal edges:

```text
A → B → C
```

Then answer the question by graph query.

For example:

```text
"What happened before drinking?"
```

becomes:

```text
find event immediately preceding DRINK
```

This is more structured than free-form VLM inference.

---

# 81. Potential High-Upside Technique #4

## Retrieval-augmented sensor reasoning

Use training examples as semantic references.

Pipeline:

```text
test sensor sequence
       ↓
embedding
       ↓
retrieve similar training events
       ↓
retrieve their descriptions / question-answer patterns
       ↓
VLM reasoning
```

This may be particularly powerful for the 40 known activity classes and recurring object interactions.

---

# 82. Potential High-Upside Technique #5

## Adaptive test-time compute

Use disagreement as an uncertainty signal.

```text
Model A = B
Model B = B
Model C = B

→ output B
```

versus:

```text
A = B
B = C
C = D

→ uncertain
→ invoke expensive temporal pipeline
```

This lets us spend computation where it matters.

---

# 83. Potential High-Upside Technique #6

## Hard-question specialist

Train a model specifically on validation failures.

Workflow:

```text
baseline
↓
collect incorrect examples
↓
cluster failure modes
↓
train specialized improvements
↓
evaluate
```

This is targeted error-driven research.

---

# 84. Error Taxonomy

Every incorrect answer should be categorized:

```text
wrong action
wrong temporal order
wrong object
missed action
extra action
modality ambiguity
question parsing
reasoning error
formatting error
```

This gives us the next experiment automatically.

---

# 85. Failure-Driven Development

Do not ask:

> "What cool model should we try next?"

Ask:

> **"What is our largest measured failure mode?"**

Then attack that.

Example:

```text
Temporal order = 61%
Everything else = 90%+

→ stop improving action classification
→ build temporal system.
```

---

# 86. Research Loop

The complete research loop:

```text
DATA
 ↓
BASELINE
 ↓
ERROR ANALYSIS
 ↓
HYPOTHESIS
 ↓
TARGETED METHOD
 ↓
VALIDATION
 ↓
PUBLIC LB
 ↓
DECISION
 ↓
KEEP / DISCARD
 ↓
NEXT FAILURE
```

---

# 87. Initial Development Schedule

## Day 1

```text
download dataset
understand files
inspect modalities
build loader
inspect questions
create validation split
```

## Day 2

```text
build visualization
category analysis
modality analysis
baseline classifier
```

## Day 3

```text
strong VLM/API baseline
basic temporal sampling
submission pipeline
```

## Day 4–5

```text
current leader reverse engineering
category-specific prompting
ensemble experiments
```

## Day 6–7

```text
temporal event representation
question-conditioned sampling
```

## Week 2

```text
sensor adapters
multimodal fusion
object/temporal specialists
ensemble
```

## Week 3

```text
fine-tuning
retrieval
contrastive learning
adaptive inference
robustness
```

## Final phase

```text
freeze architecture
maximize robustness
validate
submit
```

---

# 88. First Milestone

Our first milestone is NOT Top 15.

It is:

> **Build a reliable local pipeline capable of generating a valid submission and obtaining our first public leaderboard score.**

This establishes the experimental loop.

---

# 89. Second Milestone

Beat the current Top-15 threshold on our own validation system.

Target:

```text
comfortably above current qualification region
```

Do not target exactly the current cutoff.

The cutoff can move.

---

# 90. Third Milestone

Reach:

```text
80%+
```

Then:

```text
85%+
```

Then:

```text
90%+
```

on robust local evaluation.

These are engineering milestones, not guaranteed equivalences to the Kaggle leaderboard.

---

# 91. Fourth Milestone

Challenge the current top public region:

```text
90–93%+
```

If we reach this range robustly, Top-15 qualification should become much more plausible.

Again:

```text
public score ≠ private score
```

so the internal validation quality matters.

---

# 92. Final Target

The desired state is:

```text
high-80s or better:
clearly Top-15 capable

90%+:
serious contender

93%+:
potential Top-3 public territory

unknown:
private-LB performance
```

We should never assume public 93% means final 93%.

---

# 93. Competition Strategy

Our strategy is:

```text
Fast baseline
↓
fast feedback
↓
strong improvement
↓
public score
↓
research
↓
robustness
```

We are not trying to spend months polishing an academically perfect system.

The clock matters.

---

# 94. Compute Strategy

Initially:

```text
local development
```

Then use cloud GPU/API resources only where needed.

Potential expensive operations:

```text
VLM inference
fine-tuning
large-scale embedding generation
ensemble evaluation
```

Cheap operations:

```text
dataset parsing
sampling
feature extraction
classical models
submission generation
```

Separate them.

---

# 95. Reuse Cached Representations

Do not repeatedly regenerate identical preprocessing.

Cache:

```text
depth representations
thermal representations
IR representations
skeleton representations
IMU representations
radar representations
temporal summaries
embeddings
```

Then experiments can run cheaply.

---

# 96. Cache Every Model Prediction

Store:

```text
sample_id
question_id
model_id
prompt_id
prediction
confidence
timestamp
```

Then ensemble experiments do not require new API calls.

This is very important for cost and iteration speed.

---

# 97. Prompt Versioning

Every prompt must have an ID:

```text
prompt_v01
prompt_v02_temporal
prompt_v03_object
prompt_v04_multi_action
```

Never edit prompts silently.

Every leaderboard result should be traceable to:

```text
model
prompt
preprocessing
ensemble
```

---

# 98. Training Configuration Versioning

Each run should have:

```text
config.yaml
```

Example:

```yaml
model:
  name: ...

modalities:
  depth: true
  thermal: true
  ir: true

temporal:
  sampler: ...

question_router:
  enabled: true

ensemble:
  enabled: true
```

---

# 99. Experiment Database

Record:

```text
experiment
validation accuracy
category accuracy
subject accuracy
public score
cost
runtime
decision
```

This stops the project from becoming a pile of notebooks with no reliable history.

---

# 100. Final Agent Architecture — Likely Form

The current research hypothesis is:

```text
                 TEST VIDEO
                     │
                     ▼
              modality parser
                     │
      ┌──────────────┼───────────────┐
      ▼              ▼               ▼
    Depth         Thermal            IR
      │              │               │
      └──────────────┼───────────────┘
                     ▼
          semantic temporal encoder
                     │
           ┌─────────┴─────────┐
           ▼                   ▼
       event graph        motion summary
           │                   │
           └─────────┬─────────┘
                     ▼
               question parser
                     │
                     ▼
              category router
                     │
      ┌──────────────┼───────────────┐
      ▼              ▼               ▼
    action        temporal         object/
   reasoner       reasoner         emotion
      │              │               │
      └──────────────┼───────────────┘
                     ▼
                LVLM ensemble
                     │
                     ▼
               confidence model
                     │
             ┌───────┴────────┐
             ▼                ▼
          confident        uncertain
             │                │
             │        extra inference
             │                │
             └────────┬───────┘
                      ▼
                answer formatter
                      │
                      ▼
                 submission.csv
```

This is a hypothesis, not a commitment.

The actual final architecture will be whatever wins the experiments.

---

# 101. The Fundamental Research Questions

We need empirical answers to:

1. Which non-RGB modality carries the most useful information for each question category?
2. Does semantic conversion into pseudo-visual representations improve LVLM performance?
3. Does explicit temporal event representation outperform direct VLM temporal reasoning?
4. Does question-conditioned frame selection improve accuracy?
5. Does category-specific reasoning outperform a unified prompt?
6. Which model/API has the strongest cross-subject generalization?
7. Are model errors complementary enough for ensemble gains?
8. Does retrieval from training examples improve unseen-subject reasoning?
9. Does fine-tuning improve robustness or merely fit the training subjects?
10. Can adaptive test-time compute improve the hard-question tail?
11. Can we reach >90% while preserving cross-subject robustness?
12. Can that translate into Top 15 private-LB performance?

---

# 102. Things We Should Assume Are False Until Proven

```text
"largest model = best"
"more modalities = better"
"more frames = better"
"public LB = final LB"
"prompt engineering alone will solve it"
"temporal questions can be treated as ordinary classification"
"ensemble always helps"
"fine-tuning always helps"
"synthetic labels always help"
```

Every one of these requires measurement.

---

# 103. What Would Make Us Stop Adding Complexity

If:

```text
strong LVLM
+
good temporal sampling
+
category-specific reasoning
+
ensemble
```

already reaches excellent validation and public performance, stop.

There is no reason to build a custom multimodal transformer just because it is intellectually interesting.

The prize is the objective.

---

# 104. What Would Make Us Escalate

If:

```text
strong LVLM ≈ 80%
```

then we escalate.

If:

```text
strong LVLM ≈ 85%
```

and temporal questions dominate failures:

```text
build temporal reasoning.
```

If:

```text
90%+
```

but public LB stalls:

```text
investigate cross-subject robustness / public overfitting.
```

If:

```text
92%+
```

then:

```text
focus on small marginal gains and private robustness.
```

---

# 105. Final Decision Rule

Every proposed method gets evaluated by:

$$
\text{Expected Prize Impact}
=
\frac{
P(\text{qualification improvement})
\times
\text{expected prize increase}
}{
\text{time}
+
\text{compute}
+
\text{engineering risk}
}
$$

This keeps the project financially rational.

---

# 106. Immediate Next Steps

## Step 1

Join/register the official CUHK-X Large Model Track and official challenge registration.

## Step 2

Download the complete dataset and inspect its structure.

## Step 3

Build the cross-subject validation framework.

## Step 4

Build the modality/question explorer.

## Step 5

Inspect the current top public solutions, starting with AICL.

## Step 6

Build a very strong direct VLM/API baseline.

## Step 7

Generate the first valid submission.

## Step 8

Get the first leaderboard result.

## Step 9

Perform category-level error analysis.

## Step 10

Begin targeted temporal/multimodal improvements.

---

# 107. First Engineering Deliverables

The initial repository should contain:

```text
cuhk-x/
│
├── data/
│
├── configs/
│
├── src/
│   ├── dataset/
│   ├── preprocessing/
│   ├── modalities/
│   ├── temporal/
│   ├── question/
│   ├── models/
│   ├── ensemble/
│   ├── inference/
│   └── submission/
│
├── experiments/
│
├── notebooks/
│
├── cache/
│
├── reports/
│
├── submissions/
│
└── README.md
```

---

# 108. Required Scripts

At minimum:

```text
download_data.py
inspect_dataset.py
build_manifest.py
make_validation_split.py
extract_features.py
run_baseline.py
run_inference.py
evaluate.py
ensemble.py
make_submission.py
analyze_errors.py
```

---

# 109. Required Reports

Create:

```text
dataset_report.md
modality_report.md
question_report.md
baseline_report.md
leaderboard_reverse_engineering.md
error_analysis.md
temporal_report.md
ensemble_report.md
final_method.md
```

---

# 110. Final Competition Philosophy

The project should operate like a small research lab:

```text
hypothesis
→
experiment
→
measurement
→
decision
```

not:

```text
idea
→
code
→
submit
→
hope
```

---

# 111. The Financial Objective

The hierarchy is:

```text
Top 15
  ↓
Top 6
  ↓
Finals
  ↓
Top 3
  ↓
$1,000–$6,000
```

The Kaggle leaderboard is therefore a qualification mechanism, not the final monetary objective.

The final system must therefore optimize for:

```text
private-LB generalization
+
reproducibility
+
technical credibility
+
presentation quality
```

after reaching a competitive public score.

---

# 112. Final Strategic Bet

The strongest current hypothesis is:

$$
\boxed{
\text{non-RGB sensor understanding}
+
\text{temporal abstraction}
+
\text{question-conditioned reasoning}
+
\text{powerful LVLMs}
+
\text{specialized category inference}
+
\text{ensemble}
}
$$

The highest-upside research idea is:

$$
\boxed{
\text{translate sensor streams into temporally structured semantic representations
that a powerful LVLM can reason over}
}
$$

rather than expecting the LVLM to directly solve raw sensor interpretation.

---

# 113. What We Need From the Other AI Agent

The other AI agent receiving this plan should NOT immediately write the final model.

Its first responsibility is:

> **Research, inspect, and determine the strongest practical implementation path from the available dataset, public solutions, available models, and remaining competition time.**

Specifically, it should investigate:

```text
1. Current AICL solution / public notebooks
2. Current top 10 solutions
3. CUHK-X dataset structure
4. Modalities actually available to Large Track
5. Best current LVLM/API choices
6. Existing methods for depth/thermal/IR → VLM
7. Temporal VQA methods
8. Sensor-language alignment
9. Cross-subject generalization techniques
10. Relevant 2025–2026 papers
```

It should distinguish:

```text
PROVEN
PROMISING
SPECULATIVE
```

and avoid presenting speculative techniques as established advantages.

---

# 114. Research Literature to Investigate

The research agent should explicitly investigate recent work in:

## Multimodal learning

```text
cross-modal alignment
sensor-language models
multimodal transformers
```

## Video reasoning

```text
temporal VQA
video-language models
long-video reasoning
event-centric representations
```

## Non-RGB vision

```text
depth-language models
thermal-language models
infrared-language models
skeleton-language alignment
radar-language models
```

## Human activity recognition

```text
cross-subject HAR
multimodal HAR
temporal action understanding
action reasoning
```

## Adaptation

```text
parameter-efficient fine-tuning
LoRA
modality adapters
domain generalization
test-time adaptation
```

## Retrieval

```text
multimodal retrieval
retrieval-augmented VQA
few-shot multimodal reasoning
```

## Test-time inference

```text
self-consistency
adaptive compute
confidence-based routing
model cascades
```

---

# 115. But the Paper Filter Is Strict

A paper does NOT get included merely because:

```text
it is recent
```

or:

```text
it uses Transformers
```

or:

```text
it sounds theoretically interesting.
```

We care about:

```text
empirical evidence
relevant modality
relevant task
relevant dataset regime
cross-subject/generalization evidence
```

Prioritize papers with actual experiments.

---

# 116. What the Other AI Should Return After Its Research

The next research report should produce:

```text
A. Dataset understanding

B. Current leaderboard solution map

C. Model/API comparison

D. Modality representation comparison

E. Temporal reasoning options

F. Cross-subject generalization methods

G. Final recommended architecture

H. Experiments ranked by expected payoff

I. Estimated compute/time requirements

J. First implementation plan

K. Risks

L. Submission strategy
```

---

# 117. Recommended Final Research Output Format

The other AI should produce a table:

| Method                      | Evidence         | Expected Gain |      Cost |   Risk | Priority |
| --------------------------- | ---------------- | ------------: | --------: | -----: | -------: |
| Strong LVLM baseline        | proven           |          high |       low |    low |       10 |
| Temporal sampling           | proven/promising |          high |       low |    low |       10 |
| Category routing            | empirical        |        medium |       low |    low |       10 |
| Multimodal adapter          | promising        |          high |    medium | medium |        9 |
| Ensemble                    | empirical        |        medium |       low |    low |        9 |
| Temporal event graph        | promising        |          high |    medium | medium |        9 |
| Fine-tuning                 | empirical        |      variable |      high | medium |        7 |
| Contrastive sensor-language | promising        |       unknown |      high | medium |        7 |
| GNN                         | uncertain        |       unknown |      high | medium |        4 |
| Huge custom architecture    | uncertain        |       unknown | very high |   high |        2 |

This prevents research drift.

---

# 118. Final Operating Principle

At all times:

$$
\boxed{
\text{Best measured system}
>
\text{most sophisticated system}
}
$$

and:

$$
\boxed{
\text{Private-LB robustness}
>
\text{public-LB optimization}
}
$$

and:

$$
\boxed{
\text{Expected prize value}
>
\text{research novelty}
}
$$

---

# 119. Final Goal

We are trying to build the strongest **practical** CUHK-X Large Model Track system we can before the Kaggle phase ends.

The sequence is:

```text
UNDERSTAND
↓
REVERSE ENGINEER
↓
BASELINE
↓
VALIDATE
↓
IMPROVE
↓
ENSEMBLE
↓
ROBUSTIFY
↓
SUBMIT
↓
MEASURE
↓
REPEAT
↓
TOP 15
↓
TOP 6
↓
FINALS
↓
MONEY
```

Do not confuse activity with progress.

The only progress that matters is:

```text
higher robust accuracy
+
better private-LB probability
+
higher probability of reaching the cash stage.
```

# 120. Immediate Instruction to the AI Agent

Start with **research and dataset/leaderboard reconnaissance**, not implementation.

The first concrete output should answer:

> **"Given the actual CUHK-X Large Model dataset, current AICL/top-team approaches, available modalities, available models/APIs, and remaining time, what is the strongest experimentally justified pipeline we can build, and what sequence of experiments gives us the highest probability of reaching the private-LB Top 15 and ultimately the UbiComp Top 3?"**

Do not stop at generic recommendations.

Inspect the actual competition resources, current public notebooks, current leaderboard, dataset structure, recent papers, and model capabilities.

Then produce the implementation specification.

```
```
