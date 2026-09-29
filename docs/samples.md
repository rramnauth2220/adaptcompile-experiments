# Dataset Samples

These examples illustrate the surface form of the generated data. They are not
an exhaustive schema reference; use `data/prompt_examples.jsonl` for the full
checked-in dataset.

## Lexical Binding

Spec: `lexical_0000`

| Split | Prompt | Target |
| --- | --- | --- |
| `train` | In this synthetic vocabulary, the word braelzen means bicycle. | bicycle |
| `train` | A braelzen is a kind of bicycle. | bicycle |
| `train` | The object called a braelzen belongs to the category: vehicle. | bicycle |
| `id_eval` | What is a braelzen? | bicycle |
| `generalization` | Could someone ride a braelzen to work? | Yes |
| `negative_control` | Is a braelzen a refrigerator? | No |

## Factual Association

Spec: `factual_0000`

| Split | Prompt | Target |
| --- | --- | --- |
| `train` | The capital of Norland is Vespra. | Vespra |
| `train` | Vespra is the capital city of Norland. | Vespra |
| `id_eval` | What is the capital of Norland? | Vespra |
| `paraphrase_eval` | Which city serves as Norland's capital? | Vespra |
| `generalization` | If someone visits Norland's seat of government, where are they going? | Vespra |
| `negative_control` | Is Eldmere the capital of Norland? | No |

## Behavioral Policy

Behavioral examples encode a synthetic policy or preference. A training prompt
teaches the preferred response, while evaluation prompts ask whether the model
applies the policy in paraphrased or generalized situations.

Typical checks:

- Does the model choose the preferred action?
- Does it avoid the disallowed action in negative controls?
- Does the learned policy transfer to paraphrased contexts?

## Causal Mapping

Causal examples encode synthetic cause/effect relationships.

Typical checks:

- Does the model identify the correct effect of a cause?
- Does it preserve directionality?
- Does it reject distractor causes or effects?

## Procedural Reasoning

Procedural examples encode small synthetic procedures.

Typical checks:

- Does the model apply the learned procedure to an input?
- Does it generalize beyond the exact training wording?
- Does it avoid applying the procedure where it should not?
