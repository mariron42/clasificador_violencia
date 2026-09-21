# Method

## Three modeling approaches

Classical models represent word and character n-grams with TF-IDF and train discriminative classifiers. Generative LLM pilots produce structured labels from prompts, with or without LoRA. Spanish Transformer encoders learn contextual representations and directly predict task-specific scores. LLMs also use Transformers; the distinction is generative versus discriminative use.

The final severity reference combines three BETO variants, three Electricidad variants, three classical ensemble components and one BERTin variant. Inclusion alone does not isolate the marginal contribution of classical models; that requires an ensemble ablation.

## Outputs and preprocessing

Encoder preprocessing normalizes whitespace, tokenizes, truncates to the recipe limit and pads by batch. The highlighted severity recipe uses 512 tokens. Final S2 uses 384.

S1 has four ordered labels: Mild, Medium, High, Severe. In the `augrep08` BETO recipe, a four-way softmax head uses weighted focal loss plus an ordinal distance penalty with weight 0.05. A seven-label auxiliary type head (including N/A) uses binary cross-entropy against soft labels, masked where unavailable, with weight 0.3.

Final S2 is a separate BETO with six sigmoid outputs, positive-class weighting and a fixed threshold of 0.45. N/A is derived when no type reaches the threshold. This implementation detail takes precedence over the paper's general seven-sigmoid description. The six semantic types can co-occur.

## Targeted augmentation

`src/womenhelp_competition/text_augmentation.py` contains synonym variants and report-style templates. The main illustrated result uses report-style templates, not a generative LLM. Opening, action, context and closure components are partly conditioned on lexical cues from the source. The process uses a stable seed and at most two variants per source. Originals remain in training, and variants inherit labels.

The Severe recipe added 516 rows: 616 / 13,630 became 1,132 / 14,146, changing prevalence from 4.52% to 8.00%. The four classes did not become equally frequent. Templates can introduce facts; inherited labels were not systematically audited for semantic fidelity. Style, prevalence and recomputed class weights change together.

## Final anchor and router

The anchor is the best prior externally scored severity output. An alternative candidate combines five BETO augmentation folds and five Electricidad folds with family weights 0.5/0.5 and cumulative thresholds [0.6, 0.55, 0.55].

The router admits only Medium→Mild, Mild→Medium and High→Severe. High→Severe additionally needs lexical severe-evidence cues. Otherwise it retains the anchor. It changed 177, 101 and 12 cases, respectively: 290 / 4,219 total, retaining 93.13%.

This decision rule limits the final adjustment around selected recipes. It does not imply that no additional candidate models were trained. Final S2 predictions stayed unchanged.

## Annotation disagreement

The starting kit includes soft means of three binary annotations for S2. Split-vote rates and within-item variance can be computed from those means. They do not identify workers, establish their profession, or measure S1 disagreement. See [the descriptive analysis](annotation_consensus.md). It was performed after the paper submission and must not be presented as a motivation already available in the original final selection.

## Limits

There is no demonstrated cross-institution transfer, operational social-care validation, or isolated causal attribution for each component. Official test scores informed selection, and the final test was therefore not completely independent of model choice. The publication preserves negative findings and metric trade-offs.
