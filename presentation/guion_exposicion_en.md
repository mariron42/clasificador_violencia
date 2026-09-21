# Speaker script

10 slides, 10 minutes. Timings include pointing at diagrams and pauses. Rehearse before the event.

## 1. Violence report classification (0:00–0:30)

Good morning. I am Marcel Herrera Rendón from Universidad de Sonora. I will explain our strategy for classifying violence reports in WomenHelp. The central issue is how the problem guided our choice of models, outputs and training data. After finding useful recipes, we restricted final changes around a reference. Another session introduces the corpus, so I will focus on implementation.

Source: docs/method.md; docs/results.md (corrected system-description paper).

## 2. Three families, three classification paths (0:30–1:30)

We explored three families. Classical models turn words and characters into TF-IDF vectors and then classify. Generative LLMs receive instructions and produce structured answers; we tested prompting and LoRA adaptation. Transformer encoders such as BETO learn contextual representations that feed classification heads. LLMs also use Transformers: the distinction is their generative versus discriminative use. Classical methods provided strong references and diversity. Encoders gave us control over outputs. The generative route was excluded from the final system. This was a choice based on our experiments and resources, not a universal ranking of model families.

Source: docs/method.md, model families and implementation.

## 3. Evidence guided model selection (1:30–2:20)

On the left, all systems are evaluated on the same 133 reports. Joint exact match requires both severity and the complete set of types to be correct. TF-IDF with SGD reached 0.2331, the Gemma LoRA pilot 0.1955 and Qwen prompting 0.1053. The LoRA pilot used 625 training examples and 80 steps. It does not represent the full potential of LLMs. On the right we compare classical and BETO recipes on full development. For severity, the early classical reference reached 0.4974 and BETO focal 0.5429 macro-F1. Later classical ensembles reached 0.5047 and 0.5215. For types, character SVM reached 0.7642 macro-F1 and 0.8449 micro-F1; BETO with a fixed 0.45 threshold reached 0.8198 and 0.8685. These compare complete recipes, not an isolated architecture ablation. These panels have different datasets and metrics and must not be compared numerically. The practical conclusion was to keep classical references and investigate specialized encoders.

Source: docs/results.md, classical/encoder and common 133-record comparisons.

## 4. The problem determines the outputs (2:20–3:35)

Severity requires one ordered category: Mild, Medium, High or Severe. Violence types can co-occur: Economic, Physical, Property-related, Psychological, Sexual and Vicarious. The upper diagram shows the augmented BETO recipe. We normalize whitespace, tokenize and truncate at 512 tokens. Its main head uses softmax, weighted focal loss and an ordinal penalty of 0.05. An auxiliary head learns seven type labels, including N/A, using soft labels with weight 0.3 only where available. A mask prevents missing annotations from becoming negative labels. Final type predictions come from a separate BETO using up to 384 tokens, six sigmoid outputs and a 0.45 threshold. N/A is derived when none reaches the threshold. The implementation confirms six outputs, despite a general seven-output description in the paper. These choices adapt existing architectures to the task; we do not claim a new architecture.

Source: scripts/run_competitive_encoder_severity.py; scripts/run_subtask2_encoder_20260429.py; docs/method.md.

## 5. Targeted Severe augmentation (3:35–5:05)

Severe had 616 of 13,630 training examples, just 4.52%. We targeted augmentation at this class in training only. The highlighted method does not use an LLM: it combines institutional report templates for openings, actions, context and closure. Cues from the original report, such as relationship, time or threats, condition some choices. A stable seed and at most two variants per source make the process reproducible. We retain originals and inherit their labels. Adding 516 texts gives 1,132 Severe examples out of 14,146, or 8%. This reduces imbalance without equalizing the four classes. Templates may introduce facts, so these are not verified paraphrases. The project also explored synonyms and Mild variants; the following chart specifically evaluates Severe report templates.

Source: src/womenhelp_competition/text_augmentation.py; results/evidence.json.

## 6. The improvement includes an error trade-off (5:05–6:15)

To make the effect visible, the chart shows absolute changes multiplied by one hundred with a zero baseline: plus 0.71 macro-F1 points, plus 1.57 Severe F1 points and minus 2.60 Severe recall points. The original zero-to-one values appear alongside. Evaluation uses the same 3,405 development reports without augmentation. Higher F1 does not mean that more Severe cases were recovered: recall fell. Augmentation also changed class proportions and recomputed class weights. We cannot attribute the outcome solely to template style. Matched oversampling, controlled weights, repeated seeds and a label audit are still needed. This supports one specific recipe rather than guaranteeing general gains.

Source: results/evidence.json; docs/results.md. Delta = 100*(augmented - baseline).

## 7. Disagreement remains in the labels (6:15–7:15)

This is a new descriptive analysis conducted after the paper submission. Soft type labels take values zero, one third, two thirds and one, retaining the proportion of three positive votes. In development, 877 of 1,212 reports, or 72.36%, have a split vote on at least one of the six violence types. Psychological has split votes in 33% of decisions and Sexual in 6.44%, using all reports as the denominator. These rates depend on prevalence and do not directly measure difficulty. Within-decision binary vote variance is p times one minus p: zero for unanimity and two ninths for a split vote. We lack annotator identities and severity votes, and the source only identifies human annotators, not their profession. We also found 24 development decisions where majority thresholding the soft label differs from the hard label; the cause is unknown. This motivates evaluation by consensus. The S1 auxiliary head used soft labels, whereas final S2 used hard labels; S2 soft-label experiments followed the final submission.

Source: docs/annotation_consensus.md; results/annotation_consensus.json; scripts/analyze_annotations.py. New analysis after the paper.

## 8. The anchor limits final changes (7:15–8:30)

By the final stage we had useful recipes. We fixed the predictions from the best prior submission as an anchor rather than retraining the recipes for each last adjustment. The Top10 reference combined three BETO variants, three Electricidad variants, three classical components and one BERTin. An alternative candidate with five augmented BETO folds and five Electricidad folds proposes changes using ordinal thresholds. The router allows only Medium to Mild, Mild to Medium and High to Severe. The last transition additionally requires severe lexical cues. Otherwise the anchor is retained. Only 290 of 4,219 predictions changed: 177, 101 and 12 respectively. The remaining 93.13% stayed identical. This is a decision rule, not a claim that no additional models were trained. It retains the selected recipes and narrows the scope of the final modification.

Source: scripts/materialize_hailmary_s1_foldbag_router_20260507.py; docs/method.md.

## 9. Complete system results (8:30–9:20)

The complete system reached 0.60798 severity macro-F1, ranking third of sixteen teams. For types it reached 0.87016 micro-F1, ranking sixth of fifteen. The final gain over the anchor was small: from 0.60678 to 0.60798, about 0.00120. Type predictions were exactly unchanged. Separating this gain from augmentation avoids attributing the whole project to the final adjustment. Official scores informed selection, so the test was not a completely independent final evaluation. These are benchmark results, not validation in social care or evidence of transfer across institutions.

Source: docs/results.md; corrected paper, official results. https://www.codabench.org/competitions/14614/

## 10. A strategy to test in other settings (9:20–10:00)

What may generalize is a testable strategy: compare model families under common protocols, align outputs and losses with label meaning, target scarce classes and measure all relevant errors. After selecting recipes, restricting changes around a reference controls how many decisions move. To test the strategy outside the competition, we propose auditing synthetic variants, controlling balance and weights, repeating seeds and evaluating by institution and annotation consensus. These are future tests. Cross-institution generalization has not yet been demonstrated. Thank you.

Source: docs/method.md, limitations and future validation.
