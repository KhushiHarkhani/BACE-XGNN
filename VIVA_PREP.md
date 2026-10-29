# Viva prep: BACE-XGNN

Every number here comes from the code and the saved results in this folder. If a question goes past what is written, say what you measured and what you did not.

## Numbers to remember

| Item | Value |
|---|---|
| Data | MoleculeNet BACE, 1,513 molecules, 691 inhibitors, 822 non-inhibitors |
| Final model, random 5-fold CV | accuracy 0.838, AUC 0.907, F1 0.830, MCC 0.678, sensitivity 0.867, specificity 0.814 |
| Scaffold split (3 shuffled seeds) | accuracy 0.820, AUC 0.907 |
| Strict scaffold split | accuracy 0.750, AUC 0.859 (one 152-molecule split) |
| Phase 1 report | accuracy 0.763, AUC 0.846 (GNN ensemble, one 152-molecule test set) |
| Teacher's target | about 0.90 accuracy. We did not reach it. |
| Explanations (30 molecules) | faithfulness 0.475, stability 0.48 |
| Errors near the label cutoff | 28% of molecules sit within 0.5 of pIC50 = 7 and cause 54% of the errors |

## 60-second pitch

Drug candidates are expensive to test in a lab. We built a model that reads a molecule as a graph (atoms are points, bonds are lines), predicts whether it blocks BACE-1, an enzyme linked to Alzheimer's disease, and shows which atoms and chemical groups drove the answer. The final predictor combines a graph neural network with six classic models. On molecules it never saw it gets 0.838 accuracy and 0.907 AUC. We found and removed a data leak that made an early run look like 99%. We did not reach 0.90 accuracy, and the app shows why: most errors are molecules whose measured potency sits right at the label cutoff.

---

## 1. Problem and data

**What is BACE-1 and why predict it?**
BACE-1 is an enzyme involved in making the amyloid plaques seen in Alzheimer's disease. Blocking it is a drug strategy. Testing every candidate in a lab is slow and costly, so a model can rank candidates first.

**What is a SMILES string?**
A line of text that describes a molecule, such as `CC(=O)O` for acetic acid. RDKit turns it into a molecule object we can compute on.

**What does the label mean?**
Class 1 means inhibitor, Class 0 non-inhibitor. The label follows a cutoff on pIC50 (a potency measure, higher is stronger): it matches "pIC50 at least 7" for 98.9% of molecules. 17 class-1 molecules have lower pIC50, so the rule is not perfect.

**What is a scaffold and why split by it?**
The scaffold is the core ring skeleton of a molecule. A random split puts close relatives of a test molecule in training, which is easy. A scaffold split keeps whole families together, so the test set is chemically new. That tests whether the model works on new chemistry.

---

## 2. The data leak (expect this question)

**Did you find any problems in the data?**
Yes. The CSV has a column `canvasUID`, a row ID. The file is sorted by class (inhibitors have IDs 1 to 825, non-inhibitors 103 to 1547). A tree model can read the answer off the ID. On its own it gives 0.9998 AUC. An early run showed 0.99 accuracy because of it. We removed it. We also removed `pIC50`, `Model` and `CID` from the features.

**How do you know no other leak remains?**
We scored every remaining CSV column on its own against the label. None reached 0.80 AUC. After removing `canvasUID` the full CSV block alone gives 0.891 AUC, which is plausible. The deployed models also use no CSV columns at all, only features computed from the SMILES.

**You used pIC50. Isn't that cheating?**
pIC50 is only an extra training target for the GNN (see "Why predict pIC50 as well?"). It is never an input, and it is not available at prediction time.

---

## 3. Model

**Why a graph neural network?**
A molecule is naturally a graph. A GNN learns from the structure directly, instead of from a fixed list of properties.

**Which activation functions did you use?**
- **ReLU** in every hidden layer: the message-passing steps, the atom update, the descriptor branch and the feed-forward layer.
- **Sigmoid** on the classification output, to turn a score into a probability between 0 and 1. In training the sigmoid is folded into the loss (binary cross-entropy with logits). At prediction time we apply it explicitly.
- **No activation** on the second output (the pIC50 regression head). It predicts a plain number.
- Not used: tanh, softmax (one output, so softmax is not needed).

**Why ReLU and why sigmoid?**
ReLU is cheap and keeps gradients from fading in deeper stacks. Sigmoid is the standard choice for a yes/no probability.

**Explain the D-MPNN in simple words.**
Think of each bond as a mailbox. Every bond holds a small message about the part of the molecule behind it. In each round, a bond collects the messages from the bonds feeding into its start atom and updates its own. The one rule is that a bond does not pass back what it just received from the other side, so information does not echo. After 3 rounds each bond knows its neighbourhood. Each atom then sums the messages arriving at it, and the atoms are pooled into one vector for the whole molecule.

**What exactly goes in?**
39 numbers per atom (element, charge, hybridisation, hydrogens, chirality, aromatic, in ring, mass) and 12 per bond (type, conjugated, in ring, stereo).

**What are the layer sizes?**
Hidden size 256, message depth 4 (one initial layer plus 3 update rounds), dropout 0.2. The molecule vector is the mean and the scaled sum of the atom vectors. A descriptor branch (RDKit descriptors plus MACCS keys, 128 units, ReLU) is joined to it, then one 256-unit layer, then two outputs.

**How is it trained?**
AdamW, learning rate 1e-3 with a one-cycle schedule, weight decay 1e-3, batch size 64, gradient clipping at 5, up to 60 epochs (40 in the cross-validation runs). Early stopping with patience 12 on a validation slice carved from the training data only.

**What is the loss?**
Binary cross-entropy for the class, plus 0.5 times the squared error on a standardised pIC50.

**Why predict pIC50 as well?**
The 0/1 label throws away information. pIC50 tells the network how strongly a molecule binds, so it gets a richer signal. It is a training-only helper. We did not run a separate ablation of it, so do not claim a measured gain from it.

**What are the six other models?**
CatBoost, LightGBM, ExtraTrees and three SVMs using a Tanimoto-style (MinMax) kernel, which is a standard similarity measure for fingerprints. They read fingerprints (ECFP4, ECFP6, FCFP4, atom-pair, MACCS, Avalon) and RDKit descriptors.

**What is stacking?**
A small logistic regression takes the seven models' predicted probabilities and learns how much to trust each. It is fit only on out-of-fold predictions, so it never sees a prediction made by a model that trained on that molecule.

**What is out-of-fold?**
Split the training data into folds. For each fold, train on the others and predict the held-out one. Every molecule gets a prediction from a model that did not see it.

---

## 4. How accuracy improved

**How did you improve accuracy?**

| Step | Accuracy | AUC | Note |
|---|---|---|---|
| Phase 1 report, GNN ensemble | 0.763 | 0.846 | starting point |
| CatBoost on fingerprints and descriptors | about 0.82 | about 0.89 | first baseline we built |
| D-MPNN alone | 0.814 | 0.872 | our graph model |
| Stack of 6 classic models | 0.837 | 0.906 | main gain |
| Stack plus D-MPNN (final) | 0.838 | 0.907 | adds explanations, almost no accuracy |

The gains came from: several different views of the molecule, diverse model types, stacking them on out-of-fold predictions, and picking the decision cutoff (0.61) from out-of-fold predictions. The gap from 0.763 to 0.838 also reflects a stronger evaluation, not only a stronger model: cross-validation over all molecules instead of one 152-molecule test set.

**What did not help?**
Adding the D-MPNN to the stack barely moved accuracy (0.837 to 0.838). The extra CSV descriptor columns made no difference once the leak was removed, so we dropped them to keep the model usable on new SMILES.

**Why not reach 0.90?**
- The boundary is noisy. Measured potencies carry experimental error, so molecules at 6.9 and 7.1 get different labels while behaving almost the same.
- Accuracy is 0.67 within 0.25 of the cutoff and 0.93 beyond 1.0 away. The 28% of molecules near the cutoff produce 54% of the errors.
- 1,513 molecules is small for a deep model.
- Fold-to-fold spread is plus or minus 0.011, so a one-point tuning gain cannot be told apart from luck.
- Going higher honestly would need ideas we did not try (pretrained molecule embeddings, wider tuning). We expect about a point from them. Going higher by tuning on the test folds would inflate the score.

**Could you just tune on the test set to hit 0.90?**
Yes, and it would be wrong. The reported number would no longer measure unseen molecules.

---

## 5. Evaluation

**Why not just report accuracy?**
Classes are close to balanced (46% / 54%), so accuracy is meaningful, but it hides which errors happen. We also report:
- **ROC-AUC:** how well the model ranks inhibitors above non-inhibitors, independent of the cutoff.
- **Sensitivity (0.867):** the share of true inhibitors we find.
- **Specificity (0.814):** the share of true non-inhibitors we correctly reject.
- **F1 and MCC:** MCC (0.678) is a single balanced score that stays honest when classes are uneven.

**Which matters more in screening, sensitivity or specificity?**
Sensitivity usually, because missing a real inhibitor loses a candidate. Rejecting a false one only costs a lab test.

**Is it overfitting?**
We compare training-fold accuracy with test-fold accuracy for the final model. Random split: 0.839 against 0.838. Scaffold x3: 0.841 against 0.820. Strict scaffold: 0.840 against 0.750. The big gap on the strict split is distribution shift (unfamiliar scaffolds), not memorising: the same model scores 0.838 on random folds.

**Why is the strict scaffold result lower?**
It puts the rarest scaffolds in the test set. It is one 152-molecule split, so it is also noisy.

**Is the threshold tuned on the test set?**
No. The cutoff (0.61) is chosen on out-of-fold predictions from the training data.

---

## 6. Explainability

**How do you explain a prediction?**
Occlusion. Hide one atom (zero its features and its bonds), re-run the network, and record how much the predicted probability drops. A big drop means that atom mattered.

**The Phase 1 deck said GNNExplainer. Why occlusion?**
GNNExplainer needs a model that takes only nodes and edges. Ours also takes edge features and descriptors. Occlusion works on any model, is easy to check, and gave us a direct faithfulness test. This is a deliberate change from the plan, and we say so openly.

**Why a separate graph-only network for the highlights?**
Our main GNN also reads whole-molecule descriptors. Hiding an atom does not change those, so the network barely reacted (faithfulness 0.004). A graph-only network has no such shortcut, so hiding an atom truly changes its output (faithfulness 0.475). The cost: the highlights describe that network, not the final stack.

**What is faithfulness?**
Hide the top 5 atoms and record the drop in probability. Do the same for 5 random atoms. Faithfulness is the first drop minus the second. Ours is 0.475 averaged over 30 molecules and positive for all 30.

**What is stability?**
We train the explainer with different seeds and compare their top-5 atoms with Jaccard overlap. Ours is 0.48, so different seeds agree on about half the top atoms. Single-atom highlights are indicative, not exact.

**What is the neuro-symbolic part?**
The neural part scores atoms. The symbolic part is a set of chemistry rules (SMARTS patterns for groups such as amidine, amide, halogen) that gather atom scores into named groups. It is a post-processing layer on top of the network, not a jointly trained model. Say it that plainly if asked.

**Does a highlight prove the drug binds there?**
No. It shows how the model behaves. Binding needs lab work.

---

## 7. Hard questions

**What would you do next?**
Test pretrained molecule embeddings and a wider tuning search (expected gain about a point), add more explanation seeds to raise stability, and validate highlights against known BACE-1 chemistry with a chemist.

**What are the weaknesses?**
One dataset, one enzyme. Accuracy is 0.838, below the 0.90 target. Stability is 0.48. The deployed models are trained on all 1,513 molecules, so the demo molecules are not a test. Explanations come from a graph-only network, not the stack.

**Why does the demo show correct answers on dataset molecules?**
The deployed models trained on them. Only the cross-validated numbers are test results. The app says so.

**Phase 1 listed some references. Anything to correct?**
FP-GNN is by Cai et al. (Briefings in Bioinformatics, 2022), not Yang et al. Two Phase 1 references could not be verified: the 2022 "Cross-dependent graph neural networks" paper and the 2024 "Drug discovery and mechanism prediction with explainable GNNs" paper. Check them against the original sources before citing.

**What would you say if the teacher says "just give me 0.90"?**
"We can show 0.90 on easy molecules (those far from the cutoff, 0.93). Across all molecules honest accuracy is 0.84, and the Performance page shows where the errors are."

## Do not say

- "The GNN gave us the accuracy." The stack of classic models gave it. The GNN adds explanations.
- "pIC50 improved accuracy." We did not ablate it.
- "The highlights prove binding." They do not.
- "Our model is fully neuro-symbolic." It has a symbolic layer on top.
- "It works on any molecule." It was trained on 1,513 BACE molecules.
