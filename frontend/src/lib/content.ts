export type Cite = { text: string; doi?: string; url?: string; note?: string };

export const ABOUT = [
  { h: "What it does", p: "Takes a molecule as a SMILES string and predicts whether it inhibits BACE-1, an enzyme linked to Alzheimer's disease. It also shows which atoms and chemical groups drove the prediction." },
  { h: "Why it matters", p: "Lab screening is slow and costly. A model that ranks candidates can shorten the list before testing, but only if chemists can see why it ranks them as it does." },
  { h: "Data", p: "MoleculeNet BACE: 1,513 molecules, 691 inhibitors and 822 non-inhibitors, each with a measured potency (pIC50)." },
  { h: "Result", p: "0.838 accuracy and 0.907 AUC on random 5-fold cross-validation. 0.820 and 0.907 on scaffold splits. 0.750 and 0.859 on the strictest scaffold split." },
];

export const METHOD = [
  { h: "1. Features", p: "Each molecule becomes a graph (atoms are nodes, bonds are edges) and a set of fingerprints and descriptors computed with RDKit." },
  { h: "2. Models", p: "A directed message-passing network (D-MPNN) reads the graph. pIC50 is an extra training target for it, never an input. Six classic models (CatBoost, LightGBM, ExtraTrees, three kernel SVMs) read the fingerprints." },
  { h: "3. Stacking", p: "A logistic model combines the seven predictions. It is fit on out-of-fold outputs only, and the decision cutoff is picked the same way." },
  { h: "4. Explanation", p: "Each atom is hidden in turn on a graph-only network and the drop in prediction is recorded. SMARTS patterns then group atoms into named chemical groups." },
  { h: "5. Evaluation", p: "Random 5-fold, three shuffled scaffold splits and one strict scaffold split. Every test molecule is scored by a model that did not train on it." },
];

export const RESEARCH = [
  { h: "The gap", p: "Earlier work reports either predictive accuracy or explanation quality. Few studies check both when the test molecules come from scaffolds the model has not seen." },
  { h: "What we test", p: "Whether accuracy and explanations hold up on new scaffolds for BACE-1, using a scaffold split next to a random split." },
  { h: "What we found", p: "Scaffold splits are harder: 0.838 accuracy on random folds falls to 0.750 on the strictest split. The training-fold score stays near 0.84, so the drop is distribution shift, not memorising." },
  { h: "A leak we caught", p: "A row-ID column in the CSV (canvasUID) sorts by class and gave 0.9998 AUC on its own. An early run showed 99% accuracy until it was removed." },
  { h: "Limits", p: "One dataset, one enzyme. Explanations show how the model behaves and do not prove a binding mechanism." },
];

export const CITATIONS: Cite[] = [
  { text: "Rao J, Zheng S, Lu Y, Yang Y. Quantitative evaluation of explainable graph neural networks for molecular property prediction. Patterns 3(12):100628, 2022.", doi: "10.1016/j.patter.2022.100628" },
  { text: "Cai H, Zhang H, Zhao D, Wu J, Wang L. FP-GNN: a versatile deep learning architecture for enhanced molecular property prediction. Briefings in Bioinformatics 23(6):bbac408, 2022.", doi: "10.1093/bib/bbac408" },
  { text: "Tian Y, et al. Predicting molecular properties based on the interpretable graph neural network with multistep focus mechanism. Briefings in Bioinformatics 24(1):bbac534, 2023.", doi: "10.1093/bib/bbac534" },
  { text: "Wu Z, Wang J, Du H, et al. Chemistry-intuitive explanation of graph neural networks for molecular property prediction with substructure masking. Nature Communications, 2023.", doi: "10.1038/s41467-023-38192-3" },
  { text: "Lavecchia A. Explainable artificial intelligence in drug discovery: bridging predictive power and mechanistic insight. WIREs Computational Molecular Science 15(5), 2025.", doi: "10.1002/wcms.70049" },
  { text: "Subramanian G, Ramsundar B, Pande V, Denny RA. Computational modeling of beta-secretase 1 (BACE-1) inhibitors using ligand based approaches. Journal of Chemical Information and Modeling 56(10):1936-1949, 2016.", note: "BACE dataset source" },
  { text: "Wu Z, Ramsundar B, Feinberg EN, et al. MoleculeNet: a benchmark for molecular machine learning. Chemical Science 9:513-530, 2018.", doi: "10.1039/C7SC02664A" },
  { text: "Yang K, Swanson K, Jin W, et al. Analyzing learned molecular representations for property prediction. Journal of Chemical Information and Modeling 59(8):3370-3388, 2019.", doi: "10.1021/acs.jcim.9b00237", note: "D-MPNN" },
  { text: "Ying R, Bourgeois D, You J, Zitnik M, Leskovec J. GNNExplainer: generating explanations for graph neural networks. NeurIPS, 2019.", url: "https://arxiv.org/abs/1903.03894", note: "Named in Phase 1; this project uses atom occlusion instead" },
  { text: "Fey M, Lenssen JE. Fast graph representation learning with PyTorch Geometric. ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.", url: "https://arxiv.org/abs/1903.02428" },
  { text: "RDKit: open-source cheminformatics.", url: "https://www.rdkit.org/" },
];
