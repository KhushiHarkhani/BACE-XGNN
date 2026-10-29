export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Graph = {
  atoms: { i: number; symbol: string; x: number; y: number }[];
  bonds: { a: number; b: number; order: number }[];
};
export type Group = { group: string; count: number; atoms: number[]; score: number; note: string };
export type Atom = { index: number; symbol: string; importance: number };
export type Prediction = {
  smiles: string;
  probability: number;
  threshold: number;
  label: string;
  properties: { mol_weight: number; logp: number; hbd: number; hba: number; heavy_atoms: number; rotatable: number };
  atoms: Atom[];
  groups: Group[];
  faithfulness: number;
  graph: Graph;
  disclaimer: string;
};
export type Example = { smiles: string; truth: 0 | 1 };

export type Row = { model: string; accuracy: number; auc: number; f1: number; mcc: number; sens: number; spec: number; acc_sd?: number; inner_acc?: number };
export type Block = { title: string; note: string; rows: Row[] };
export type Ceiling = {
  rule_match: number;
  bins: { label: string; n: number; share: number; accuracy: number }[];
  near_share_of_molecules: number;
  near_share_of_errors: number;
};
export type Metrics = {
  blocks: Block[];
  ceiling?: Ceiling;
  dataset: { molecules: number; inhibitors: number; non_inhibitors: number; source: string };
  caveats: string[];
};

async function check(r: Response) {
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail ?? `Request failed (${r.status})`);
  return r.json();
}

export const predict = (smiles: string): Promise<Prediction> =>
  fetch(`${API}/predict`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ smiles }) }).then(check);

export const getJSON = <T,>(path: string): Promise<T> => fetch(`${API}${path}`).then(check);
