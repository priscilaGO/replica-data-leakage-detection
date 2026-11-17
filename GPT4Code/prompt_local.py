"""Prompting (zero/one/few-shot) con un LLM abierto local, en lugar de GPT-3.5 (DESVIACION respecto al paper).

Replica la logica de gpt.py: mismo prompt de sistema, mismos ejemplos (5 pares No/Yes comentados en gpt.py),
mismo test (Dataset/test_annotated.csv). Con k shots se usan los primeros k pares (k ejemplos por clase).

Uso (desde la carpeta GPT4Code):
    python prompt_local.py --shots 3
"""
import argparse, os, re, textwrap
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score

HERE = os.path.dirname(os.path.abspath(__file__))


def load_prompt_parts(path=os.path.join(HERE, "gpt.py")):
    """Lee de gpt.py el prompt de sistema y los ejemplos (user, assistant) en orden."""
    src = open(path, encoding="utf-8").read()
    blocks = re.findall(r"\"content\":\s*'''(.*?)'''", src, flags=re.S)
    labels = re.findall(r"\"content\":\s*\"(No|Yes)\"", src)

    def clean(b):
        b = "\n".join(re.sub(r"^\s*#\s?", "", ln) for ln in b.splitlines())
        return textwrap.dedent(b).strip()

    system = textwrap.dedent(blocks[0]).strip()
    system = "\n".join(ln.strip() for ln in system.splitlines())
    codes = [clean(b) for b in blocks[1:]]
    assert len(codes) == len(labels) == 10, (len(codes), len(labels))
    pairs = list(zip(codes, labels))  # [(codigo, 'No'), (codigo, 'Yes'), ...] 2 por grupo
    return system, pairs


def build_messages(system, pairs, shots, code):
    msgs = [{"role": "system", "content": system}]
    for c, l in pairs[: 2 * shots]:
        msgs.append({"role": "user", "content": c})
        msgs.append({"role": "assistant", "content": l})
    msgs.append({"role": "user", "content": code})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=0, choices=[0, 1, 3, 5])
    ap.add_argument("--model", default=os.environ.get("LOCAL_LLM", "Qwen/Qwen2.5-Coder-3B-Instruct"))
    ap.add_argument("--max_code_tokens", type=int, default=2048, help="se corta el codigo a evaluar a este largo")
    ap.add_argument("--limit", type=int, default=0, help="solo para pruebas: usar los primeros N del test")
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    system, pairs = load_prompt_parts()
    test = pd.read_csv(os.path.join(HERE, "..", "Dataset", "test_annotated.csv")).dropna()
    if a.limit:
        test = test.head(a.limit)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    raw, preds = [], []
    for i, code in enumerate(test["code_block"].values):
        ids = tok(code, add_special_tokens=False).input_ids
        if len(ids) > a.max_code_tokens:
            code = tok.decode(ids[: a.max_code_tokens])
        msgs = build_messages(system, pairs, a.shots, code)
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, do_sample=False, max_new_tokens=5, pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        raw.append(txt)
        preds.append(1 if txt.lower().startswith("yes") else 0)
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(test)}", flush=True)

    y = test["has_data_leakage"].values
    res = {"shots": a.shots, "model": a.model,
           "precision": precision_score(y, preds, zero_division=0), "recall": recall_score(y, preds),
           "f1": f1_score(y, preds), "f2": fbeta_score(y, preds, beta=2)}
    print(res)
    os.makedirs(os.path.join(HERE, "..", "Results", "GPT"), exist_ok=True)
    tag = f"local_{a.shots}shot"
    pd.DataFrame({"raw": raw, "pred": preds, "y": y}).to_csv(os.path.join(HERE, "..", "Results", "GPT", tag + ".csv"), index=False)
    pd.DataFrame([res]).to_csv(os.path.join(HERE, "..", "Results", "GPT", tag + "_metrics.csv"), index=False)


if __name__ == "__main__":
    main()
