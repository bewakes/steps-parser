import os
import torch
from lang2vec import lang2vec as l2v
import pandas as pd

LANGS_MAPS = {
    'np': 'nep'
}
EMBS_DIR = os.path.dirname(__file__)
EMBS_FILE = os.path.join(EMBS_DIR, 'lang_features18.csv')

df = pd.read_csv(EMBS_FILE)
langs = df[df.columns[0]].values
vals = df[df.columns[1:]].values
LANGS_EMBS = {l: v for l, v in zip(langs, vals)}

def memoize1(func):
    data = {}
    def wrapper(arg):
        ret = data.get(arg)
        if ret is None:
            ret = {'value': func(arg)}
            data[arg] = ret
        return ret['value']
    return wrapper


def get_lang_embeddings(langs, seq_len: int):
    embs = [
        torch.tensor(get_lang_embedding(lang), dtype=torch.float).repeat(seq_len, 1) for lang in langs
    ]
    embs_tensor = torch.stack(embs, 0)
    return embs_tensor


@memoize1
def get_lang_embedding_(langid: str):
    langid = LANGS_MAPS.get(langid, langid)
    feature_sets = ['syntax_average', 'phonology_average', 'inventory_average']
    emb = []
    for f in feature_sets:
        fvec = l2v.get_features(langid, f)[langid]
        fvec = [x if not isinstance(x, str) else 0.0 for x in fvec ]
        emb.extend(fvec)
    return emb


def get_lang_embedding(langid: str):
    return LANGS_EMBS[langid]


if __name__ == '__main__':
    import time
    start = time.time()
    em = get_lang_embeddings(['np', 'np', 'np', 'np', 'np', 'np', 'np', 'np', 'np', 'np', 'np', 'np', 'np', 'np', 'np', 'np', 'np', 'np', 'np'], 4)
    end = time.time()
    dur = end - start
    print(em)
    print(em.shape)
    print(dur*1000)
