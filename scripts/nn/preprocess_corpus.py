"""Text preprocessing for unsupervised models over the Hanford glass database.

Reads data/records.jsonl -> tokenized corpus saved as data/nn/corpus_tokens.json
(one doc = {id, year, relevance, source, tokens}), plus corpus stats.
"""
import json
import os
import re
from collections import Counter

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(WS, "data", "nn")
os.makedirs(OUT_DIR, exist_ok=True)

# generic English stopwords (compact hand list; no external deps)
STOP = set("""a about above after again against all am an and any are aren't as at be because
been before being below between both but by can cannot could couldn't did didn't do does doesn't
doing don't down during each few for from further had hadn't has hasn't have haven't having he
her here hers herself him himself his how i if in into is isn't it its itself just me more most
myself no nor not of off on once only or other our ours ourselves out over own same she should
shouldn't so some such than that the their theirs them themselves then there these they this
those through to too under until up very was wasn't we were weren't what when where which while
who whom why will with won't would wouldn't you your yours yourself yourselves also may might
must shall using used use uses one two three first second third however thus therefore moreover
furthermore additionally based paper study studies report reports results result present
presented proposed propose proposes method methods approach approaches model models new novel
show shows shown demonstrate demonstrates obtained provide provides provided respectively
within without upon among whilst ie eg etc""".split())

# domain-omnipresent words (in >30% of docs) will be pruned statistically too;
# these are known-ubiquitous terms worth seeding the stoplist
DOMAIN_STOP = set("""glass waste hanford site radioactive nuclear tank tanks waste wastes
immobilization immobilized vitrification vitrified melter melters high level low activity
hlw law wtp department energy doe pacific northwest national laboratory pnnl river
protection project richland washington savannah srnl abstract document documents rights
reserved copyright""".split())

# report/program boilerplate: appears in gov't reports but never as technical
# content; letting it into topics creates NPMI=-1 grab-bag topics
REPORT_STOP = set("""tasked task statement statements procedures procedure personnel
contractors contractor annual summary report reports program programs project
projects plan plans revision revisions chapter section appendix attachment cover
page pages prepared preparation submitted issued conducted completed performed
activities activity effort funded funding fiscal fy budget cost costs schedule
milestone milestones deliverable deliverables objective objectives purpose""".split())
STOP |= REPORT_STOP

TOKEN_RE = re.compile(r"[a-z][a-z\-]{2,}")

def tokenize(text):
    toks = TOKEN_RE.findall(text.lower())
    return [t for t in toks if t not in STOP and not t.isdigit() and len(t) > 3 or
            (t in ("tc", "so3", "na2o", "al2o3", "b2o3") )]

def main():
    docs = []
    n_skipped_tangential = 0
    for line in open(os.path.join(WS, "data", "records.jsonl")):
        d = json.loads(line)
        # train on the curated tiers only; the ~11k 'tangential' records are
        # query noise (particle physics, concrete, general energy) by design
        if d.get("relevance") == "tangential":
            n_skipped_tangential += 1
            continue
        title = d.get("title") or ""
        abstract = d.get("abstract") or ""
        # title words weigh 3x by repetition (standard trick)
        text = (title + " ") * 3 + abstract
        toks = tokenize(text)
        if len(toks) < 8:
            continue
        docs.append({
            "id": d["id"],
            "year": d.get("year"),
            "relevance": d.get("relevance", ""),
            "type": d.get("type", ""),
            "sources": d.get("_sources", []),
            "title": title,
            "tokens": toks,
        })
    print(f"skipped {n_skipped_tangential} tangential records")

    # frequency-based vocab pruning
    df = Counter()
    tf = Counter()
    for d in docs:
        seen = set(d["tokens"])
        df.update(seen)
        tf.update(d["tokens"])
    n = len(docs)
    # keep terms in 0.4% .. 25% of docs, at least 12 occurrences
    vocab = sorted(w for w, c in df.items() if c >= max(12, 0.004 * n) and c <= 0.25 * n)
    vocab_set = set(vocab)
    for d in docs:
        d["tokens"] = [t for t in d["tokens"] if t in vocab_set]
    docs = [d for d in docs if len(d["tokens"]) >= 8]

    with open(os.path.join(OUT_DIR, "corpus_tokens.json"), "w") as f:
        json.dump(docs, f)
    with open(os.path.join(OUT_DIR, "vocab.json"), "w") as f:
        json.dump(vocab, f)

    total_tok = sum(len(d["tokens"]) for d in docs)
    print(f"docs kept: {len(docs)} / 17946")
    print(f"vocab: {len(vocab)}")
    print(f"tokens: {total_tok/1e6:.2f}M (mean {total_tok/len(docs):.0f}/doc)")
    print("top-20 kept terms:", tf.most_common(20))

if __name__ == "__main__":
    main()
