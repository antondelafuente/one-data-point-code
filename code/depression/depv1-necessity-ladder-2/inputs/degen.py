import json, re
from collections import Counter
def hard_loop(t):
    lines=[l for l in t.split("\n") if l.strip()]
    if lines and Counter(lines).most_common(1)[0][1]/len(lines) > 0.5: return "line-loop"
    words=re.findall(r"\S+", t)
    for n in (3,5):  # any n-gram immediately repeated >15x
        grams=[" ".join(words[i:i+n]) for i in range(0,len(words)-n)]
        run=1
        for i in range(1,len(grams)):
            run = run+1 if grams[i]==grams[i-n] else 1
            if run>15: return "ngram-loop"
    return None
if __name__=="__main__":
    import sys
    n=0
    for l in open(sys.argv[1]):
        r=json.loads(l); v=hard_loop(r["trajectory"])
        if v: n+=1; print(v, r["subject"], r["id"])
    print("flagged:",n)
