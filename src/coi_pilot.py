"""(10) Pilot kit: private conflict-of-interest check between two offices (commutative encryption PSI).

Party A = procurement authority (holds the list of bidders on a tender).
Party B = ethics / integrity office (holds firms linked to evaluation-committee members,
          e.g. from asset declarations or the company register).
Goal: A learns which bidders are on B's list; neither party sees the rest of the other's list.

Each party runs its own steps on its own machine and exchanges only blinded files:

  A1  python coi_pilot.py a-blind  bidders.txt         -> a_secret.key, a_to_b.json
  B1  python coi_pilot.py b-blind  related_firms.txt a_to_b.json
                                                       -> b_secret.key, b_to_a.json
  A2  python coi_pilot.py a-match  b_to_a.json         -> prints the overlapping bidders

Inputs are plain text, one firm per line. Names are normalised (case, punctuation, legal
suffixes) before hashing, so "ABC Ltd" matches "A.B.C. Limited". Better: use company
registration numbers on both sides when available.

Maths: values are hashed into the quadratic-residue subgroup of the RFC 3526 2048-bit safe-prime
group; E_k(y) = y^k mod p. Exponentiation commutes, so E_a(E_b(y)) = E_b(E_a(y)): equal
double-blinded values identify the same firm. Security relies on the Decisional Diffie-Hellman
assumption. Lists are shuffled so positions leak nothing. Keys never leave their owner.

Pilot checklist (for the partner institution): legal basis under the Data Protection and Privacy
Act 2019; agreed normalisation; key storage; a test run on dummy lists; independent review of code.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blind_eval as be  # noqa: E402

SUFFIX = r"\b(limited|ltd|co|company|uganda|u|smc|enterprises?|ent|and|&)\b"


def norm(name: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", name.lower().replace("&amp;", "&"))
    return re.sub(r"\s+", "", re.sub(SUFFIX, " ", s))


def read_list(p):
    return [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]


def a_blind(bidders_file):
    k = be.keygen()
    Path("a_secret.key").write_text(str(k))
    names = read_list(bidders_file)
    blinded = [dict(id=i, v=hex(be.blind(be.h(norm(n)), k))) for i, n in enumerate(names)]
    Path("a_index.json").write_text(json.dumps({str(i): n for i, n in enumerate(names)}))  # stays with A
    random.shuffle(blinded)
    Path("a_to_b.json").write_text(json.dumps(blinded))
    print(f"A: blinded {len(names)} bidders -> a_to_b.json (send to B). Keep a_secret.key and a_index.json.")


def b_blind(related_file, a_to_b):
    k = be.keygen()
    Path("b_secret.key").write_text(str(k))
    from_a = json.loads(Path(a_to_b).read_text())
    double_a = [dict(id=x["id"], v=hex(be.blind(int(x["v"], 16), k))) for x in from_a]
    mine = [hex(be.blind(be.h(norm(n)), k)) for n in read_list(related_file)]
    random.shuffle(mine)
    Path("b_to_a.json").write_text(json.dumps(dict(double_blinded_bidders=double_a, b_blinded=mine)))
    print(f"B: returned {len(double_a)} double-blinded bidders and {len(mine)} blinded own entries -> b_to_a.json")


def a_match(b_to_a):
    k = int(Path("a_secret.key").read_text())
    idx = json.loads(Path("a_index.json").read_text())
    msg = json.loads(Path(b_to_a).read_text())
    b_double = {hex(be.blind(int(v, 16), k)) for v in msg["b_blinded"]}
    hits = [idx[str(x["id"])] for x in msg["double_blinded_bidders"] if x["v"] in b_double]
    print("Conflicts of interest found:" if hits else "No overlap found.")
    for h in hits:
        print(" -", h)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"a-blind": lambda: a_blind(sys.argv[2]), "b-blind": lambda: b_blind(sys.argv[2], sys.argv[3]),
     "a-match": lambda: a_match(sys.argv[2])}.get(cmd, lambda: print(__doc__))()
