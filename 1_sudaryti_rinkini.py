"""
1 ŽINGSNIS. Vertinimo rinkinio sudarymas (500 angliškų užklausų).

Atsisiunčia tris viešus rinkinius iš Hugging Face ir sudaro keturis poaibius:
    K  100  JBB-Behaviors, harmful      klasė 1
    J  150  in-the-wild jailbreak       klasė 1  (šablonai su įterpta JBB elgsena)
    N  100  JBB-Behaviors, benign       klasė 0
    R  150  OR-Bench-Hard-1K            klasė 0

Rezultatas: duomenys/rinkinys.csv (lietuviški stulpeliai kol kas tušti).

Paleidimas:  python 1_sudaryti_rinkini.py
"""
import random
import re

import pandas as pd
from datasets import load_dataset

from bendra import RINKINYS, RINKINIO_STULPELIAI, RANDOM_SEED, rasyti_csv

J_KIEKIS = 150
R_KIEKIS = 150
MAKS_ILGIS = 3000  # ženklų

# Vieta jailbreak šablone, į kurią įrašomas tikrasis prašymas,
# pvz. [INSERT PROMPT HERE], {prompt}, <your question>, [QUESTION]
VIETA = re.compile(
    r"[\[\{<]\s*(?:insert[^\]\}>]{0,40}|prompt|your (?:prompt|question|request)[^\]\}>]{0,20}"
    r"|question|query|request)\s*[\]\}>]",
    re.IGNORECASE,
)
DAZNI_ZODZIAI = {"the", "and", "you", "to", "of", "is", "a", "in", "your", "will"}


def anglu_kalba(tekstas):
    """Paprasta patikra: beveik visos raidės lotyniškos ir yra dažnų angliškų žodžių."""
    raides = [c for c in tekstas if c.isalpha()]
    if not raides:
        return False
    ascii_dalis = sum(c.isascii() for c in raides) / len(raides)
    zodziai = re.findall(r"[a-z]+", tekstas.lower())
    anglisku = sum(z in DAZNI_ZODZIAI for z in zodziai) / max(len(zodziai), 1)
    return ascii_dalis >= 0.97 and anglisku >= 0.05


def normalizuoti(tekstas):
    return re.sub(r"\s+", " ", tekstas).strip().lower()


def main():
    rng = random.Random(RANDOM_SEED)
    eilutes = []

    # --- K ir N: JBB-Behaviors -------------------------------------------------
    jbb_k = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful").to_pandas()
    jbb_n = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="benign").to_pandas()

    for i, r in enumerate(jbb_k.itertuples(), 1):
        eilutes.append(dict(id=f"K{i:03d}", poaibis="K", klase=1, saltinis="JBB-harmful",
                            kategorija=r.Category, tekstas_en=r.Goal))

    # --- J: in-the-wild jailbreak šablonai ------------------------------------
    itw = load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts",
                       "jailbreak_2023_12_25", split="train").to_pandas()
    itw = itw[itw["jailbreak"] == True]  # noqa: E712
    itw = itw[itw["prompt"].str.len() <= MAKS_ILGIS]
    itw = itw[itw["prompt"].apply(anglu_kalba)]
    itw = itw.assign(_norm=itw["prompt"].apply(normalizuoti)).drop_duplicates("_norm")
    print(f"In-the-wild: po filtrų liko {len(itw)} užklausų, atrenkama {J_KIEKIS}")

    j_imtis = itw.sample(n=J_KIEKIS, random_state=RANDOM_SEED)
    elgsenos = jbb_k["Goal"].tolist()
    su_vieta = 0
    for i, r in enumerate(j_imtis.itertuples(), 1):
        sablonas = r.prompt
        kategorija = ""
        if VIETA.search(sablonas):
            elgsena_idx = rng.randrange(len(elgsenos))
            sablonas = VIETA.sub(lambda _: elgsenos[elgsena_idx], sablonas)
            kategorija = jbb_k.iloc[elgsena_idx]["Category"]
            su_vieta += 1
        eilutes.append(dict(id=f"J{i:03d}", poaibis="J", klase=1, saltinis="in-the-wild",
                            kategorija=kategorija, tekstas_en=sablonas))
    print(f"J poaibis: {su_vieta} šablonų su įterpta JBB elgsena, "
          f"{J_KIEKIS - su_vieta} naudojami nepakeisti")

    for i, r in enumerate(jbb_n.itertuples(), 1):
        eilutes.append(dict(id=f"N{i:03d}", poaibis="N", klase=0, saltinis="JBB-benign",
                            kategorija=r.Category, tekstas_en=r.Goal))

    # --- R: OR-Bench-Hard-1K ----------------------------------------------------
    orb = load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train").to_pandas()
    r_imtis = orb.sample(n=R_KIEKIS, random_state=RANDOM_SEED)
    for i, r in enumerate(r_imtis.itertuples(), 1):
        eilutes.append(dict(id=f"R{i:03d}", poaibis="R", klase=0, saltinis="OR-Bench-Hard-1K",
                            kategorija=r.category, tekstas_en=r.prompt))

    df = pd.DataFrame(eilutes)
    df["tekstas_lt"] = ""
    df["tekstas_lt_en"] = ""
    rasyti_csv(df[RINKINIO_STULPELIAI], RINKINYS)
    print(f"\nĮrašyta: {RINKINYS}")
    print(df.groupby(["poaibis", "klase"]).size().to_string())


if __name__ == "__main__":
    main()
