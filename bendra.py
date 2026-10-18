"""
Bendri nustatymai ir pagalbinės funkcijos visiems skriptams.
Čia aprašyta CSV failų struktūra - jei norite ką nors pakeisti, keiskite tik čia.
"""
import os
from pathlib import Path

import pandas as pd

# --- Katalogai ---------------------------------------------------------------
SAKNIS = Path(__file__).resolve().parent
DUOMENYS = SAKNIS / "duomenys"
REZULTATAI = SAKNIS / "rezultatai"

RINKINYS = DUOMENYS / "rinkinys.csv"            # 500 užklausų su vertimais
PATIKRA = DUOMENYS / "patikra.csv"              # 50 porų rankinė vertimo patikra
VERTIMO_ZURNALAS = DUOMENYS / "vertimo_zurnalas.json"
PROGNOZES = REZULTATAI / "prognozes.csv"        # visų modelių sprendimai

RANDOM_SEED = 42

# --- rinkinys.csv stulpeliai -------------------------------------------------
# id            unikalus užklausos kodas, pvz. K001, J017, N100, R150
# poaibis       K - tiesioginė kenksminga, J - jailbreak,
#               N - nekenksminga ta pačia tema, R - kraštinis atvejis (OR-Bench)
# klase         1 - nesaugi, 0 - saugi
# saltinis      iš kurio rinkinio paimta (JBB-harmful, in-the-wild, JBB-benign, OR-Bench-Hard-1K)
# kategorija    žalos kategorija iš šaltinio (informacinė)
# tekstas_en    originali angliška užklausa              -> scenarijus S1
# tekstas_lt    išversta į lietuvių kalbą                 -> scenarijus S2
# tekstas_lt_en lietuviška versija išversta atgal į anglų -> scenarijus S3
RINKINIO_STULPELIAI = ["id", "poaibis", "klase", "saltinis", "kategorija",
                       "tekstas_en", "tekstas_lt", "tekstas_lt_en"]

SCENARIJAI = {"S1": "tekstas_en", "S2": "tekstas_lt", "S3": "tekstas_lt_en"}

# --- patikra.csv stulpeliai (pildomi rankomis) --------------------------------
# prasme               3 - išsaugota visiškai, 2 - iš dalies, 1 - prarasta
# klase_islaikyta      taip / ne  (jei "ne" - užklausa pašalinama iš analizės)
# struktura_islaikyta  taip / ne
# pastabos             laisvas tekstas
PATIKROS_STULPELIAI = ["id", "poaibis", "klase", "tekstas_en", "tekstas_lt",
                       "prasme", "klase_islaikyta", "struktura_islaikyta", "pastabos"]

# --- prognozes.csv stulpeliai ------------------------------------------------
# modelis     klasifikatoriaus trumpas pavadinimas
# scenarijus  S1, S2 arba S3
# pred        modelio sprendimas: 1 - nesaugi, 0 - saugi
# tikimybe    nesaugios klasės tikimybė (tik Prompt Guard 2, kitiems tuščia)
# atsakymas   neapdorotas modelio atsakymas (generatyviniams modeliams)
# atpazinta   ar pavyko nuskaityti sprendimą (False -> pred = 0)
# laikas_ms   vienos užklausos apdorojimo laikas milisekundėmis
PROGNOZIU_STULPELIAI = ["modelis", "scenarijus", "id", "poaibis", "klase",
                        "pred", "tikimybe", "atsakymas", "atpazinta", "laikas_ms"]


def skaityti_csv(kelias):
    """CSV skaitomas kaip tekstas, kad tušti langeliai liktų tušti, o ne NaN."""
    return pd.read_csv(kelias, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def rasyti_csv(df, kelias):
    """Atominis įrašymas: pirma į laikiną failą, tada pervadinama.
    utf-8-sig koduotė - kad Excel teisingai rodytų lietuviškas raides."""
    kelias = Path(kelias)
    kelias.parent.mkdir(parents=True, exist_ok=True)
    laikinas = kelias.with_suffix(".tmp")
    df.to_csv(laikinas, index=False, encoding="utf-8-sig")
    os.replace(laikinas, kelias)


def pasalintini_id():
    """Užklausos, kurių vertimas rankinėje patikroje pakeitė klasę."""
    if not PATIKRA.exists():
        return set()
    p = skaityti_csv(PATIKRA)
    return set(p.loc[p["klase_islaikyta"].str.strip().str.lower() == "ne", "id"])
