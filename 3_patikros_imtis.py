"""
3 ŽINGSNIS. Rankinės vertimo patikros imtis.

Atsitiktinai (sėkla 42) atrenka 50 porų proporcingai poaibiams:
    K 10, N 10, J 15, R 15
ir sukuria duomenys/patikra.csv. Jį atsidarykite Excel ar Numbers programoje
ir užpildykite tris stulpelius:
    prasme               3 - išsaugota visiškai, 2 - iš dalies, 1 - prarasta
    klase_islaikyta      taip / ne
    struktura_islaikyta  taip / ne
Jei klase_islaikyta = "ne", užklausa bus automatiškai pašalinta iš analizės
visuose scenarijuose (S1, S2, S3).

Jei patikra.csv jau yra, jis neperrašomas (kad neprarastumėte įvertinimų).

Paleidimas:  python 3_patikros_imtis.py
"""
import sys

import pandas as pd

from bendra import (RINKINYS, PATIKRA, PATIKROS_STULPELIAI, RANDOM_SEED,
                    skaityti_csv, rasyti_csv)

IMTIS = {"K": 10, "N": 10, "J": 15, "R": 15}


def main():
    if PATIKRA.exists() and "--perrasyti" not in sys.argv:
        print(f"{PATIKRA} jau yra. Norėdami sukurti iš naujo, pridėkite --perrasyti")
        return

    df = skaityti_csv(RINKINYS)
    if (df["tekstas_lt"].str.strip() == "").any():
        print("Dėmesio: ne visos užklausos išverstos. Pirmiausia paleiskite 2_versti.py")

    dalys = [df[df["poaibis"] == p].sample(n=k, random_state=RANDOM_SEED)
             for p, k in IMTIS.items()]
    imtis = pd.concat(dalys).sort_values("id")
    for stulp in ["prasme", "klase_islaikyta", "struktura_islaikyta", "pastabos"]:
        imtis[stulp] = ""
    rasyti_csv(imtis[PATIKROS_STULPELIAI], PATIKRA)
    print(f"Sukurta {len(imtis)} porų patikros imtis: {PATIKRA}")


if __name__ == "__main__":
    main()
