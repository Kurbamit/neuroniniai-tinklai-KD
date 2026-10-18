"""
2 ŽINGSNIS. Užklausų vertimas S2 ir S3 scenarijams.

    S2:  tekstas_en    -> tekstas_lt      (anglų -> lietuvių)
    S3:  tekstas_lt    -> tekstas_lt_en   (lietuvių -> anglų, atgalinis vertimas)

Abiem kryptimis naudojama ta pati vertimo sistema (Google Translate).
Skriptą galima nutraukti ir paleisti iš naujo - jau išversti langeliai praleidžiami.

Paleidimas:
    python 2_versti.py                    # abi kryptys, nemokamas Google Translate
    python 2_versti.py --kryptis lt       # tik EN -> LT
    python 2_versti.py --kryptis atgal    # tik LT -> EN
    python 2_versti.py --backend cloud    # oficialus Google Cloud Translation API
                                          # (reikia GOOGLE_API_KEY aplinkos kintamojo)
    python 2_versti.py --limit 5          # išbandyti su 5 užklausomis
"""
import argparse
import json
import os
import time
from datetime import datetime

from bendra import RINKINYS, VERTIMO_ZURNALAS, skaityti_csv, rasyti_csv

MAKS_GABALAS = 4500  # Google Translate riba ~5000 ženklų vienai užklausai
ISSAUGOTI_KAS = 10   # kas kiek eilučių įrašyti į diską


# --- Vertimo sistemos ----------------------------------------------------------
def sukurti_vertikli(backend, saltinis, tikslas):
    if backend == "google":
        from deep_translator import GoogleTranslator
        vert = GoogleTranslator(source=saltinis, target=tikslas)
        return lambda t: vert.translate(t) or ""

    if backend == "cloud":
        import requests
        raktas = os.environ["GOOGLE_API_KEY"]
        url = "https://translation.googleapis.com/language/translate/v2"

        def cloud(t):
            r = requests.post(url, params={"key": raktas}, timeout=60,
                              data={"q": t, "source": saltinis, "target": tikslas,
                                    "format": "text"})
            r.raise_for_status()
            return r.json()["data"]["translations"][0]["translatedText"]
        return cloud

    if backend == "testinis":  # be interneto, tik grandinei patikrinti
        return lambda t: f"[{tikslas}] {t}"

    raise ValueError(backend)


def suskaidyti(tekstas):
    """Ilgas tekstas skaidomas pagal eilutes į gabalus iki MAKS_GABALAS ženklų.
    Grąžinamos poros (gabalas, jungtis), kad "".join(g + j) tiksliai atkurtų tekstą."""
    if len(tekstas) <= MAKS_GABALAS:
        return [(tekstas, "")]
    vienetai = []
    eilutes = tekstas.split("\n")
    for i, eilute in enumerate(eilutes):
        jungtis = "\n" if i < len(eilutes) - 1 else ""
        while len(eilute) > MAKS_GABALAS:            # labai ilga eilutė be lūžių
            vienetai.append((eilute[:MAKS_GABALAS], ""))
            eilute = eilute[MAKS_GABALAS:]
        vienetai.append((eilute, jungtis))

    gabalai, dabartinis, jung = [], None, ""
    for eilute, jungtis in vienetai:
        if dabartinis is not None and len(dabartinis) + len(jung) + len(eilute) > MAKS_GABALAS:
            gabalai.append((dabartinis, jung))
            dabartinis = None
        dabartinis = eilute if dabartinis is None else dabartinis + jung + eilute
        jung = jungtis
    gabalai.append((dabartinis, jung))
    return gabalai


def versti(tekstas, vertiklis, pauze, bandymai=5):
    if not tekstas.strip():
        return tekstas
    rezultatas = []
    for gabalas, jungtis in suskaidyti(tekstas):
        if not gabalas.strip():
            rezultatas.append(gabalas + jungtis)
            continue
        for bandymas in range(bandymai):
            try:
                rezultatas.append(vertiklis(gabalas) + jungtis)
                break
            except Exception as e:  # tinklo klaida, per daug užklausų ir pan.
                laukti = 5 * 2 ** bandymas
                print(f"   klaida ({e.__class__.__name__}: {e}); kartojama po {laukti} s")
                time.sleep(laukti)
        else:
            raise RuntimeError("Nepavyko išversti po kelių bandymų")
        time.sleep(pauze)
    return "".join(rezultatas)


# --- Pagrindinė dalis ----------------------------------------------------------
def versti_stulpeli(df, is_stulp, i_stulp, saltinis, tikslas, args):
    vertiklis = sukurti_vertikli(args.backend, saltinis, tikslas)
    liko = df.index[(df[i_stulp].str.strip() == "") & (df[is_stulp].str.strip() != "")]
    if args.limit:
        liko = liko[: args.limit]
    print(f"\n{is_stulp} -> {i_stulp}: verčiama {len(liko)} užklausų")

    for n, idx in enumerate(liko, 1):
        df.at[idx, i_stulp] = versti(df.at[idx, is_stulp], vertiklis, args.pauze)
        print(f"   {n}/{len(liko)}  {df.at[idx, 'id']}")
        if n % ISSAUGOTI_KAS == 0:
            rasyti_csv(df, RINKINYS)
    rasyti_csv(df, RINKINYS)
    return len(liko)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kryptis", choices=["lt", "atgal", "abi"], default="abi")
    ap.add_argument("--backend", choices=["google", "cloud", "testinis"], default="google")
    ap.add_argument("--pauze", type=float, default=0.5, help="sekundės tarp užklausų")
    ap.add_argument("--limit", type=int, default=0, help="išversti tik N užklausų (bandymui)")
    args = ap.parse_args()

    df = skaityti_csv(RINKINYS)
    zurnalas = json.loads(VERTIMO_ZURNALAS.read_text("utf-8")) if VERTIMO_ZURNALAS.exists() else []

    for kryptis, is_s, i_s, sal, tik in [("lt", "tekstas_en", "tekstas_lt", "en", "lt"),
                                         ("atgal", "tekstas_lt", "tekstas_lt_en", "lt", "en")]:
        if args.kryptis in (kryptis, "abi"):
            kiekis = versti_stulpeli(df, is_s, i_s, sal, tik, args)
            if kiekis:
                zurnalas.append({"data": datetime.now().isoformat(timespec="seconds"),
                                 "kryptis": f"{sal}->{tik}", "sistema": args.backend,
                                 "uzklausu": kiekis})

    VERTIMO_ZURNALAS.write_text(json.dumps(zurnalas, ensure_ascii=False, indent=2), "utf-8")
    tusti = {s: int((df[s].str.strip() == "").sum()) for s in ["tekstas_lt", "tekstas_lt_en"]}
    print(f"\nBaigta. Neišverstų liko: {tusti}")


if __name__ == "__main__":
    main()
