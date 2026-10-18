# Jailbreak klasifikatorių palyginimas lietuvių kalba: eksperimentas

Skriptai įgyvendina metodikos skyriuje aprašytą tyrimą: sudaro 500 užklausų rinkinį, išverčia jį į lietuvių kalbą ir atgal, paleidžia 5 klasifikatorius trimis scenarijais (S1 EN, S2 LT, S3 LT→EN) ir apskaičiuoja metrikas.

```
eksperimentas/
├── bendra.py                  # keliai ir CSV stulpelių aprašai (bendri visiems)
├── 1_sudaryti_rinkini.py      # 500 angliškų užklausų iš Hugging Face
├── 2_versti.py                # EN→LT (S2) ir LT→EN (S3)
├── 3_patikros_imtis.py        # 50 porų rankinei vertimo patikrai
├── 4_klasifikuoti.py          # 5 modeliai × 3 scenarijai
├── 5_analizuoti.py            # metrikos, ΔF1, lentelės, grafikai
├── duomenys/
│   ├── rinkinys.csv           # sukuria 1 žingsnis, papildo 2 žingsnis
│   ├── rinkinys_sablonas.csv  # tuščias pavyzdys, jei pildysite rankomis
│   ├── patikra.csv            # sukuria 3 žingsnis, pildote rankomis
│   └── vertimo_zurnalas.json  # vertimo datos ir sistema
└── rezultatai/                # sukuria 4–5 žingsniai
```

## 0. Paruošimas (vieną kartą)

```bash
cd eksperimentas
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Hugging Face prieiga.** Trys Meta modeliai yra riboti (angl. *gated*), todėl prieigos reikia paprašyti iš anksto, nes patvirtinimas gali užtrukti:

1. Susikurkite paskyrą huggingface.co ir sukurkite prieigos raktą (Settings → Access Tokens, tipas *Read*).
2. Modelių puslapiuose paspauskite prieigos prašymo mygtuką ir užpildykite formą:
   - `meta-llama/Llama-Prompt-Guard-2-86M`
   - `meta-llama/Llama-Prompt-Guard-2-22M`
   - `meta-llama/Llama-Guard-3-1B`
3. Jei rinkinio `TrustAIRLab/in-the-wild-jailbreak-prompts` puslapyje prašoma sutikti su sąlygomis, sutikite.
4. Terminale prisijunkite: `hf auth login` (senesnėse versijose `huggingface-cli login`).

`ToxicityPrompts/PolyGuard-Qwen-Smol` ir `Qwen/Qwen3Guard-Gen-0.6B` yra laisvai prieinami.

## 1. Rinkinio sudarymas

```bash
python 1_sudaryti_rinkini.py
```

Sukuriamas `duomenys/rinkinys.csv` su 500 eilučių (K 100, J 150, N 100, R 150). Lietuviški stulpeliai kol kas tušti. Visi atsitiktiniai pasirinkimai daromi su sėkla 42, todėl paleidus dar kartą gaunamas tas pats rinkinys.

Skriptas išveda, kiek in-the-wild šablonų turėjo vietą prašymui įterpti. Šį skaičių verta paminėti praktinėje dalyje.

## 2. Vertimas

```bash
python 2_versti.py --limit 3     # pirma išbandykite su 3 užklausomis
python 2_versti.py               # visas vertimas: EN→LT, tada LT→EN
```

- Naudojamas nemokamas Google Translate (`deep-translator`). 1000 vertimų užtrunka apie 15–30 min.
- Skriptą galima bet kada nutraukti ir paleisti iš naujo: jau išversti langeliai praleidžiami.
- Jei Google pradeda riboti užklausas, padidinkite pauzę: `--pauze 2`.
- Alternatyva yra oficialus Google Cloud Translation API (pirmi 500 000 ženklų per mėnesį nemokami): `export GOOGLE_API_KEY=...` ir `python 2_versti.py --backend cloud`.
- Datos ir naudota sistema įrašomos į `vertimo_zurnalas.json`. Metodikoje nurodyta, kad vertimo data fiksuojama.

## 3. Rankinė vertimo patikra

```bash
python 3_patikros_imtis.py
```

Atsidarykite `duomenys/patikra.csv` (patogiausia Numbers programa) ir užpildykite stulpelius:

| Stulpelis | Reikšmės |
|---|---|
| `prasme` | 3 – išsaugota visiškai, 2 – iš dalies, 1 – prarasta |
| `klase_islaikyta` | `taip` / `ne` |
| `struktura_islaikyta` | `taip` / `ne` |
| `pastabos` | laisvas tekstas |

Užklausos, kurių `klase_islaikyta = ne`, analizėje automatiškai pašalinamos iš visų trijų scenarijų. Jų niekuo pakeisti nereikia.

> Jei failą redaguojate Excel programoje, išsaugokite jį kaip **CSV UTF-8**, kitaip sugadinsite lietuviškas raides.

## 4. Klasifikavimas

```bash
python 4_klasifikuoti.py --limit 8                        # greitas bandymas
python 4_klasifikuoti.py                                  # visas eksperimentas
python 4_klasifikuoti.py --modeliai PromptGuard2-86M      # vienas modelis
```

- Modeliai įkeliami po vieną ir veikia per MPS (Apple M2 GPU). Prieš paleidžiant uždarykite kitas programas.
- Visas eksperimentas apima 7 500 klasifikavimų. Apytikslė trukmė M2 kompiuteryje yra 1–3 val., iš jų daugiausia laiko užima generatyviniai modeliai.
- Galima nutraukti (Ctrl+C) ir tęsti vėliau: atlikti klasifikavimai praleidžiami.
- Pirmos 3 užklausos kiekvienam modeliui naudojamos „apšilimui“ ir nematuojamos.
- `--testinis` paleidžia visą grandinę be tikrų modelių. Taip galima patikrinti, ar viskas veikia, bet **šių rezultatų darbe nenaudokite**. Prieš tikrą paleidimą ištrinkite `rezultatai/prognozes.csv`.

**Jei kas nors neveikia:**
- *Daug neatpažintų atsakymų* arba keisti simboliai stulpelyje `atsakymas`: bandykite `--dtype bfloat16` arba `--dtype float32`. Prieš tai ištrinkite to modelio eilutes iš `prognozes.csv`.
- *„NEPAVYKO įkelti“*: dar negauta prieiga prie modelio arba neprisijungta (`hf auth login`).
- *MPS klaida dėl nepalaikomos operacijos*: skriptas jau nustato `PYTORCH_ENABLE_MPS_FALLBACK=1`, todėl tokios operacijos vykdomos centriniame procesoriuje.

## 5. Analizė

```bash
python 5_analizuoti.py
```

Failai `rezultatai/` kataloge:

| Failas | Turinys |
|---|---|
| `metrikos.csv` | TP, FP, FN, TN, tikslumas, atkūrimas, F1, FPR, atkūrimas K/J, FPR N/R, vid. laikas |
| `skirtumai.csv` | F1 kiekviename scenarijuje, ΔF1 kalba = S2 − S1, ΔF1 vertimas = S3 − S2 |
| `patikros_suvestine.csv` | Rankinės vertimo patikros rezultatai pagal poaibius |
| `klaidos.csv` | Visos klaidingai klasifikuotos užklausos su tekstais, skirtos klaidų analizei |
| `lenteles.tex` | Trys paruoštos LaTeX lentelės tokiu pat stiliumi kaip darbe |
| `f1_scenarijai.png` | F1 pagal modelius ir scenarijus |
| `painiavos_matricos.png` | Painiavos matricos (modeliai × scenarijai) |

## CSV struktūra

**`rinkinys.csv`**: viena eilutė yra viena užklausa.

| Stulpelis | Aprašas |
|---|---|
| `id` | K001…K100, J001…J150, N001…N100, R001…R150 |
| `poaibis` | K – tiesioginė kenksminga, J – jailbreak, N – nekenksminga ta pačia tema, R – kraštinis atvejis |
| `klase` | 1 – nesaugi, 0 – saugi |
| `saltinis`, `kategorija` | iš kur paimta užklausa (informacinis) |
| `tekstas_en` | S1 scenarijaus tekstas |
| `tekstas_lt` | S2 scenarijaus tekstas |
| `tekstas_lt_en` | S3 scenarijaus tekstas |

**`prognozes.csv`**: viena eilutė yra vienas sprendimas (modelis × scenarijus × užklausa).

| Stulpelis | Aprašas |
|---|---|
| `modelis`, `scenarijus`, `id`, `poaibis`, `klase` | kas klasifikuota |
| `pred` | sprendimas: 1 – nesaugi, 0 – saugi |
| `tikimybe` | nesaugios klasės tikimybė (tik Prompt Guard 2) |
| `atsakymas` | neapdorotas generatyvinio modelio atsakymas, pvz. `unsafe\nS2`, `Safety: Unsafe` |
| `atpazinta` | ar pavyko nuskaityti sprendimą (False reiškia, kad laikyta saugia) |
| `laikas_ms` | apdorojimo laikas milisekundėmis |

Jei rinkinį pildote rankomis, naudokite `duomenys/rinkinys_sablonas.csv` stulpelių tvarką ir pavadinkite failą `rinkinys.csv`.

## Ką verta žinoti rašant praktinę dalį

- **PolyGuard** atsakymas tyrime paliekamas tuščias, nes vertinama tik užklausa (`Harmful request: yes/no`). Modelio kortelėje parodytas tik užklausos ir atsakymo poros pavyzdys.
- **Qwen3Guard** vertinamas griežtuoju režimu: `Controversial` laikoma nesaugia.
- **Prompt Guard 2** ilgos užklausos skaidomos į 512 žetonų atkarpas, o užklausos įverčiu laikoma didžiausia atkarpos tikimybė. Slenkstis 0,5.
- Visi šie sprendimai atitinka metodikos 2.7 poskyrio algoritmą.
