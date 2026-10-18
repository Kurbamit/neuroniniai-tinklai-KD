"""
5 ŽINGSNIS. Rezultatų analizė pagal metodiką.

Iš rezultatai/prognozes.csv apskaičiuoja:
    - tikslumą, atkūrimą, F1, FPR kiekvienam modeliui ir scenarijui (formulės 2.6 poskyryje)
    - atkūrimą K ir J poaibiams, FPR - N ir R poaibiams
    - vidutinį apdorojimo laiką
    - ΔF1 kalba = F1(S2) - F1(S1),  ΔF1 vertimas = F1(S3) - F1(S2)
    - rankinės vertimo patikros suvestinę
Užklausos, kurių vertimas pakeitė klasę (patikra.csv: klase_islaikyta = ne),
pašalinamos iš visų scenarijų.

Rezultatai rašomi į rezultatai/:
    metrikos.csv, skirtumai.csv, patikros_suvestine.csv, klaidos.csv,
    lenteles.tex (paruoštos LaTeX lentelės), f1_scenarijai.png, painiavos_matricos.png

Paleidimas:  python 5_analizuoti.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bendra import (RINKINYS, PATIKRA, PROGNOZES, REZULTATAI, SCENARIJAI,
                    skaityti_csv, rasyti_csv, pasalintini_id)

# Spalvos: patikrinta kategorinė paletė (3 scenarijai) ir vieno atspalvio skalė matricoms
SPALVOS = {"S1": "#2a78d6", "S2": "#eb6834", "S3": "#1baf7a"}
MELYNA_SKALE = ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
TEKSTAS, TEKSTAS_2, TINKLELIS = "#1f1f1e", "#5f5e5a", "#e4e3df"
SCENARIJU_PAVADINIMAI = {"S1": "S1 (EN)", "S2": "S2 (LT)", "S3": "S3 (LT→EN)"}


def dalyba(a, b):
    return a / b if b else 0.0


def metrikos(d):
    y, p = d["klase"].to_numpy(), d["pred"].to_numpy()
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    tikslumas, atkurimas = dalyba(tp, tp + fp), dalyba(tp, tp + fn)
    f1 = dalyba(2 * tikslumas * atkurimas, tikslumas + atkurimas)

    def poaibio(poaibis, kas):
        s = d[d["poaibis"] == poaibis]
        if s.empty:
            return np.nan
        return s["pred"].mean()  # K/J: dalis aptiktų (atkūrimas); N/R: dalis užblokuotų (FPR)

    return dict(n=len(d), TP=tp, FP=fp, FN=fn, TN=tn,
                tikslumas=tikslumas, atkurimas=atkurimas, F1=f1, FPR=dalyba(fp, fp + tn),
                atkurimas_K=poaibio("K", "a"), atkurimas_J=poaibio("J", "a"),
                FPR_N=poaibio("N", "f"), FPR_R=poaibio("R", "f"),
                laikas_ms=d["laikas_ms"].mean(), neatpazinta=int((~d["atpazinta"]).sum()))


def ikelti():
    p = skaityti_csv(PROGNOZES)
    p["klase"] = p["klase"].astype(int)
    p["pred"] = p["pred"].astype(int)
    p["laikas_ms"] = pd.to_numeric(p["laikas_ms"])
    p["atpazinta"] = p["atpazinta"].str.lower() == "true"
    p = p.drop_duplicates(["modelis", "scenarijus", "id"], keep="last")

    salinti = pasalintini_id()
    if salinti:
        print(f"Pašalinamos {len(salinti)} užklausos, kurių vertimas pakeitė klasę: "
              f"{', '.join(sorted(salinti))}")
    p = p[~p["id"].isin(salinti)]

    # Porinis palyginimas: kiekvienam modeliui paliekamos tik užklausos, turinčios visus scenarijus
    scenariju_sk = p.groupby("modelis")["scenarijus"].nunique()
    pilnos = p.groupby(["modelis", "id"])["scenarijus"].transform("nunique") == \
        p["modelis"].map(scenariju_sk)
    if (~pilnos).any():
        print(f"Dėmesio: {(~pilnos).sum()} eilučių neturi visų scenarijų ir neįtraukiamos")
    return p[pilnos]


def patikros_suvestine():
    if not PATIKRA.exists():
        return None
    pk = skaityti_csv(PATIKRA)
    pk = pk[pk["prasme"].str.strip() != ""]
    if pk.empty:
        return None
    eil = []
    for poaibis, d in [("Visi", pk)] + list(pk.groupby("poaibis")):
        prasme = pd.to_numeric(d["prasme"], errors="coerce")
        eil.append(dict(poaibis=poaibis, poru=len(d),
                        prasme_3=int((prasme == 3).sum()), prasme_2=int((prasme == 2).sum()),
                        prasme_1=int((prasme == 1).sum()), prasme_vid=round(prasme.mean(), 2),
                        klase_pakito=int((d["klase_islaikyta"].str.lower() == "ne").sum()),
                        struktura_pakito=int((d["struktura_islaikyta"].str.lower() == "ne").sum())))
    return pd.DataFrame(eil)


# --- LaTeX lentelės (tokiu pat stiliumi kaip darbe) -----------------------------------
def sk(x, skaitmenys=3):
    if pd.isna(x):
        return "--"
    return f"{x:.{skaitmenys}f}".replace(".", ",").replace("-", "$-$")


def latex_lentele(df, stulpeliai, antrastes, pavadinimas, zyme, plociai):
    eil = [r"\begin{table}[H]",
           rf"  \caption{{{pavadinimas}}}\label{{{zyme}}}",
           r"  \lentelesTekstas", r"  \small",
           rf"  \begin{{tabularx}}{{\textwidth}}{{|{'|'.join(plociai)}|}}", r"    \hline",
           "    " + " & ".join(rf"\textbf{{{a}}}" for a in antrastes) + r" \\", r"    \hline"]
    for _, r in df.iterrows():
        eil.append("    " + " & ".join(str(r[s]) for s in stulpeliai) + r" \\")
        eil.append(r"    \hline")
    eil += [r"  \end{tabularx}", r"  \saltinis{Šaltinis: sudaryta autoriaus}", r"\end{table}", ""]
    return "\n".join(eil)


# --- Grafikai -------------------------------------------------------------------------
def stilius(ax):
    for k in ("top", "right", "left"):
        ax.spines[k].set_visible(False)
    ax.spines["bottom"].set_color(TINKLELIS)
    ax.tick_params(colors=TEKSTAS_2, length=0)
    ax.yaxis.grid(True, color=TINKLELIS, linewidth=0.8)
    ax.set_axisbelow(True)


def grafikas_f1(m, modeliai, scenarijai):
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=200)
    plotis = 0.8 / len(scenarijai)
    x = np.arange(len(modeliai))
    for i, sc in enumerate(scenarijai):
        reiksmes = [m.loc[(mod, sc), "F1"] for mod in modeliai]
        stulpai = ax.bar(x + (i - (len(scenarijai) - 1) / 2) * plotis, reiksmes,
                         plotis - 0.03, color=SPALVOS[sc], label=SCENARIJU_PAVADINIMAI[sc])
        ax.bar_label(stulpai, labels=[sk(v, 2) for v in reiksmes], padding=2,
                     fontsize=7, color=TEKSTAS_2)
    ax.set_xticks(x, modeliai, fontsize=8, color=TEKSTAS)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: sk(v, 1)))
    ax.set_ylabel("F1", color=TEKSTAS_2)
    stilius(ax)
    ax.legend(frameon=False, ncol=len(scenarijai), loc="upper center",
              bbox_to_anchor=(0.5, 1.12), fontsize=8, labelcolor=TEKSTAS)
    fig.tight_layout()
    fig.savefig(REZULTATAI / "f1_scenarijai.png")
    plt.close(fig)


def grafikas_matricos(m, modeliai, scenarijai):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("melyna", MELYNA_SKALE)
    fig, asys = plt.subplots(len(modeliai), len(scenarijai), dpi=200,
                             figsize=(2.3 * len(scenarijai), 2.1 * len(modeliai)), squeeze=False)
    for i, mod in enumerate(modeliai):
        for j, sc in enumerate(scenarijai):
            ax = asys[i, j]
            r = m.loc[(mod, sc)]
            mat = np.array([[r.TP, r.FP], [r.FN, r.TN]])
            ax.imshow(mat, cmap=cmap, vmin=0, vmax=max(1, r.n / 2))
            for a in range(2):
                for b in range(2):
                    sviesus = mat[a, b] > r.n / 4
                    ax.text(b, a, int(mat[a, b]), ha="center", va="center", fontsize=9,
                            color="white" if sviesus else TEKSTAS)
            ax.set_xticks([0, 1], ["Nesaugi", "Saugi"], fontsize=6, color=TEKSTAS_2)
            ax.set_yticks([0, 1], ["Nesaugi", "Saugi"], fontsize=6, color=TEKSTAS_2)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(SCENARIJU_PAVADINIMAI[sc], fontsize=8, color=TEKSTAS)
            if j == 0:
                ax.set_ylabel(f"{mod}\nprognozė", fontsize=7, color=TEKSTAS)
            if i == len(modeliai) - 1:
                ax.set_xlabel("tikroji klasė", fontsize=7, color=TEKSTAS_2)
    fig.tight_layout()
    fig.savefig(REZULTATAI / "klasifikavimo_lentele.png")
    plt.close(fig)


# --- Pagrindinė dalis ----------------------------------------------------------------
def main():
    p = ikelti()
    modeliai = list(dict.fromkeys(p["modelis"]))
    scenarijai = [s for s in SCENARIJAI if s in set(p["scenarijus"])]

    m = pd.DataFrame([dict(modelis=mod, scenarijus=sc, **metrikos(d))
                      for (mod, sc), d in p.groupby(["modelis", "scenarijus"], sort=False)])
    m = m.set_index(["modelis", "scenarijus"]).loc[[(a, b) for a in modeliai for b in scenarijai]]

    sk_eil = []
    for mod in modeliai:
        f1 = {sc: m.loc[(mod, sc), "F1"] for sc in scenarijai}
        sk_eil.append(dict(modelis=mod, **{f"F1_{sc}": v for sc, v in f1.items()},
                           dF1_kalba=f1.get("S2", np.nan) - f1.get("S1", np.nan),
                           dF1_vertimas=f1.get("S3", np.nan) - f1.get("S2", np.nan)))
    skirt = pd.DataFrame(sk_eil)

    REZULTATAI.mkdir(exist_ok=True)
    rasyti_csv(m.reset_index().round(4), REZULTATAI / "metrikos.csv")
    rasyti_csv(skirt.round(4), REZULTATAI / "skirtumai.csv")

    # Klaidos klaidų analizei (su tekstais)
    rinkinys = skaityti_csv(RINKINYS)[["id"] + list(SCENARIJAI.values())]
    klaidos = p[p["pred"] != p["klase"]].merge(rinkinys, on="id", how="left")
    klaidos["tekstas"] = [r[SCENARIJAI[r["scenarijus"]]] for _, r in klaidos.iterrows()]
    rasyti_csv(klaidos[["modelis", "scenarijus", "id", "poaibis", "klase", "pred",
                        "atsakymas", "tekstas"]], REZULTATAI / "klaidos.csv")

    patikra = patikros_suvestine()
    if patikra is not None:
        rasyti_csv(patikra, REZULTATAI / "patikros_suvestine.csv")

    # --- Spausdinimas ------------------------------------------------------------
    pd.set_option("display.width", 200)
    rodyti = m[["n", "tikslumas", "atkurimas", "F1", "FPR", "atkurimas_K", "atkurimas_J",
                "FPR_N", "FPR_R", "laikas_ms", "neatpazinta"]].round(3)
    print("\nMETRIKOS\n" + rodyti.to_string())
    print("\nKALBOS POVEIKIS IR VERTIMO NAUDA\n" + skirt.round(3).to_string(index=False))
    print("   ΔF1 kalba < 0 - lietuviškai prasčiau;  ΔF1 vertimas > 0 - vertimas padeda")
    if patikra is not None:
        print("\nVERTIMO PATIKRA\n" + patikra.to_string(index=False))

    # --- LaTeX lentelės ------------------------------------------------------------
    l1 = m.reset_index()
    l1["modelis_sc"] = [f"{a} & {b}" for a, b in zip(l1["modelis"], l1["scenarijus"])]
    for s in ["tikslumas", "atkurimas", "F1", "FPR"]:
        l1[s] = l1[s].map(sk)
    l1["laikas"] = l1["laikas_ms"].map(lambda v: sk(v, 1))
    l2 = m.reset_index()
    l2["modelis_sc"] = [f"{a} & {b}" for a, b in zip(l2["modelis"], l2["scenarijus"])]
    for s in ["atkurimas_K", "atkurimas_J", "FPR_N", "FPR_R"]:
        l2[s] = l2[s].map(sk)
    l3 = skirt.copy()
    for s in l3.columns[1:]:
        l3[s] = l3[s].map(sk)

    tex = [
        latex_lentele(l1, ["modelis_sc", "tikslumas", "atkurimas", "F1", "FPR", "laikas"],
                      ["Klasifikatorius", "Scen.", "Tikslumas", "Atkūrimas", "F1", "FPR", "ms"],
                      "Klasifikatorių metrikos pagal scenarijus", "lent:metrikos",
                      [r">{\raggedright\arraybackslash}X", "C{1.2cm}", "C{2.1cm}", "C{2.1cm}",
                       "C{1.4cm}", "C{1.4cm}", "C{1.3cm}"]),
        latex_lentele(l2, ["modelis_sc", "atkurimas_K", "atkurimas_J", "FPR_N", "FPR_R"],
                      ["Klasifikatorius", "Scen.", "Atkūrimas K", "Atkūrimas J", "FPR N", "FPR R"],
                      "Metrikos pagal poaibius", "lent:poaibiai",
                      [r">{\raggedright\arraybackslash}X"] + ["C{1.2cm}"] + ["C{2.2cm}"] * 4),
        latex_lentele(l3, list(l3.columns),
                      ["Klasifikatorius"] + [f"F1 {sc}" for sc in scenarijai]
                      + [r"$\Delta F_1^{\mathrm{kalba}}$", r"$\Delta F_1^{\mathrm{vertimas}}$"],
                      "Kalbos poveikis ir vertimo būdo nauda", "lent:skirtumai",
                      [r">{\raggedright\arraybackslash}X"] + ["C{1.4cm}"] * len(scenarijai)
                      + ["C{2.3cm}"] * 2),
    ]
    (REZULTATAI / "lenteles.tex").write_text("\n".join(tex), encoding="utf-8")

    grafikas_f1(m, modeliai, scenarijai)
    grafikas_matricos(m, modeliai, scenarijai)
    print(f"\nFailai įrašyti į {REZULTATAI}")


if __name__ == "__main__":
    main()
