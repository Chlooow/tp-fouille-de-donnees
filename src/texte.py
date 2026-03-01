import fitz

def extract_text_from_pdf(chemin_pdf):
    doc = fitz.open(chemin_pdf)
    chapitre27 = ""
    for page in doc:
        texte_page = page.get_text()
        texte_page = " ".join(texte_page.split())
        chapitre27 += texte_page + " \n"
    doc.close()
    return chapitre27.strip()

def bout_de_texte(texte, longueur=1000):
    mots = texte.split()
    morceaux = []
    for i in range(0, len(mots), longueur):
        morceau = " ".join(mots[i:i+longueur])
        morceaux.append(morceau)
    return morceaux