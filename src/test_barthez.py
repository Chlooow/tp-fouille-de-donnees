# Les imports
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from texte import extract_text_from_pdf, bout_de_texte
import nltk
nltk.download('punkt')
import evaluate
from matplotlib import pyplot as plt
import numpy as np
import nltk
import seaborn as sns

nltk.download('punkt_tab')

def main():
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Charger le texte du PDF
    chemin_du_pdf = r"../ressources\Amin-Maalouf-Leon-l-africain-Chap27.pdf"
    print("Extraction du texte du chapitre 27")
    texte_chapitre27 = extract_text_from_pdf(chemin_du_pdf)
    print(f"longeur du texte extrait: {len(texte_chapitre27)} caractères")
    print("Texte extrait avec succès")  
    print("apercu du texte extrait:", texte_chapitre27[:500], "...\n")

    # Charger le modèle 
    model_name = "moussaKam/barthez-orangesum-abstract"
    print("Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Chargement du modèle...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    print("Modèle BARThez chargé avec succès")

    # Affichage des tokens
    phrase_demo = "Elle avait prononcé ces mots en arabe, mais avec cet accent circassien que tous les Cairotes reconnaissent sans peine, puisqu’il est celui des sultans et des officiers mamelouks. Avant que j’aie pu répondre, le marchand était revenu, avec l’offre d’un prix"
    tokens_demo = tokenizer.tokenize(phrase_demo)
    print("\nTEST DE TOKENISATION")
    print(f"Texte : {phrase_demo}")
    print(f"Tokens : {tokens_demo}")
    print("----------------------------\n")

    # On prepare le texte pour le modèle
    print("Préparation du texte pour le modèle...")
    morceaux_texte = bout_de_texte(texte_chapitre27, longueur=1000)
    resumes = []

    # ensuite pour chaque segment on faire la tokenization
    for i, morceau in enumerate(morceaux_texte):
        print(f"Traitement du morceau {i+1}/{len(morceaux_texte)}...")
        inputs = tokenizer(morceau, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"], 
                num_beams=4,
                max_length=150, 
                early_stopping=True
            )
        resume = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        resumes.append(resume)
        print(f"Résumé du morceau {i+1}:\n{resume}\n")
    
    # Concaténer tous les résumés
    resume_final = " ".join(resumes)
    print("Résumé final du chapitre 27:\n", resume_final)
    print("Résumé généré avec succès")

    # Sauvegarder le model du chapitre 27
    print("Sauvegarde du modèle et du tokenizer...")
    model.save_pretrained("./models/barthez-orangesum-abstract")
    tokenizer.save_pretrained("./models/barthez-orangesum-abstract")
    print("Modèle et tokenizer sauvegardés avec succès")
    print("\n")
    print("\n")

 

    def afficher_attention(text, model, tokenizer, device):
        inputs = tokenizer(text, return_tensors="pt").to(device)
        # On demande explicitement les attentions
        outputs = model(**inputs, output_attentions=True)
        
        # On prend la dernière couche d'attention de l'encodeur
        # Forme : [layer][batch][head][seq][seq]
        attentions = outputs.encoder_attentions[-1][0, 0].detach().cpu().numpy()
        
        tokens = tokenizer.tokenize(text)
        
        plt.figure(figsize=(10, 8))
        # On filtre visuellement ou par couleur les valeurs > 0.5
        sns.heatmap(attentions, annot=True, cmap="YlGnBu", 
                    xticklabels=tokens, yticklabels=tokens)
        plt.title("Matrice d'Attention (Dernière couche - Tête 0)")
        plt.xlabel("Mots clés regardés")
        plt.ylabel("Mot en cours de traitement")
        
        os.makedirs("./visualisations", exist_ok=True)
        plt.savefig("./visualisations/matrice_attention.png")
        plt.show()

    # Appelle la fonction avec une phrase courte du chapitre
    phrase_attention = "Les pyramides ne doivent pas être loin d’ici."
    afficher_attention(phrase_attention, model, tokenizer, device)

    # Essai du modèle sur une petit corpus
    print("Test du modèle sur un petit corpus...")
    texte_test = (
    "Dans une vallée cachée derrière les montagnes bleues, vivait une licorne nommée Azélia. "
    "Son pelage brillait comme l’aube et sa corne diffusait une lumière dorée capable d’apaiser les tempêtes. "
    "Un jour, elle découvrit un village plongé dans l’obscurité depuis la disparition mystérieuse du soleil. "
    "Guidée par son courage, Azélia traversa la forêt interdite pour retrouver l’éclat perdu du ciel. "
    "Depuis ce jour, les habitants racontent que chaque arc-en-ciel est la trace de son galop lumineux."
    )
    
    tokens_test = tokenizer(texte_test, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        summary_ids_test = model.generate(
            tokens_test["input_ids"], 
            num_beams=4,
            max_length=150, 
            early_stopping=True
        )
    resume_test = tokenizer.decode(summary_ids_test[0], skip_special_tokens=True)
    print("Résumé du texte de test:\n", resume_test)
    print("\n")

    #sauvegarder le resume dans un fichier texte
    os.makedirs("./resumes", exist_ok=True)
    chemin_du_resume = "./resumes/resume_chapitre27.txt"
    if not os.path.exists(chemin_du_resume):
        with open(chemin_du_resume, "w", encoding="utf-8") as f:
            f.write(resume_final)
        print("Résumé sauvegardé dans './resumes/resume_chapitre27.txt'")
    else:
        print(f"Le fichier '{chemin_du_resume}' existe déjà. Résumé non sauvegardé pour éviter d'écraser le fichier existant.")

    # Evaluation du modèle
    print("Evaluation du modèle...")
    print("\n")
    generated_summaries = [resume_final]  # Résumé généré par le modèle
    reference_summary = "En l'an 920 de l'hégire au Caire, après avoir assisté à une exécution publique traumatisante, " \
    "Léon rencontre la princesse circassienne Nour chez le marchand Akbar. Pour aider la jeune veuve en difficulté financière, " \
    "Léon lui achète une tapisserie au prix fort, marquant le début d'une idylle secrète qui les mène jusqu'aux pyramides de Guizeh. Après une nuit passée dans un village voisin, Nour révèle à Léon son périlleux secret : elle cache son fils Bayazid, " \
    "un héritier du trône ottoman dont l'existence même représente une menace mortelle pour eux deux."

    rouge = evaluate.load("rouge")
    resultats = rouge.compute(predictions=generated_summaries, references=[reference_summary])
    print(f"Résultats de l'évaluation ROUGE sur chapitre 27:\n{resultats}")
    print("\n")
    # evaluation du teste du modèle sur le texte de test
    generated_summaries_test = [resume_test]  # Résumé généré par le modèle pour le texte de test
    reference_summary_test = "Azélia, une licorne courageuse, sauve un village plongé dans l'obscurité en restaurant la lumière du soleil." # Le texte de test lui-même comme référence pour l'évaluation
    resultats_test = rouge.compute(predictions=generated_summaries_test, references=[reference_summary_test])
    print(f"Résultats de l'évaluation ROUGE pour le texte de test:\n{resultats_test}")

    print(" \n")
    print("Evaluation avec Bertscore...")
    print("\n")
    bertscore = evaluate.load("bertscore")
    print("Evaluation du chapitre 27 avec Bertscore...")
    resultats_bertscore = bertscore.compute(predictions=generated_summaries, references=[reference_summary], lang="fr")
    print(f"Résultats de l'évaluation Bertscore sur chapitre 27:\n{resultats_bertscore}")
    print("\n")
    print("Evaluation du texte de test avec Bertscore...")
    resultats_bertscore_test = bertscore.compute(predictions=generated_summaries_test, references=[reference_summary_test], lang="fr")
    print(f"Résultats de l'évaluation Bertscore pour le texte de test:\n{resultats_bertscore_test}")
    print("Evaluation terminée.")

    print(" \n")
    print("Visualisation des résultats...")
    os.makedirs("./visualisations", exist_ok=True)
    rouge_chap27 = [
        resultats["rouge1"],
        resultats["rouge2"],
        resultats["rougeL"]
    ]

    rouge_test = [
        resultats_test["rouge1"],
        resultats_test["rouge2"],
        resultats_test["rougeL"]
    ]

    labels_rouge = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    # Graphique ROUGE
    plt.figure()
    plt.plot(labels_rouge, rouge_chap27)
    plt.plot(labels_rouge, rouge_test)
    plt.title("Comparaison ROUGE - Chapitre 27 vs Texte Test")
    plt.xlabel("Métriques")
    plt.ylabel("Score")
    plt.legend(["Chapitre 27", "Texte Test"])
    plt.savefig("./visualisations/comparaison_rouge.png")
    plt.close()

    bert_chap27 = [
        resultats_bertscore["precision"][0],
        resultats_bertscore["recall"][0],
        resultats_bertscore["f1"][0]
    ]

    bert_test = [
        resultats_bertscore_test["precision"][0],
        resultats_bertscore_test["recall"][0],
        resultats_bertscore_test["f1"][0]
    ]

    labels_bert = ["Precision", "Recall", "F1"]

    # Graphique BERTScore
    plt.figure()
    plt.plot(labels_bert, bert_chap27)
    plt.plot(labels_bert, bert_test)
    plt.title("Comparaison BERTScore - Chapitre 27 vs Texte Test")
    plt.xlabel("Métriques")
    plt.ylabel("Score")
    plt.legend(["Chapitre 27", "Texte Test"])
    plt.savefig("./visualisations/comparaison_bertscore.png")
    plt.close()

    x = np.arange(len(labels_rouge))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, rouge_chap27, width, label="Chapitre 27")
    plt.bar(x + width/2, rouge_test, width, label="Texte Test")

    plt.xticks(x, labels_rouge)
    plt.ylabel("Score")
    plt.title("Comparaison ROUGE")
    plt.legend()

    plt.savefig("./visualisations/comparaison_barre_rouge.png")
    plt.close()

    x = np.arange(len(labels_bert))

    plt.figure()
    plt.bar(x - width/2, bert_chap27, width, label="Chapitre 27")
    plt.bar(x + width/2, bert_test, width, label="Texte Test")

    plt.xticks(x, labels_bert)
    plt.ylabel("Score")
    plt.title("Comparaison BERTScore")
    plt.legend()

    plt.savefig("./visualisations/comparaison_barre_bertscore.png")
    plt.close()

    print("Graphiques sauvegardés dans le dossier './visualisations'")

    # ------------------------------------------------------------------

    # Autre façon d'améliorer 

    phrases = nltk.sent_tokenize(texte_chapitre27, language='french')
    bout = []
    bout_courant = ""

    for phrase in phrases:
        if len(bout_courant) + len(phrase) > 3000:
            bout.append(bout_courant)
            bout_courant = phrase
        else:
            bout_courant += " " + phrase
    if bout_courant:
        bout.append(bout_courant)

    print(f"Le texte a été découpé en {len(bout)} morceaux logiques.")

    sous_resumes = []
    for i, morceau in enumerate(bout):
        print(f"Génération du sous-résumé {i+1}/{len(bout)}...")
        inputs = tokenizer(morceau, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                num_beams=4,
                no_repeat_ngram_size=3,
                max_length=150, 
                early_stopping=True
            )
        sous_resumes.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))

    print("\nFusion des morceaux pour le résumé final boosté...")
    texte_combine = " ".join(sous_resumes)
    inputs_final = tokenizer(texte_combine, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    with torch.no_grad():
        summary_ids_final = model.generate(
            inputs_final["input_ids"],
            num_beams=6,
            do_sample=True, 
            temperature=0.8,
            max_length=300, 
            early_stopping=True
        )
    
    resume_final_boost = tokenizer.decode(summary_ids_final[0], skip_special_tokens=True)
    print("\n--- RÉSUMÉ BOOSTÉ FINAL ---\n", resume_final_boost)

    # Sauvegarde
    with open("./resumes/resume_chapitre27_boost.txt", "w", encoding="utf-8") as f:
        f.write(resume_final_boost)

    print("\nÉvaluation du résumé boosté...")
    
    # ROUGE
    resultats_rouge_boost = rouge.compute(
        predictions=[resume_final_boost], 
        references=[reference_summary]
    )
    
    # BERTScore
    resultats_bert_boost = bertscore.compute(
        predictions=[resume_final_boost], 
        references=[reference_summary], 
        lang="fr"
    )

    print(f"\n[SCORES BOOSTÉS] ROUGE: {resultats_rouge_boost}")
    print(f"[SCORES BOOSTÉS] BERTScore F1: {resultats_bert_boost['f1'][0]}")

    # Petit comparatif rapide
    print(f"\nComparaison F1 BERTScore :")
    print(f"Standard : {resultats_bertscore['f1'][0]:.4f}")
    print(f"Boosté   : {resultats_bert_boost['f1'][0]:.4f}")

    # visualisation resumé boosté
    print("\nGénération des graphiques comparatifs (Standard vs Boosté)...")
    
    # Préparation des données ROUGE
    labels_rouge = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    scores_standard = [resultats["rouge1"], resultats["rouge2"], resultats["rougeL"]]
    scores_boost = [resultats_rouge_boost["rouge1"], resultats_rouge_boost["rouge2"], resultats_rouge_boost["rougeL"]]

    x = np.arange(len(labels_rouge))
    width = 0.35

    # Graphique ROUGE : Standard vs Boosté
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, scores_standard, width, label='Standard', color='skyblue')
    plt.bar(x + width/2, scores_boost, width, label='Boosté (Map-Reduce)', color='salmon')
    plt.xticks(x, labels_rouge)
    plt.ylabel('Score')
    plt.title('Comparaison ROUGE : Méthode Standard vs Boostée')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("./visualisations/comparaison_standard_vs_boost_rouge.png")
    plt.close()

    # Graphique BERTScore : Standard vs Boosté
    labels_bert = ["F1 Score"]
    bert_std = [resultats_bertscore['f1'][0]]
    bert_boost = [resultats_bert_boost['f1'][0]]

    x_bert = np.arange(len(labels_bert))

    plt.figure(figsize=(8, 6))
    plt.bar(x_bert - width/2, bert_std, width, label='Standard', color='skyblue')
    plt.bar(x_bert + width/2, bert_boost, width, label='Boosté (Map-Reduce)', color='salmon')
    plt.xticks(x_bert, labels_bert)
    plt.ylim(0.5, 0.7) # On zoom pour voir la différence
    plt.ylabel('Score BERTScore')
    plt.title('Comparaison BERTScore : Standard vs Boosté')
    plt.legend()
    plt.savefig("./visualisations/comparaison_standard_vs_boost_bertscore.png")
    plt.close()

    print("Nouveaux graphiques sauvegardés dans './visualisations'")

if __name__ == "__main__":
    main()