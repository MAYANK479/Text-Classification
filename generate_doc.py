from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

doc_name = "NLP_Mayank_Final_repro_documentation.pdf"
text = """NLP Naive Bayes Sentiment Analysis - Documentation\n\n1. Overview\nThis report reproduces Mayank's project from PDF using NB/SVM/LR/NBSVM on IMDB and Sentiment140 datasets.\n\n2. Datasets\n- IMDB Movie Reviews: http://ai.stanford.edu/~amaas/data/sentiment/\n- Sentiment140: http://help.sentiment140.com/for-students\n\n3. Preprocessing\n- Lowercase\n- Remove URLs/mentions/hashtags/non-alphanumerics\n- Tokenize regex ([a-z0-9']+)\n- Stopword removal (sklearn EN minus negations)\n- Porter stemming\n\n4. Feature Extraction\n- TF-IDF (max_features=10000, ngram=(1,2), sublinear_tf=True, min_df=2)\n\n5. Models\n- MultinomialNB(alpha=1.0)\n- LinearSVC(C=1.0)\n- LogisticRegression(C=1.0)\n- NBSVM hybrid with NB log-count ratio\n\n6. Evaluation\n- 80/20 stratified split\n- 5-fold CV\n- Accuracy, F1, confusion matrix, log loss, and charts\n\n7. Run command\npython3 nlp_nb_sentiment.py\n"""

styles = getSampleStyleSheet()
story = []
for line in text.split("\n"):
    if line.strip() == "":
        story.append(Spacer(1, 8))
    else:
        story.append(Paragraph(line, styles["BodyText"]))
        story.append(Spacer(1, 4))

SimpleDocTemplate(doc_name, pagesize=letter).build(story)
print("Created", doc_name)
