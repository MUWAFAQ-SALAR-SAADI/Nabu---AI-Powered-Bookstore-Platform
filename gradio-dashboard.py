import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader  # ✅ استبدال langchain_community
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

import gradio as gr

# ------------------------------
# مجلد المشروع
project_dir = os.path.dirname(os.path.abspath(__file__))

# تحميل متغيرات البيئة
load_dotenv()

# ------------------------------
# مسارات الملفات
books_file = os.path.join(project_dir, "books_with_emotions.csv")
tagged_file = os.path.join(project_dir, "tagged_description.txt")

# التحقق من وجود الملفات
if not os.path.exists(books_file):
    raise FileNotFoundError(f"❌ File not found: {books_file}")

if not os.path.exists(tagged_file):
    raise FileNotFoundError(f"❌ File not found: {tagged_file}")

# ------------------------------
# تحميل بيانات الكتب
books = pd.read_csv(books_file)
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# ------------------------------
# تحميل النصوص مع ترميز utf-8
raw_documents = TextLoader(tagged_file, encoding="utf-8").load()

# تقسيم النصوص
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(raw_documents)

# ------------------------------
# استخدام Hugging Face Embeddings بدل OpenAI
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, hf_embeddings)

# ------------------------------
# دوال التوصية
def retrieve_semantic_recommendations(query: str, category: str = None, tone: str = None,
                                      initial_top_k: int = 50, final_top_k: int = 16) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_description = " ".join(description.split()[:30]) + "..."
        authors_split = row["authors"].split(";")

        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

# ------------------------------
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# ------------------------------
# واجهة Gradio
with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# 📚 Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label="Enter a description of a book:",
            placeholder="e.g., A story about forgiveness"
        )
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("🔍 Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended Books", columns=4, rows=4)

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

# ------------------------------
if __name__ == "__main__":
    dashboard.launch()