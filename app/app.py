from flask import Flask, render_template, request
from src.index import InvertedIndex
from src.ingest import find_pg_catalog, load_pg_catalog
from src.ingest.kaggle_books import find_books_csv, load_books_csv
from src.rank import search
from flask import session


dataset_root = '.'

pg_path = find_pg_catalog(dataset_root)
kaggle_path = find_books_csv(dataset_root)

records = []
if pg_path:
    records = load_pg_catalog(pg_path, filter_english=True)
    dataset_name = 'pg_catalog'
elif kaggle_path:
    records = load_books_csv(kaggle_path)
    dataset_name = 'kaggle_books'
else:
    dataset_name = None

index = InvertedIndex.build(records)

app = Flask(__name__)

app.secret_key = "secret"

@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    curr_query = ""

    if request.method == "POST":
        curr_query = request.form.get("query", "")

        genres_input = request.form.get("genres", "")

        if genres_input:
            session["genres"] = genres_input

        preferred_genres = session.get("genres", "")
        preferred_genres_list = [
            g.strip().lower()
            for g in preferred_genres.split(",")
            if g
        ]

        authors_input = request.form.get("authors", "")

        if authors_input:
            session["authors"] = authors_input

        favorite_authors = session.get("authors", "")
        preferred_authors_list = [
            a.strip().lower()
            for a in favorite_authors.split(",")
            if a
        ]

        raw_results = search(
            index,
            curr_query,
            top_k=10,
            preferred_genres=preferred_genres_list,
            preferred_authors=preferred_authors_list, 
        )

        results = raw_results
        # print(results)

    return render_template(
        "index.html",
        results=results,
        current_genres=session.get("genres", ""),
        current_authors=session.get("authors", ""),
        current_query=curr_query
    )


app.run(debug=True)