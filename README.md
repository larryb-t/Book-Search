# Context-Aware Book Search and Recommendation System

CS 125 – Next Generation Search Systems

A prototype search and recommendation system that helps users discover books
based on keyword search, personal preferences, and contextual factors such as
available reading time.

## Project Structure
- data/: raw and processed datasets
- src/: ingestion, indexing, ranking, recommendation logic
- app/: user interface
- notebooks/: experiments and analysis
- docs/: system design and demo notes
- tests/: testing and validation..

## Ingestion Module
The ingestion layer in `src/ingest` builds a logical view of each book, separating
searchable tokens from display and filter fields. It currently supports both
Project Gutenberg catalogs (`pg_catalog.py`) and the Kaggle 7k books dataset
(`kaggle_books.py`), with shared text normalization and tokenization rules.

## Team
- Romina Pouya
- Sana Kimiagar
- Larry Bui Tran
