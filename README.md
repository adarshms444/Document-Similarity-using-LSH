# Document-Similarity-using-LSH

In today’s digital world, organizations often deal with large collections of text documents. But checking every pair of documents word by word is slow and expensive, especially when the dataset grows. 
This project demonstrates **document similarity analysis** using **PySpark** with **MinHash LSH (Locality Sensitive Hashing)**. It processes a set of text documents, extracts shingles, computes similarity using MinHash, and visualizes results with graphs & heatmaps.This approach lets me quickly find approximate similarities between documents. In this task, I analyzed 9 documents and tried to see how they relate to each other, and they form clusters(buckets) of similarity.


---

## Features
- Load multiple text documents into Spark
- Clean and tokenize text
- Generate **n-gram shingles** (3-grams)
- Extract features with **HashingTF**
- Apply **MinHashLSH** for approximate similarity
- Compute nearest neighbors & similarity matrix
- Visualize results with:
  - Network graph (nearest neighbors)
  - Heatmap (document similarity matrix)

---

## 📦 Installation
Make sure you have **Google Colab** or Python environment with PySpark.  

```bash
!pip install pyspark reportlab
