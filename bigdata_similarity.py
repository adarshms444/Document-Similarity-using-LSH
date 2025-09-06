# -*- coding: utf-8 -*-

# ==============================
# 0) Set up Spark in Colab
# ==============================
!pip -q install pyspark reportlab

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, split, size, explode, array_distinct, lit
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, NGram, HashingTF, MinHashLSH
from pyspark.sql.types import StringType, ArrayType
import itertools
import math
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Start Spark
spark = SparkSession.builder.appName("MinHashLSH_Report").getOrCreate()
print("Spark version:", spark.version)

# ======================================
# 1) Point to your 9 docs (edit if needed)
# ======================================
DOC_PATHS = [
    "/content/Doc1.txt",
    "/content/Doc2.txt",
    "/content/Doc3.txt",
    "/content/Doc4.txt",
    "/content/Doc5.txt",
    "/content/Doc6.txt",
    "/content/Doc7.txt",
    "/content/Doc8.txt",
    "/content/Doc9.txt",
]

# Quick validation: show which files exist/missing
existing = [p for p in DOC_PATHS if os.path.exists(p)]
missing = [p for p in DOC_PATHS if not os.path.exists(p)]
print("Found", len(existing), "files.")
if missing:
    print("Missing files:", missing)

# ===============================
# 2) Load docs into Spark DataFrame
# ===============================
def load_docs(paths):
    rows = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
        except Exception as e:
            txt = ""
            print(f"Warning: could not read {p}: {e}")
        doc_id = os.path.splitext(os.path.basename(p))[0]
        rows.append((doc_id, txt))
    return rows

rows = load_docs(existing)
df = spark.createDataFrame(rows, ["doc_id", "raw_text"])
print("Documents loaded:", df.count())
df.show(3, truncate=80)

# =======================================
# 3) Clean → tokenize → stopwords → shingles
# =======================================
df_clean = (
    df.withColumn("text", lower(col("raw_text")))
      .withColumn("text", regexp_replace(col("text"), r"[^a-z\s]", " "))
      .withColumn("text", regexp_replace(col("text"), r"\s+", " "))
)

tokenizer = RegexTokenizer(
    inputCol="text", outputCol="tokens",
    pattern=r"\W+", toLowercase=True, minTokenLength=2
)
df_tok = tokenizer.transform(df_clean)

remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
df_sw = remover.transform(df_tok)

N_SHINGLE = 3
ngram = NGram(n=N_SHINGLE, inputCol="filtered", outputCol="shingles")
df_shingles = ngram.transform(df_sw).select("doc_id", "shingles")

df_shingles = df_shingles.withColumn("shingles", array_distinct(col("shingles")))
df_shingles.show(3, truncate=100)

# =========================================================
# 4) Feature extraction for MinHashLSH
# =========================================================
hash_dim = 1 << 14
htf = HashingTF(
    inputCol="shingles", outputCol="features",
    numFeatures=hash_dim, binary=True
)
df_feat = htf.transform(df_shingles).cache()

# ===================================================
# 5) MinHash LSH model
# ===================================================
NUM_TABLES = 14
mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=NUM_TABLES)
lsh_model = mh.fit(df_feat)
df_hashed = lsh_model.transform(df_feat)

# ======================================
# 6) Similarity search (self-join)
# ======================================
DIST_THRESHOLD = 0.8
pairs = (
    lsh_model
    .approxSimilarityJoin(df_hashed, df_hashed, DIST_THRESHOLD, distCol="JaccardDistance")
    .select(
        col("datasetA.doc_id").alias("doc1"),
        col("datasetB.doc_id").alias("doc2"),
        col("JaccardDistance")
    )
    .where(col("doc1") < col("doc2"))
    .orderBy(col("JaccardDistance").asc())
)
print("≈ Similar pairs (LSH, threshold =", DIST_THRESHOLD, ")")
pairs.show(truncate=False)

TOP_K = 3
print("\nTop", TOP_K, "nearest neighbors per doc (approx):")
for r in df_hashed.select("doc_id").collect():
    doc = r["doc_id"]
    nn = lsh_model.approxNearestNeighbors(
        dataset=df_hashed,
        key=df_hashed.where(col("doc_id")==doc).select("features").head().features,
        numNearestNeighbors=TOP_K+1
    )
    nn = nn.where(col("doc_id") != lit(doc)).select("doc_id", "distCol").orderBy(col("distCol").asc())
    print(f"\n{doc}:")
    nn.show(truncate=False)

# Collect approximate nearest neighbors for visualization
nn_list = []
for r in df_hashed.select("doc_id").collect():
    doc = r["doc_id"]
    nn = lsh_model.approxNearestNeighbors(
        dataset=df_hashed,
        key=df_hashed.where(col("doc_id")==doc).select("features").head().features,
        numNearestNeighbors=TOP_K+1
    )
    nn = nn.where(col("doc_id") != lit(doc)).select("doc_id", "distCol").orderBy(col("distCol").asc()).limit(TOP_K)
    for row in nn.collect():
        nn_list.append((doc, row["doc_id"], row["distCol"]))


# Visualize approximate nearest neighbors
import networkx as nx
G = nx.DiGraph()
for doc in df_hashed.select("doc_id").collect():
    G.add_node(doc["doc_id"])

for doc1, doc2, dist in nn_list:
    G.add_edge(doc1, doc2, weight=dist)

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=12, width=2, edge_color="gray", alpha=0.6)
edge_labels = nx.get_edge_attributes(G, "weight")
edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
plt.title("Approximate Nearest Neighbors (LSH)")
plt.show()

# ========================================================
# 7) Build full similarity matrix (for heatmap)
# ========================================================
from pyspark.sql import functions as F

# Collect all document IDs
doc_ids = [r["doc_id"] for r in df_hashed.select("doc_id").collect()]

# Initialize similarity matrix with zeros
sim_matrix = pd.DataFrame(0.0, index=doc_ids, columns=doc_ids)

# Compute pairwise approximate distances
all_pairs = (
    lsh_model.approxSimilarityJoin(df_hashed, df_hashed, 1.0, distCol="JaccardDistance")
    .select(
        F.col("datasetA.doc_id").alias("doc1"),
        F.col("datasetB.doc_id").alias("doc2"),
        "JaccardDistance"
    )
    .where(F.col("doc1") < F.col("doc2"))
)

pairs_pd = all_pairs.toPandas()

# Fill similarity matrix (sim = 1 - distance)
for _, row in pairs_pd.iterrows():
    d1, d2, dist = row["doc1"], row["doc2"], row["JaccardDistance"]
    sim = 1 - dist
    sim_matrix.loc[d1, d2] = sim
    sim_matrix.loc[d2, d1] = sim

# Diagonal = 1 (self similarity)
for d in doc_ids:
    sim_matrix.loc[d, d] = 1.0

# ========================================================
# 8) Plot heatmap
# ========================================================
plt.figure(figsize=(8,6))
sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
plt.title("Document Similarity Heatmap (LSH Approximation)")
plt.tight_layout()
plt.show()
