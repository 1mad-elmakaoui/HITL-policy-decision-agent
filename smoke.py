"""Step 1 smoke test: prove the contract layer before building on it."""
import subprocess, sys, tempfile
from config.settings import load_settings
from shared.embeddings import build_embedder, fit_embedder
from shared.schema import PolicyChunk, ChunkMetadata, make_chunk_id
from shared.vector_store import build_vector_store

# 1. Schema round-trips, chunk_id is deterministic
m = ChunkMetadata(document_id="HR-1", document_version="1.0", policy_name="Test",
                  section="Section 2. Discipline", source="x.md", chunk_id="c1")
c = PolicyChunk(text="Warnings expire after twelve months.", metadata=m)
assert PolicyChunk.from_dict(c.to_dict()).text == c.text
assert make_chunk_id("HR-1","1.0","Section 2. Discipline",3) == \
       make_chunk_id("HR-1","1.0","Section 2. Discipline",3)

# 2. Embedding is stable ACROSS PROCESSES  <-- the trap
code = ("from shared.embeddings import build_embedder;"
        "print(sum(build_embedder('hashing').embed_query('termination approval')))")
runs = {subprocess.run([sys.executable,"-c",code],capture_output=True,text=True).stdout
        for _ in range(2)}
assert len(runs) == 1, f"embedding differs between processes: {runs}"

# 3. Store round-trips WITH provenance
emb = fit_embedder("hashing", [c.text])
with tempfile.TemporaryDirectory() as d:
    store = build_vector_store("memory", d, "smoke", similarity=emb.similarity)
    store.upsert([c], emb.embed_documents([c.text]))
    hit = store.query(emb.embed_query("how long do warnings last"), k=1)[0]
    assert hit.chunk.metadata.document_version == "1.0"
    assert not hit.chunk.metadata.missing_fields()

# 4. Both layers read the SAME config
s = load_settings()
assert s.retrieval.weak_evidence_score < s.retrieval.strong_evidence_score
print("Step 1 contract layer is sound.")