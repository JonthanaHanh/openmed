# LlamaIndex Redaction Postprocessor

OpenMed can redact retrieved LlamaIndex nodes locally before response synthesis.
The integration is optional: importing `openmed` or `openmed.interop` does not
import LlamaIndex, and the adapter loads `llama-index-core` only when you create
a postprocessor, ingestion transform, or framework tool.

```bash
pip install "openmed[llamaindex]"
```

## Redact retrieved nodes before synthesis

Pass the postprocessor to a LlamaIndex query engine. It copies each retrieved
node, runs `openmed.core.pii.deidentify()` over its text, and preserves the node
score and metadata. The original retrieved node is not mutated.

```python
from openmed.interop.llamaindex import create_redaction_postprocessor

redact_nodes = create_redaction_postprocessor()

query_engine = index.as_query_engine(
    node_postprocessors=[redact_nodes],
)
response = query_engine.query(
    "What follow-up is documented for this patient?"
)
```

`create_redaction_postprocessor()` uses masking defaults and the deterministic
safety sweep. Tune the de-identification settings with
`LlamaIndexRedactionConfig`.

```python
from openmed.interop.llamaindex import (
    LlamaIndexRedactionConfig,
    create_redaction_postprocessor,
)

redact_nodes = create_redaction_postprocessor(
    config=LlamaIndexRedactionConfig(
        method="mask",
        confidence_threshold=0.6,
        lang="en",
        use_safety_sweep=True,
    )
)
```

This postprocessor protects the context sent to response synthesis. It does not
rewrite text already stored in an index or vector store.

## Redact during ingestion

Use the optional transform when PHI must be removed before nodes are embedded or
stored. Place it after text splitting and before embedding in the ingestion
pipeline.

```python
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter

from openmed.interop.llamaindex import create_redaction_transform

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=32),
        create_redaction_transform(),
        embed_model,
    ]
)
redacted_nodes = pipeline.run(documents=documents)
```

The transform also copies nodes before changing their text. OpenMed performs the
redaction; LlamaIndex remains an optional orchestration consumer and is never a
core dependency.
