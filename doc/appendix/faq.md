# Appendix A: FAQ & Supplementary Information

**English** | [中文](https://github.com/CodeSoul-co/THETA/blob/main/doc/appendix/faq.zh.md)

Reference materials and supplementary information.

---

## Complete Parameter Reference

To avoid duplicated and drifting parameter definitions, the canonical parameter reference is maintained in:

- `advanced/hyperparameters.md` (recommended)
- `api/run-pipeline.md` (CLI-oriented reference)

---

## Directory Structure

```
./
├── ETM/
│   ├── main.py
│   ├── run_pipeline.py
│   ├── prepare_data.py
│   └── src/
├── data/
│   └── {dataset}/
│       └── {dataset}_cleaned.csv
├── result/
│   ├── 0.6B/
│   ├── 4B/
│   ├── 8B/
│   └── baseline/
└── embedding_models/
```

---

## Hardware Requirements

| Setup | CPU | RAM | GPU | CUDA | Storage |
|-------|-----|-----|-----|------|---------|
| Minimum | 4 cores | 8GB | 4GB VRAM | 11.8+ | 20GB |
| Recommended | 8 cores | 16GB | 12GB VRAM | 12.1+ | 50GB SSD |
| High-Performance | 16+ cores | 32GB+ | A100 40GB | 12.1+ | 200GB NVMe |

---

## FAQ

**Q: What makes THETA different?**  
A: THETA uses Qwen embeddings and neural variational inference for better semantic understanding than LDA or ETM.

**Q: Which model size to use?**  
A: 0.6B for prototyping, 4B for production, 8B for maximum quality.

**Q: Minimum dataset size?**  
A: 500+ documents with 50+ words average recommended.

**Q: Training time?**  
A: 5K docs with 0.6B on V100: ~25 min. 4B: ~50 min.

**Q: GPU required?**  
A: Traditional models can run on CPU. Local neural embeddings can be slow on CPU; use compatible GPU hardware when needed. Desktop installers include CPU dependencies.

---

## Citation

```bibtex
@misc{codesoul2026theta,
  author = {{CodeSoul-co}},
  title = {THETA: Local Topic Modeling and Research Analysis},
  year = {2026},
  howpublished = {GitHub repository},
  url = {https://github.com/CodeSoul-co/THETA},
  note = {Version 0.3.2}
}
```

---

## Contact

- GitHub: [https://github.com/CodeSoul-co/THETA](https://github.com/CodeSoul-co/THETA)
- Email: support@theta.code-soul.com

---
