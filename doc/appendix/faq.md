# Appendix A: FAQ & Supplementary Information

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
A: Yes. GPU required for preprocessing and training.

---

## Citation

```bibtex
@article{theta2024,
  title={THETA: Advanced Topic Modeling with Qwen Embeddings},
  author={CodeSoul Team},
  year={2024},
  url={https://github.com/CodeSoul-co/THETA}
}
```

---

## Contact

- Website: [https://theta.code-soul.com](https://theta.code-soul.com)
- GitHub: [https://github.com/CodeSoul-co/THETA](https://github.com/CodeSoul-co/THETA)
- Email: support@theta.code-soul.com

---

**Document Version**: 1.0.0  
**Last Updated**: February 6, 2026
