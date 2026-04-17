# 附录A：常见问题与补充信息

**[English](faq.md)** | **[中文](faq.zh.md)**

---

参考资料和补充信息。

---

## 完整参数参考

为避免参数定义重复和漂移，参数权威参考统一维护在：

- `advanced/hyperparameters.md`（推荐）
- `api/run-pipeline.md`（面向 CLI 的参考）

---

## 目录结构

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

## 硬件要求

| 配置 | CPU | 内存 | GPU | CUDA | 存储 |
|-------|-----|-----|-----|------|---------|
| 最低 | 4核 | 8GB | 4GB显存 | 11.8+ | 20GB |
| 推荐 | 8核 | 16GB | 12GB显存 | 12.1+ | 50GB SSD |
| 高性能 | 16+核 | 32GB+ | A100 40GB | 12.1+ | 200GB NVMe |

---

## 常见问题

**问：THETA有什么不同？**  
答：THETA使用通义千问嵌入和神经变分推理，相比LDA或ETM具有更好的语义理解能力。

**问：应该使用哪个模型规模？**  
答：原型开发用0.6B，生产环境用4B，追求最高质量用8B。

**问：最小数据集规模？**  
答：建议至少500篇文档，平均每篇50词以上。

**问：训练时间？**  
答：5K文档在V100上：0.6B约25分钟，4B约50分钟。

**问：需要GPU吗？**  
答：是的。预处理和训练都需要GPU。

---

## 引用

```bibtex
@article{theta2024,
  title={THETA：基于通义千问嵌入的先进主题建模},
  author={CodeSoul团队},
  year={2024},
  url={https://github.com/CodeSoul-co/THETA}
}
```

---

## 联系方式

- 网站：[https://theta.code-soul.com](https://theta.code-soul.com)
- GitHub：[https://github.com/CodeSoul-co/THETA](https://github.com/CodeSoul-co/THETA)
- 邮箱：support@theta.code-soul.com

---

**文档版本**：1.0.0  
**最后更新**：2026年2月6日