# THETA Data Format Template

## Column Naming Convention

| Purpose | Column Name | Required For |
|---------|-------------|--------------|
| Text | `text` | All models |
| Timestamp | `timestamp` | DTM |
| Covariates | `cov_*` | STM |
| Label | `label` | Supervised |

---

## DTM Template

```csv
text,timestamp
"Document about renewable energy policies...",2020
"Article discussing AI developments...",2021
"Research paper on climate change...",2022
"Policy document on public health...",2023
"Analysis of digital economy trends...",2024
```

---

## STM Template

```csv
text,cov_province,cov_category,cov_source
"Policy document on environmental protection...",Beijing,Policy,Government
"News article about economic development...",Shanghai,News,Media
"Research report on technology innovation...",Guangdong,Report,Institute
"Government announcement on public health...",Beijing,Policy,Government
"Academic paper on urban planning...",Jiangsu,Research,University
```

---

## DTM + STM Combined Template

```csv
text,timestamp,cov_province,cov_category,label
"Document content...",2023,Beijing,Policy,Environment
"Document content...",2024,Shanghai,News,Technology
```
