# Fine Print Detector

Detects potentially unfair clauses in online Terms of Service, labels each one with its unfair type, and explains it in plain English using expert legal sources.

MAAI902 Natural Language Processing & LLMs, University of Wollongong in Dubai, Autumn 2026

Author: Lorna Tarabeh

---

## The problem

Almost nobody reads Terms of Service before clicking "I agree" (Obar & Oeldorf-Hirsch, 2020), yet these documents often contain clauses that EU consumer law considers potentially unfair, such as limits on the company's liability or the right to close an account at any time. The Fine Print Detector reads a Terms of Service, flags the sentences that may be unfair, names the type, and explains why in simple language, so the user only needs to read the flagged clauses instead of the whole document.

## Pipeline

```
Pasted Terms of Service
   -> split into sentences, drop fragments under 5 words
   -> long sentences (over 128 tokens): LLM extractive summary (original words only)
   -> Legal-BERT: 8 yes/no outputs, one per unfair type
   -> flagged clause + type
   -> RAG: retrieve the closest expert explanations (MiniLM + cross-encoder reranker)
   -> Qwen2.5-7B-Instruct writes "Why this may be unfair" and "What you can do",
      using only the retrieved explanations (fixed backup sentence if none fits)
```

## Data

- **UNFAIR-ToS** from LexGLUE (`coastalcph/lex_glue`, subset `unfair_tos`): 9,414 sentences from 50 Terms of Service, labeled by legal experts with 8 unfair types (Lippi et al., 2019; Chalkidis et al., 2022). About 89% of sentences are fair.
- **Expert explanations** from Ruggeri et al. (2022) (github.com/federicoruggeri/Memnet_ToS, MIT license): 84 explanations for 5 unfair types, plus their clause-to-explanation links, used to test the retrieval. 3 explanations were written for this project for the remaining types, based on the definitions in Lippi et al. (2019), Section 3.1.

## Notebooks

| Notebook | What it does |
|---|---|
| `01_explore_data.ipynb` | Class balance, type counts, multi-label sentences |
| `02_preprocessing.ipynb` | Quality checks, sentence lengths, label checklists, tokenization, and the word-swap augmentation experiment (tested and rejected) |
| `02b_long_sentences.ipynb` | LLM extractive summaries for sentences over 128 tokens, with exact-copy checks and a hand check |
| `03_baseline.ipynb` | TF-IDF + SVM baseline, plus "always fair" and random reference models |
| `04_legal_bert.ipynb` | Legal-BERT training, comparison runs (class weights, summaries, BERT), 3 seeds, full evaluation |
| `05_rag_explanations.ipynb` | Knowledge base, retrieval tests, reranker, LLM explanations, hand check |
| `06_demo.ipynb` | Demo app (in progress) |

## Key results

### Detection (UNFAIR-ToS test set, average of 3 seeds)

| Model | Micro F1 | Macro F1 |
|---|---|---|
| Always fair (reference) | 88.9 | 10.5 |
| TF-IDF + SVM (baseline) | 95.3 | 76.5 |
| Regular BERT | 95.0 ± 0.2 | 79.1 ± 1.4 |
| Legal-BERT, no summaries | 95.2 ± 0.7 | 81.5 ± 2.8 |
| Legal-BERT + class weights | 95.8 ± 0.4 | 81.4 ± 2.3 |
| **Legal-BERT + summaries (final)** | **95.8 ± 0.2** | **82.8 ± 1.3** |
| LexGLUE Legal-BERT (best of 5 seeds) | 96.0 | 83.0 |

- Legal-BERT clearly outperformed regular BERT on the test set.
- Class weighting did not improve results, so it is not used in the final model.
- On the 17 summarized test sentences, both setups were fully correct, so the summaries did not change test scores; they are kept to protect important content in long real-world clauses.
- Scores on the 1,574 test sentences never seen in training were the same as on the full test set, so the 33 overlapping sentences did not inflate results.

### Explanations (RAG)

- Retrieval, tested on 1,712 clauses linked to explanations by experts: a correct explanation is in the top 3 for **67.8%** of clauses with MiniLM + reranker (random guessing: 31.0%).
- Hand check on one correctly flagged clause per type: relevant sources 8/8, clear 8/8, realistic tips 7/8, stayed within the sources 6/8.

## How to run

All notebooks run in Google Colab (T4 GPU for notebooks 02b, 04, 05 and the demo). Run them in order. Intermediate results are saved to a Google Drive folder `MyDrive/fine_print_detector/`:

- `long_sentence_summaries.csv`, `knowledge_base.csv`
- `run_results.csv`, `run_results_seeds.csv`, `test_results.csv`, `predictions/`
- `models/legal_bert_A/` (the trained model, about 440 MB, too large for GitHub)

Install dependencies with `pip install -r requirements.txt`.

## Limitations

- The dataset is small (50 contracts), so scores vary between seeds and between validation and test.
- Fairness is a legal judgment; some very similar clauses carry different expert labels.
- The model can be confidently wrong, and the LLM occasionally adds a point not found in the sources. Results are therefore presented as flagged for review, with the sources shown, never as legal advice.

## References

APA 7th edition.

Chalkidis, I., Fergadiotis, M., Malakasiotis, P., Aletras, N., & Androutsopoulos, I. (2020). LEGAL-BERT: The Muppets straight out of law school. *Findings of the Association for Computational Linguistics: EMNLP 2020*, 2898–2904. https://doi.org/10.18653/v1/2020.findings-emnlp.261

Chalkidis, I., Jana, A., Hartung, D., Bommarito, M., Androutsopoulos, I., Katz, D. M., & Aletras, N. (2022). LexGLUE: A benchmark dataset for legal language understanding in English. *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 4310–4330. https://doi.org/10.18653/v1/2022.acl-long.297

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, 4171–4186. https://doi.org/10.18653/v1/N19-1423

Ghosh, S., Evuru, C. K. R., Kumar, S., Ramaneswaran, S., Sakshi, S., Tyagi, U., & Manocha, D. (2023). DALE: Generative data augmentation for low-resource legal NLP. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, 8511–8565. https://doi.org/10.18653/v1/2023.emnlp-main.528

Juttu, N. P. P., Singireddy, S., Gona, S., & Timilsina, S. (2025). *Text to trust: Evaluating fine-tuning and LoRA trade-offs in language models for unfair terms of service detection* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2510.22531

Lippi, M., Pałka, P., Contissa, G., Lagioia, F., Micklitz, H.-W., Sartor, G., & Torroni, P. (2019). CLAUDETTE: An automated detector of potentially unfair clauses in online terms of service. *Artificial Intelligence and Law, 27*(2), 117–139. https://doi.org/10.1007/s10506-019-09243-2

Obar, J. A., & Oeldorf-Hirsch, A. (2020). The biggest lie on the internet: Ignoring the privacy policies and terms of service policies of social networking services. *Information, Communication & Society, 23*(1), 128–147. https://doi.org/10.1080/1369118X.2018.1486870

Perçin, S., Galassi, A., Lagioia, F., Ruggeri, F., Santin, P., Sartor, G., & Torroni, P. (2022). Combining WordNet and word embeddings in data augmentation for legal texts. *Proceedings of the Natural Legal Language Processing Workshop 2022*, 47–52. https://doi.org/10.18653/v1/2022.nllp-1.4

Ruggeri, F., Lagioia, F., Lippi, M., & Torroni, P. (2022). Detecting and explaining unfairness in consumer contracts through memory networks. *Artificial Intelligence and Law, 30*(1), 59–92. https://doi.org/10.1007/s10506-021-09288-2

Sun, C., Qiu, X., Xu, Y., & Huang, X. (2019). How to fine-tune BERT for text classification? In *Chinese Computational Linguistics: 18th China National Conference, CCL 2019* (Lecture Notes in Computer Science, Vol. 11856, pp. 194–206). Springer. https://doi.org/10.1007/978-3-030-32381-3_16

Xu, S., Broda, I., Haddad, R., Negrini, M., & Grabmair, M. (2022). Attack on unfair ToS clause detection: A case study using universal adversarial triggers. *Proceedings of the Natural Legal Language Processing Workshop 2022*, 238–245. https://doi.org/10.18653/v1/2022.nllp-1.21
