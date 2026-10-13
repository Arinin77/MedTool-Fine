# Three illustrative cases

These **three newly authored fictional cases** illustrate selection among similar renal calculators. They contain no patient records or MIMIC-derived content and are not samples drawn from either paper benchmark. The expected paths are authored references, not recorded model runs.

Read the cases below or inspect [samples.json](samples.json) for the structured version. The same four candidate cards are used in all three cases:

| ID | Calculator | Quantity and population |
|---|---|---|
| `cockcroft_gault` | Cockcroft–Gault | Estimated creatinine clearance in adults, mL/min [1] |
| `ckd_epi_2021` | 2021 CKD-EPI creatinine | Adult eGFR indexed to 1.73 m², mL/min/1.73 m² [2] |
| `bedside_schwartz` | 2009 CKiD bedside (Schwartz) | Pediatric eGFR indexed to 1.73 m²; developed mainly in children with CKD [3] |
| `timed_urine_clearance` | Timed urine creatinine clearance | Clearance from urine concentration, urine flow, and concurrent plasma creatinine, mL/min [4, 5] |

The four-card set is reduced for reading. The paper protocol still uses **ten fixed initial candidates**. These cards are concise, newly written descriptions, not the paper's executable calculator assets. Selection here is relative to this small set; the examples do not recommend a preferred equation for current patient care.

## 1. Select by the requested quantity — CALL

**Initial input:** A fictional 60-year-old man weighs 72 kg. Kidney function is stable, body habitus is not extreme, and serum creatinine is 1.0 mg/dL. Estimate creatinine clearance in mL/min using the available clinical variables; a urine collection is not available.

**Reference:** Choose `cockcroft_gault`. Its output matches estimated creatinine clearance. Adult indexed eGFR describes a different quantity, the pediatric equation does not match the population, and collection-based clearance requires urine measurements.

`(140 − 60) × 72 / (72 × 1.0) = 80 mL/min` [1]

## 2. Select by population, then clarify — ASK → CALL

**Initial input:** A fictional 12-year-old child with stable CKD has IDMS-standardized serum creatinine of 1.0 mg/dL. Obtain a quick height-and-creatinine estimate of pediatric eGFR, indexed to 1.73 m². Height is absent from the initial record.

**Controlled clarification:** ASK for measured height → “The child's measured height is 150 cm.”

**Reference:** Select `bedside_schwartz`, obtain height, then CALL. The adult equations do not match this child's population; timed urine clearance does not answer the requested indexed eGFR task.

`0.413 × 150 / 1.0 = 61.95 mL/min/1.73 m²` (equivalently `41.3 × 1.50 / 1.0`) [3]

## 3. Preserve the measurement goal — ASK → ABSTAIN

**Initial input:** A fictional 60-year-old man weighs 72 kg and has stable kidney function. A complete urine collection has volume 1200 mL and urine creatinine 100 mg/dL; plasma creatinine measured during the collection is 1.0 mg/dL. Calculate creatinine clearance from this actual collection, without body-surface-area indexing. Its elapsed collection time is not included.

**Controlled clarification:** ASK for the actual collection duration → “The collection start and end times were not recorded, and the actual duration cannot be recovered.”

**Reference:** ABSTAIN because the required duration is explicitly unavailable. The calculation needs urine flow: `volume_mL / duration_min`. Do not assume a 24-hour duration. Switching to an adult estimating equation would answer a different question, even though its inputs are available. No numeric result is assigned. [4, 5]

## Reading the JSON

- `model_visible` contains only the initial case and candidate IDs. The matching entries in `tool_catalog` are also public inputs.
- `answer_environment` contains controlled replies, to be revealed only after the corresponding ASK.
- `reference` contains the authored action path, parameters, arithmetic, and rationale for readers/evaluators.

Do not send the entire JSON file to a model under evaluation: that would reveal both future answers and the reference. Public downloadability does not change these information boundaries. This is an illustrative interchange format, **not a direct input to the packaged runner or Scorer**. The separate [offline BMI demo](../scripts/run_offline_demo.py) remains the executable runtime example. No model API is needed to read these cases.

## 中文说明

这里只提供 **3 条新编虚构样例**，用于展示相似计算器的适用性选择：① 成人肌酐清除率，直接 CALL；② 儿童 eGFR，询问身高后 CALL；③ 按实际收集计算尿肌酐清除率，确认时长无法获取后 ABSTAIN。它们不取自 MIMIC，也不是论文原始数据的抽样或模型实测结果。

初始病例、受控回答、参考答案分别存放。四张候选卡是便于阅读的缩减展示；正式实验的十候选协议不变。本文件与 JSON 可用于了解任务结构，不能据此复现论文表格。

## Formula and scope references

1. [Cockcroft & Gault (1976), Prediction of creatinine clearance from serum creatinine](https://doi.org/10.1159/000180580).
2. [NIDDK: eGFR equations for adults](https://www.niddk.nih.gov/research-funding/research-programs/kidney-clinical-research-epidemiology/laboratory/glomerular-filtration-rate-equations/adults).
3. [NIDDK: 2009 creatinine-based CKiD bedside equation](https://www.niddk.nih.gov/research-funding/research-programs/kidney-clinical-research-epidemiology/laboratory/glomerular-filtration-rate-equations/children-adolescents-young-adults).
4. [University of Iowa: creatinine clearance specimen and timing requirements](https://www.healthcare.uiowa.edu/path_handbook/handbook/test534.html).
5. [Pickering et al. (2012), Four hour creatinine clearance is better than plasma creatinine for monitoring renal function in critically ill patients](https://pmc.ncbi.nlm.nih.gov/articles/PMC3580664/).

These references support the equations and measurement requirements; the case narratives and values were independently composed. They do not establish clinical validation of the examples or of the released software.
