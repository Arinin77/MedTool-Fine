# MedTool-Fine

Core implementation for **MedTool-Fine: Benchmarking Applicability-Aware Selection among Similar Clinical Calculators**.

This package contains CLEAR and the five study baselines: Direct, CoT, Static, ReAct, and AgentMD. It includes candidate retrieval, controlled clarification, calculator execution, and deterministic evaluation. ReAct and AgentMD are the study's task adaptations, not unmodified copies of the original systems.

## Quick start

Python 3.11–3.12 is supported by the package metadata; the offline checks were run with Python 3.12.14. No API key or third-party dependency is needed for the synthetic example:

```console
python scripts/run_offline_demo.py
python -m unittest discover -s tests -v
```

The example uses a newly authored BMI formula, fictional inputs, and scripted model responses. Its ten candidate entries are software test fixtures. They are not paper benchmark cases, original calculator assets, model performance results, or measured token usage. The script uses a temporary directory and removes its generated files when finished. If your environment restricts temporary directories, set `TEMP` and `TMP` to a writable directory first.

An optional editable install is available with `python -m pip install -e .`. Dense retrieval additionally needs the packages in `requirements-retrieval.txt` and the two MedCPT checkpoints listed in `configs/retrieval.json`; weights are not included or automatically downloaded.

## Illustrative cases

[Three short fictional cases](examples/README.md) and their [structured JSON](examples/samples.json) illustrate selection among similar renal calculators: CALL, ASK → CALL, and ASK → ABSTAIN. They were newly authored without patient records or MIMIC-derived content. Initial inputs, controlled replies, and authored references are separated. These reading examples use four candidate cards; they are not benchmark samples or inputs for the packaged runner, whose ten-candidate protocol is unchanged.

## Methods and protocol

- Tool methods receive ten initial candidates from BM25 + MedCPT reciprocal rank fusion. Candidates remain fixed after clarification.
- Interactive methods have a five-model-request budget, including AgentMD selection and baseline `Finish` requests. Each ASK requests 1–8 concepts.
- CLEAR updates its candidate/evidence records from received answers, then submits CALL, ASK, or ABSTAIN. CALL and ABSTAIN end its episode.
- The study uses temperature 0 and a maximum of 12,000 output tokens. The included provider guard follows the GPT-5.4 branch. A live model transport must be supplied separately; it was not exercised in the offline tests.

The source lives under `src/medtool_fine/`. The synthetic script demonstrates the complete runtime/evaluation input contract. Configuration files document the paper settings and placeholder resource paths; they contain no credentials. Calculator references and future answers must stay in the evaluator/controlled-answer environment, outside model-visible inputs.

The bundled `Scorer` and example use the Clinical evaluation branch. `precision_calc.score_call` / `score_episode` retain the Calc numeric rules and require a policy whose `dataset` is `calc`; the corresponding Clinical functions require `clinical`. These branches have different reference formats and must not be interchanged. Dense retrieval and live provider integration have not been validated by the offline example.

## Release scope

This is a **core code release**, with a synthetic software test and three fictional reading examples. The paper evaluates 130 calculators, 220 MTF-Calc tasks, and 596 MTF-Clinical tasks. Those assets, benchmark reference answers, saved model outputs, result logs, development tools, and experiment history are not distributed here. Reproducing the paper's tables requires the corresponding calculator/task resources and model access; the included examples do not reproduce those results.

Calculator redistribution permissions are being reviewed separately. MIMIC-derived resources must follow the [PhysioNet source terms and sharing guidance](https://physionet.org/content/mimiciv/3.1/). The reviewed MTF-Clinical data include manual curation; this package does not promise one-command reconstruction. Full dataset access arrangements are not provided by this code package.

## Attribution and use

See [third-party notices and license status](NOTICE.md). No open-source license has yet been assigned to this source release. Choosing public hosting does not itself choose a license.

ChatGPT/Codex assisted with code implementation, debugging, and package preparation. This research software is not validated for patient-care decisions.
