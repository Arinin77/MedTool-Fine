# Attribution and license status

The distribution license for the project-authored code is pending author approval. This local candidate does not grant a blanket MIT or other open-source license. Source attribution is not a substitute for redistribution permission.

External libraries and pretrained weights are not bundled. The optional dependencies retain their upstream terms: [NumPy](https://github.com/numpy/numpy/blob/v2.2.6/LICENSE.txt), [Transformers](https://github.com/huggingface/transformers/blob/v4.44.2/LICENSE), and [PyTorch](https://github.com/pytorch/pytorch/blob/v2.8.0/LICENSE).

Retrieval uses Qiao Jin et al., *MedCPT: Contrastive Pre-trained Transformers with Large-scale PubMed Search Logs for Zero-shot Biomedical Information Retrieval* (Bioinformatics, 2023). The [official repository notice](https://github.com/ncbi/MedCPT/blob/main/LICENSE) and [query](https://huggingface.co/ncbi/MedCPT-Query-Encoder)/[article](https://huggingface.co/ncbi/MedCPT-Article-Encoder) model cards describe the upstream terms. No model files are redistributed here.

The AgentMD baseline is a task adaptation of the method associated with [Clinical-Tool-Learning](https://github.com/ncbi-nlp/Clinical-Tool-Learning). Its [upstream notice](https://github.com/ncbi-nlp/Clinical-Tool-Learning/blob/main/LICENSE) identifies the NLM government work and requests attribution. The ReAct baseline follows the reasoning/action/observation approach of Shunyu Yao et al., [*ReAct: Synergizing Reasoning and Acting in Language Models*](https://arxiv.org/abs/2210.03629). These method attributions do not imply that external repositories are bundled unchanged.

The calculator resource and both benchmark datasets are excluded. Any later asset release needs its own applicable permissions, notices, and access arrangements.
