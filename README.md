# ECNU transformers Reading List

本reading-list是华东师范大学智能知识管理团队(负责人：王晓玲教授)在2022年下半年讨论班的论文阅读目录.

本讨论班围绕transformer模型这个目前大火的模型，分享学习其架构设计，预训练和微调方面的最新研究进展。这个reading-list还在不断地完善中，我们会持续更新这个reading-list. 我们非常欢迎**Pull Request** ! 如果有好的建议，欢迎联系 wzhu@stu.ecnu.edu.cn.


## Contents
* [Transformer Architectures](#xxx)
    * [针对序列建模](#xxx)
    * [ViT](#)
    * [针对图数据](#xxx)
 * [Pre-training](#)
    * [NLP中的预训练改进](#)  --》 DeBERTam, UIE,  
    * [ViT预训练](#)
    * [多模态预训练](#xxxx)
        * [VL](#xxx)    --> BEIT-3， diffusion model, clip
        * [文档](#xxxx)    --> LayoutLM-v3
    * [推荐中的预训练](#xxxx)
    * [图上的预训练](#xxxx)
* [Fine-tuning](#ML)
    * [微调涨分有效方法](#xxxx)  --> r-drop, flooding
    * [鲁棒性](#xxxx)    --> flooding-X
    * [参数高效微调](#xxxxxx)
      * adapters, prompt, 本征空间，无梯度优化
    * [提示学习](#xxxxxx)
      * prompting: chain-of-thought prompting， 
Can Prompt Probe Pretrained Language Models? Understanding the Invisible Risks from a Causal View
      * prompting： Multitask Prompted Training Enables Zero-Shot Task Generalization
      * demonstration tuning：
      * instruction tuning：
    * [去偏](#xxxx)
      * 模态去偏；
      * 特征去偏；
    * [推理加速](#xxxx)
      * 剪枝：PLATON: Pruning Large Transformer Models with Upper Confidence Bound of Weight Importance
      * 蒸馏： Adapt-and-Distill: Developing Small, Fast and Effective Pretrained Language Models for Domains
      * MoE化  --> MoEBERT: from BERT to Mixture-of-Experts via Importance-Guided Adaptation
      * 针对设备优化：EdgeFormer


<h2 id="surveys">综述</h2>

* Lin, Tianyang, Yuxin Wang, Xiangyang Liu and Xipeng Qiu. **A Survey of Transformers.** ArXiv abs/2106.04554 (2021): n. pag.
* Qiu, Xipeng, Tianxiang Sun, Yige Xu, Yunfan Shao, Ning Dai and Xuanjing Huang. **Pre-trained Models for Natural Language Processing: A Survey.** ArXiv abs/2003.08271 (2020): n. pag.
* Khan, Salman Hameed, Muzammal Naseer, Munawar Hayat, Syed Waqas Zamir, Fahad Shahbaz Khan and Mubarak Shah. “Transformers in Vision: A Survey.” ACM Computing Surveys (CSUR) (2022): n. pag.
* Yu, Junliang, Hongzhi Yin, Xin Xia, Tong Chen, Jundong Li and Zi-Liang Huang. “Self-Supervised Learning for Recommender Systems: A Survey.” ArXiv abs/2203.15876 (2022): n. pag.
* Han, Xu, Zhengyan Zhang, Ning Ding, Yuxian Gu, Xiao Liu, Yuqi Huo, Jiezhong Qiu, Liang Zhang, Wentao Han, Minlie Huang, Qin Jin, Yanyan Lan, Yang Liu, Zhiyuan Liu, Zhiwu Lu, Xipeng Qiu, Ruihua Song, Jie Tang, Ji-rong Wen, Jinhui Yuan, Wayne Xin Zhao and Jun Zhu. “Pre-Trained Models: Past, Present and Future.” AI Open 2 (2021): 225-250.
* Du, Yifan, Zikang Liu, Junyi Li and Wayne Xin Zhao. “A Survey of Vision-Language Pre-Trained Models.” IJCAI (2022).
* Liu, Pengfei, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi and Graham Neubig. “Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing.” ArXiv abs/2107.13586 (2021): n. pag.

<h2 id="tutorial">Tutorial</h2>

* Vision-Language Pretraining: Current Trends and the Future. ACL-2022. https://vlp-tutorial-acl2022.github.io/
* Tutorial on MultiModal Machine Learning. CVPR-2022. https://cmu-multicomp-lab.github.io/mmml-tutorial/cvpr2022/
* Beyond Convolutional Neural Networks. CVPR-2022. https://sites.google.com/view/cvpr-2022-beyond-cnn
* Denoising Diffusion-based Generative Modeling: Foundations and Applications. CVPR-2022. https://cvpr2022-tutorial-diffusion-models.github.io/
* Pre-training Methods for Neural Machine Translation. ACL-2021. https://lileicc.github.io/TALKS/2021-ACL/



<h2 id="research_papers">Research papers</h2>


<h3 id="Transformer_architecture">Transformer Architecture</h3>


<h4 id="Transformer_on_sequence">序列建模中的Transformer架构</h4>

* What Dense Graph Do You Need for Self-Attention
* Flowformer: Linearizing Transformers with Conservation Flows
* cosFormer: Rethinking Softmax in Attention
* Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention
* Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
* DA-transformer


<h4 id="vit">Vision Transformers</h4>

* MPViT : Multi-Path Vision Transformer for Dense Prediction
  - Paper: https://arxiv.org/abs/2112.11010
  - Code: https://github.com/youngwanLEE/MPViT
  - 中文解读: https://mp.weixin.qq.com/s/Q9-crEOz5IYzZaNoq8oXfg

* Mobile-Former: Bridging MobileNet and Transformer
  - Paper: https://arxiv.org/abs/2108.05895
  - Code: None
  - 中文解读：https://mp.weixin.qq.com/s/yo5KmB2Y7t2R4jiOKI87HQ

* MetaFormer is Actually What You Need for Vision
  - Paper: https://arxiv.o**rg/abs/2111.11418
  - Code: https://github.com/sail-sg/poolformer

* Shunted Self-Attention via Multi-Scale Token Aggregation
  - Paper(Oral): https://arxiv.org/abs/2111.15193
  - Code: https://github.com/OliverRensu/Shunted-Transformer


<h4 id="graph_transformers">Graph Transformers</h4>

* Wu, Zhanghao, Paras Jain, Matthew A. Wright, Azalia Mirhoseini, Joseph Gonzalez and Ioan Cristian Stoica. “Representing Long-Range Context for Graph Neural Networks with Global Attention.” NeurIPS (2021). (https://github.com/ucbrise/graphtrans)
* Hussain, Md Shamim, Mohammed J. Zaki and D. Subramanian. “Edge-augmented Graph Transformers: Global Self-attention is Enough for Graphs.” ArXiv abs/2108.03348 (2021): n. pag.
* Zhao, Jianan, Chaozhuo Li, Qian Wen, Yiqi Wang, Yuming Liu, Hao Sun, Xing Xie and Yanfang Ye. “Gophormer: Ego-Graph Transformer for Node Classification.” ArXiv abs/2110.13094 (2021): n. pag.
* Rethinking Graph Transformers with Spectral Attention


<h3 id="pretraining">Pre-training</h3>

<h4 id="language_pretraining">Language pretraining</h4>

* He, Pengcheng, Jianfeng Gao and Weizhu Chen. “DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing.” ArXiv abs/2111.09543 (2021): n. pag.
* mT6: Multilingual Pretrained Text-to-Text Transformer with Translation Pairs
* DeltaLM: Encoder-Decoder Pre-training for Language Generation and Translation by Augmenting Pretrained Multilingual Encoders. Shuming Ma, Li Dong, Shaohan Huang, Dongdong Zhang, Alexandre Muzio, Saksham Singhal, Hany Hassan Awadalla, Xia Song, Furu Wei. CoRR abs/2106.13736.
* Qin, Yujia, Jiajie Zhang, Yankai Lin, Zhiyuan Liu, Peng Li, Maosong Sun and Jie Zhou. “ELLE: Efficient Lifelong Pre-training for Emerging Data.” ArXiv abs/2203.06311 (2022): n. pag.
* Unified Structure Generation for Universal Information Extraction







<h3 id="finetuning">Fine-tuning</h3>


<h4 id="微调涨分有效方法">微调涨分有效方法</h4>

* R-Drop
* Document-Level Relation Extraction with Adaptive Focal Loss and Knowledge Distillation
* Circle Loss: A Unified Perspective of Pair Similarity Optimization
* flooding
* Dissecting Supervised Contrastive Learning


<h4 id="鲁棒性">鲁棒性</h4>

* Flooding-X: Improving BERT’s Resistance to Adversarial Attacks via Loss-Restricted Fine-Tuning
* A Word is Worth A Thousand Dollars: Adversarial Attack on Tweets Fools Stock Prediction
* Adversarial Training for Improving Model Robustness? Look at Both Prediction and Interpretation
* AugLy: Data Augmentations for Robustness



<h4 id="参数高效微调">参数高效微调</h4>

* Compacter: Efficient Low-Rank Hypercomplex Adapter Layers
* VL-Adapters (CVPR-2022)
* LoRA: Low-Rank Adaptation of Large Language Models
* BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models
* Delta Tuning: A Comprehensive Study of Parameter Efficient Methods for Pre-trained Language Models
* Empowering parameter-efficient transfer learning by recognizing the kernel structure in self-attention
* Towards a Unified View of Parameter-Efficient Transfer Learning
* HyperPrompt: Prompt-based Task-Conditioning of Transformers



<h4 id="提示学习">提示学习</h4>

* T0 

* Chain of Thought Prompting Elicits Reasoning in Large Language Models
* Can Prompt Probe Pretrained Language Models? Understanding the Invisible Risks from a Causal View
* Multitask Prompted Training Enables Zero-Shot Task Generalization
* Can large language models reason about medical questions
* Chain of Thought Imitation with Procedure Cloning
* Inferring Implicit Relations with Language Models
* Can language models learn from explanations in context
* The Unreliability of Explanations in Few-Shot In-Context Learning
* Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?
* GrIPS: Gradient-free, Edit-based Instruction Search for Prompting Large Language Models


<h4 id="去偏">去偏</h4>

* Don't Discard All the Biased Instances: Investigating a Core Assumption in Dataset Bias Mitigation Techniques
* Discover and Mitigate Unknown Biases with Debiasing Alternate Networks
* Language-biased image classification: evaluation based on semantic representations
* A Closer Look at Debiased Temporal Sentence Grounding in Videos: Dataset, Metric, and Approach
* Barlow constrained optimization for Visual Question Answering
* Debiasing Methods in Natural Language Understanding Make Bias More Accessible
* How Gender Debiasing Affects Internal Model Representations, and Why It Matters
* Bias Mitigation in Machine Translation Quality Estimation


<h4 id="推理加速">推理加速</h4>

* Structured Pruning Learns Compact and Accurate Models
* Train Flat, Then Compress: Sharpness-Aware Minimization Learns More Compressible Models

* PLATON: Pruning Large Transformer Models with Upper Confidence Bound of Weight Importance
* Adapt-and-Distill: Developing Small, Fast and Effective Pretrained Language Models for Domains
* Exploring Extreme Parameter Compression for Pre-trained Language Models

* MoEBERT: from BERT to Mixture-of-Experts via Importance-Guided Adaptation
* EdgeFormer