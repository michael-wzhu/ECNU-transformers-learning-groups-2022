# ECNU transformers Reading List

本reading-list是华东师范大学智能知识管理团队(负责人：王晓玲教授)在2022年下半年讨论班的论文阅读目录.

本讨论班围绕transformer模型这个目前大火的模型，分享学习其架构设计，预训练和微调方面的最新研究进展。这个reading-list还在不断地完善中，我们会持续更新这个reading-list. 我们非常欢迎**Pull Request** ! 如果有好的建议，欢迎联系 wzhu@stu.ecnu.edu.cn.


## Contents
* [综述](#surveys)
* [Tutorial](#tutorial)
* [Research papers](#research_papers)
  * [Transformer Architectures](#xxx)
      * [针对序列建模](#Transformer_on_sequence)
      * [Vision Transformers](#vit) 
      * [Graph Transformers](#graph_transformer)
  * [Pre-training](#pretraining)
     * [Language pretraining](#language_pretraining)
     * [Vision pretraining](#vision_pretrain)    
     * [Vision-Language pretraining](#vision_lang) 
     * [Speech-Language pretraining](#speech_lang)
     * [Document pretraining](#document_pretraining)
     * [Time-series pretraining](#ts_pretraining) 
     * [Recomendation pretraining](#pretraining_in_recomendation) 
  * [Fine-tuning](#finetuning)
      * [微调涨分有效方法](#微调涨分有效方法)  
      * [Robustness](#robustness)  
      * [Parameter efficient fine-tuning](#pet) 
      * [Prompt learning](#prompt_learning) 
      * [Debiasing](#debiasing)
      * [Inference speedup](#inference_speedup)


<h2 id="surveys">综述</h2>

* [Lin, Tianyang, Yuxin Wang, Xiangyang Liu and Xipeng Qiu. **A Survey of Transformers.**](https://arxiv.org/pdf/2106.04554.pdf)
* [Qiu, Xipeng, Tianxiang Sun, Yige Xu, Yunfan Shao, Ning Dai and Xuanjing Huang. **Pre-trained Models for Natural Language Processing: A Survey.** ](https://arxiv.org/abs/2003.08271)
* [Khan, Salman Hameed, Muzammal Naseer, Munawar Hayat, Syed Waqas Zamir, Fahad Shahbaz Khan and Mubarak Shah. “Transformers in Vision: A Survey.” ACM Computing Surveys (CSUR) (2022)](https://dl.acm.org/doi/abs/10.1145/3505244)
* [Yu, Junliang, Hongzhi Yin, Xin Xia, Tong Chen, Jundong Li and Zi-Liang Huang. “Self-Supervised Learning for Recommender Systems: A Survey.”](https://arxiv.org/abs/2203.15876)
* [Han, Xu, Zhengyan Zhang, Ning Ding, Yuxian Gu, Xiao Liu, Yuqi Huo, Jiezhong Qiu, Liang Zhang, Wentao Han, Minlie Huang, Qin Jin, Yanyan Lan, Yang Liu, Zhiyuan Liu, Zhiwu Lu, Xipeng Qiu, Ruihua Song, Jie Tang, Ji-rong Wen, Jinhui Yuan, Wayne Xin Zhao and Jun Zhu. “Pre-Trained Models: Past, Present and Future.” AI Open 2 (2021): 225-250.](https://arxiv.org/abs/2106.07139)
* [Du, Yifan, Zikang Liu, Junyi Li and Wayne Xin Zhao. “A Survey of Vision-Language Pre-Trained Models.” IJCAI (2022).](https://arxiv.org/abs/2202.10936)
* [Liu, Pengfei, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi and Graham Neubig. “Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing.”](https://arxiv.org/abs/2107.13586)
* [Transformers in Time Series: A Survey ](https://github.com/qingsongedu/time-series-transformers-review)


<h2 id="tutorial">Tutorial</h2>

* [Vision-Language Pretraining: Current Trends and the Future. ACL-2022.](https://vlp-tutorial-acl2022.github.io/)
* [Tutorial on MultiModal Machine Learning. CVPR-2022.](https://cmu-multicomp-lab.github.io/mmml-tutorial/cvpr2022/)
* [Beyond Convolutional Neural Networks. CVPR-2022.](https://sites.google.com/view/cvpr-2022-beyond-cnn)
* [Denoising Diffusion-based Generative Modeling: Foundations and Applications. CVPR-2022.](https://cvpr2022-tutorial-diffusion-models.github.io/) 
* [Pre-training Methods for Neural Machine Translation. ACL-2021.](https://lileicc.github.io/TALKS/2021-ACL/)
* [Contrastive Data and Learning for Natural Language Processing. ACL-2022]( https://contrastive-nlp-tutorial.github.io/)
* [Robust Time Series Analysis and Applications: An Industrial Perspective. KDD-2022](https://github.com/DAMO-DI-ML/KDD2022-Tutorial-Time-Series)
* [Time Series in Healthcare: Challenges and Solutions. AAAI-2022](https://www.vanderschaar-lab.com/time-series-in-healthcare/)


<h2 id="research_papers">Research papers</h2>


<h3 id="Transformer_architecture">Transformer Architecture</h3>


<h4 id="Transformer_on_sequence">序列建模中的Transformer架构</h4>

* [What Dense Graph Do You Need for Self-Attention](https://arxiv.org/abs/2205.14014)
* [Flowformer: Linearizing Transformers with Conservation Flows](https://arxiv.org/abs/2202.06258)
* [cosFormer: Rethinking Softmax in Attention](https://arxiv.org/abs/2202.08791)
* [Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention](https://arxiv.org/abs/2102.03902)
* [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://arxiv.org/abs/2106.13008)
* [Directed Acyclic Transformer for Non-Autoregressive Machine Translation (ICML2022)](https://proceedings.mlr.press/v162/huang22m.html)


<h4 id="vit">Vision Transformers</h4>

* [MPViT: Multi-Path Vision Transformer for Dense Prediction](https://arxiv.org/abs/2112.11010)
* [Mobile-Former: Bridging MobileNet and Transformer](https://arxiv.org/abs/2108.05895)
* [MetaFormer is Actually What You Need for Vision](https://arxiv.o**rg/abs/2111.11418)
* [Shunted Self-Attention via Multi-Scale Token Aggregation](https://arxiv.org/abs/2111.15193)


<h4 id="graph_transformer">Graph Transformers</h4>

* [Wu, Zhanghao, Paras Jain, Matthew A. Wright, Azalia Mirhoseini, Joseph Gonzalez and Ioan Cristian Stoica. “Representing Long-Range Context for Graph Neural Networks with Global Attention.” NeurIPS (2021). ](https://arxiv.org/abs/2201.08821)
* [ Hussain, Md Shamim, Mohammed J. Zaki and D. Subramanian. “Edge-augmented Graph Transformers: Global Self-attention is Enough for Graphs.”](http://arxiv-export-lb.library.cornell.edu/pdf/2108.03348)
* [Zhao, Jianan, Chaozhuo Li, Qian Wen, Yiqi Wang, Yuming Liu, Hao Sun, Xing Xie and Yanfang Ye. “Gophormer: Ego-Graph Transformer for Node Classification.”](https://arxiv.org/abs/2110.13094)
* [Rethinking Graph Transformers with Spectral Attention](https://arxiv.org/abs/2106.03893)


<h3 id="pretraining">Pre-training</h3>


<h4 id="language_pretraining">Language pretraining</h4>

* [He, Pengcheng, Jianfeng Gao and Weizhu Chen. “DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing.” ](https://arxiv.org/abs/2111.09543)
* [mT6: Multilingual Pretrained Text-to-Text Transformer with Translation Pairs](https://arxiv.org/abs/2104.08692)
* [Shuming Ma, Li Dong, Shaohan Huang, Dongdong Zhang, Alexandre Muzio, Saksham Singhal, Hany Hassan Awadalla, Xia Song, Furu Wei. DeltaLM: Encoder-Decoder Pre-training for Language Generation and Translation by Augmenting Pretrained Multilingual Encoders. ](https://arxiv.org/abs/2106.13736)
* [Qin, Yujia, Jiajie Zhang, Yankai Lin, Zhiyuan Liu, Peng Li, Maosong Sun and Jie Zhou. “ELLE: Efficient Lifelong Pre-training for Emerging Data.” ](https://arxiv.org/abs/2203.06311)
* [Unified Structure Generation for Universal Information Extraction](https://arxiv.org/abs/2203.12277)


<h4 id="vision_pretrain">Vision pretraining</h4>

* [Efficient Self-supervised Vision Pretraining with Local Masked Reconstruction](https://arxiv.org/abs/2206.00790)
* [Bootstrapped Masked Autoencoders for Vision BERT Pretraining](https://arxiv.org/abs/2207.07116)
* [BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254)


<h4 id="vision_lang">Vision-Language pretraining</h4>

* [VLMo: Unified vision-language pre-training](https://arxiv.org/abs/2111.02358)
* [VL-BEiT: Generative Vision-Language Pre-training](https://arxiv.org/abs/2206.01127)
* [Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks (BEiT-3)](https://arxiv.org/abs/2208.10442)


<h4 id="speech_lang">Speech-Language pretraining</h4>

* [SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing](https://arxiv.org/abs/2110.07205)
* [WavLM: Large-Scale Self-Supervised Pre-training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900)
* [Unified Speech-Text Pre-training for Speech Translation and Recognition](https://arxiv.org/abs/2204.05409)
* [Wav2Seq: Pre-training Speech-to-Text Encoder-Decoder Models Using Pseudo Languages](https://arxiv.org/abs/2205.01086)


<h4 id="document_pretraining">Document pretraining</h4>

* [LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318)
* [LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding](https://arxiv.org/abs/2012.14740)
* [LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking](https://arxiv.org/abs/2204.08387)
* [LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding](https://arxiv.org/abs/2104.08836)
* [MarkupLM: markup language model pre-training for visually-rich document understanding](https://arxiv.org/abs/2110.08518)
* [DiT: Self-supervised Document Image Transformer. ](https://arxiv.org/abs/2203.02378)
* [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282)
* [LayoutReader: Pre-training of Text and Layout for Reading Order Detection](https://arxiv.org/abs/2108.11591)

<h4 id="ts_pretraining">Time-series pretraining</h4>

* [Pre-training Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting. KDD-2022](https://dl.acm.org/doi/abs/10.1145/3534678.3539396)
* [Utilizing Expert Features for Contrastive Learning of Time-Series Representations. PMLR-2022](https://proceedings.mlr.press/v162/nonnenmacher22a.html)
* [Self-supervised Contrastive Representation Learning for Semi-supervised Time-Series Classification. TPAMI-Under review](https://arxiv.org/abs/2208.06616)
* [TARNet: Task-Aware Reconstruction for Time-Series Transformer.KDD-2022](https://dl.acm.org/doi/abs/10.1145/3534678.3539329)
* [Self-Supervised Time Series Representation Learning with Temporal-Instance Similarity Distillation. ICML-2022 Pre-training Workshop](https://openreview.net/pdf?id=nhtkdCvVLIh)


<h4 id="pretraining_in_recomendation">Recomendation pretraining</h4>

* [Towards Universal Sequence Representation Learning for Recommender Systems , KDD 2022](https://arxiv.org/pdf/2206.05941.pdf)
* [Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5)](https://arxiv.org/pdf/2203.13366.pdf)
* [CrossCBR: Cross-view Contrastive Learning for Bundle Recommendation](https://arxiv.org/pdf/2206.00242.pdf)
* [XSimGCL: Towards Extremely Simple Graph Contrastive Learning for Recommendation](https://arxiv.org/pdf/2209.02544.pdf)



<h3 id="finetuning">Fine-tuning</h3>


<h4 id="微调涨分有效方法">微调涨分有效方法</h4>

* [R-Drop: Regularized Dropout for Neural Networks](https://arxiv.org/abs/2106.14448)
* [Document-Level Relation Extraction with Adaptive Focal Loss and Knowledge Distillation](https://arxiv.org/abs/2203.10900)
* [Circle Loss: A Unified Perspective of Pair Similarity Optimization](https://arxiv.org/abs/2002.10857)
* [Do We Need Zero Training Loss After Achieving Zero Training Error?](https://arxiv.org/abs/2002.08709)
* [Dissecting Supervised Contrastive Learning](https://arxiv.org/pdf/2102.08817.pdf)


<h4 id="robustness">Robustness</h4>

* [A Word is Worth A Thousand Dollars: Adversarial Attack on Tweets Fools Stock Prediction ](https://arxiv.org/abs/2205.01094)
* [Adversarial Training for Improving Model Robustness? Look at Both Prediction and Interpretation](https://arxiv.org/pdf/2203.12709.pdf)
* [AugLy: Data Augmentations for Robustness](https://arxiv.org/abs/2201.06494)



<h4 id="pet">Parameter efficient fine-tuning</h4>

* [Compacter: Efficient Low-Rank Hypercomplex Adapter Layers](https://arxiv.org/abs/2106.04647)
* [VL-Adapter: Parameter-Efficient Transfer Learning for Vision-and-Language Tasks](https://arxiv.org/abs/2112.06825)
* [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
* [BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models](https://arxiv.org/abs/2106.10199)
* [Delta Tuning: A Comprehensive Study of Parameter Efficient Methods for Pre-trained Language Models](https://arxiv.org/abs/2203.06904)
* [Empowering parameter-efficient transfer learning by recognizing the kernel structure in self-attention](https://arxiv.org/abs/2205.03720)
* [Towards a Unified View of Parameter-Efficient Transfer Learning](https://arxiv.org/abs/2110.04366)
* [HyperPrompt: Prompt-based Task-Conditioning of Transformers](https://arxiv.org/abs/2203.00759)
* [Personalized Prompt Learning for Explainable Recommendation](https://arxiv.org/abs/2202.07371)



<h4 id="prompt_learning">Prompt learning</h4>

* [Multitask Prompted Training Enables Zero-Shot Task Generalization](https://arxiv.org/abs/2110.08207)

* [Chain of Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
* [Can Prompt Probe Pretrained Language Models? Understanding the Invisible Risks from a Causal View](https://arxiv.org/abs/2203.12258)
* [Multitask Prompted Training Enables Zero-Shot Task Generalization](https://arxiv.org/abs/2110.08207)
* [Can large language models reason about medical questions](https://www.x-mol.com/paper/1549474116436914176)
* [Chain of Thought Imitation with Procedure Cloning](https://arxiv.org/abs/2205.10816?context=cs)
* [Inferring Implicit Relations with Language Models](https://arxiv.org/abs/2204.13778)
* [Can language models learn from explanations in context](https://arxiv.org/abs/2204.02329#:~:text=Large%20language%20models%20can%20perform,connect%20examples%20to%20task%20principles.)
* [The Unreliability of Explanations in Few-Shot In-Context Learning](https://arxiv.org/abs/2205.03401?context=cs)
* [Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?](https://arxiv.org/abs/2202.12837)
* [GrIPS: Gradient-free, Edit-based Instruction Search for Prompting Large Language Models](https://arxiv.org/abs/2203.07281)


<h4 id="Debiasing">Debiasing</h4>

* [Don't Discard All the Biased Instances: Investigating a Core Assumption in Dataset Bias Mitigation Techniques](https://arxiv.org/abs/2109.00521)
* [Discover and Mitigate Unknown Biases with Debiasing Alternate Networks](https://arxiv.org/abs/2207.10077)
* [Language-biased image classification: evaluation based on semantic representations](https://openreview.net/forum?id=xNO7OEIcJc6)
* [A Closer Look at Debiased Temporal Sentence Grounding in Videos: Dataset, Metric, and Approach](https://arxiv.org/abs/2203.05243)
* [Barlow constrained optimization for Visual Question Answering](http://arxiv.org/abs/2203.03727#:~:text=Visual%20question%20answering%20is%20a,the%20question%20and%20image%20modalities.)
* [Debiasing Methods in Natural Language Understanding Make Bias More Accessible](http://arxiv.org/abs/2109.04095)
* [How Gender Debiasing Affects Internal Model Representations, and Why It Matters](http://arxiv.org/abs/2204.06827)
* [Bias Mitigation in Machine Translation Quality Estimation](https://aclanthology.org/2022.acl-long.104)


<h4 id="inference_speedup">Inference speedup</h4>

* [Structured Pruning Learns Compact and Accurate Models](https://arxiv.org/abs/2204.00408)
* [Train Flat, Then Compress: Sharpness-Aware Minimization Learns More Compressible Models](https://arxiv.org/abs/2205.12694)
* [PLATON: Pruning Large Transformer Models with Upper Confidence Bound of Weight Importance](https://arxiv.org/abs/2206.12562)
* [Adapt-and-Distill: Developing Small, Fast and Effective Pretrained Language Models for Domains](https://arxiv.org/abs/2106.13474)
* [Exploring Extreme Parameter Compression for Pre-trained Language Models](http://arxiv.org/abs/2205.10036)
* [MoEBERT: from BERT to Mixture-of-Experts via Importance-Guided Adaptation](https://arxiv.org/abs/2204.07675?context=cs)
* [EdgeFormer: A Parameter-Efficient Transformer for On-Device Seq2seq Generation](https://arxiv.org/abs/2202.07959)