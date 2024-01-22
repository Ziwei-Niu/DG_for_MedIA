# Domain Generalization for Medical Image Analysis

## Contributors

<!-- readme: <Ziwei-Niu>,collaborators,<zerone-fg>,contributors,<username3>/- -start -->
<!-- readme: <username1>,collaborators,<username3>,contributors,<username3>/- -end -->

🔥 This is a repository for organizing papers ,codes, and etc related to **Domain Generalization for medical Image Analysis**.

💗 Medical Image Analysis (MedIA) plays a critical role in computer aided diagnosis system, enabling accurate diagnosis and assessment for various diseases. Over the last decade, deep learning (DL) has demonstrated great success in automating various MedIA tasks such as disease diagnosis, lesion segmentation, prognosis prediction, etc. Despite their success, in many real-world healthcare scenarios, the difference in the image acquisition, such as device manufacturer, scanning protocol, image sequence, and modality, introduces domain shifts, resulting in a significant decline in performance when deploying the well-trained model to clinical sites with different data distributions. Additionally, considering that medical data involves privacy concerns, data sharing restrictions and requires manual annotations by medical experts, collecting data from all possible domains to train DL models is expensive and even prohibitively impossible. Therefore, enhancing the generalization ability of DL models in MedIA is crucial in both clinical and academic fields.

🎯 We hope that this repository can provide assistance to researchers and practitioners in medical image analysis and domain generalization.

# Table of Contents
- [Domain Generalization for Medical Image Analysis](#domain-generalization-for-medical-image-analysis)
- [Table of Contents](#table-of-contents)
- [Papers (ongoing)](#papers-ongoing)
  - [Data Level](#data-level)
    - [Data Augmentation-Based](#data-augmentation-based)
      - [(a) Normalization-based](#a-normalization-based)
      - [(b) Randomization-based](#b-randomization-based)
      - [(c) Adversarial-based](#c-adversarial-based)
    - [Data Generation-Based](#data-generation-based)
  - [Feature Level Generalization](#feature-level-generalization)
    - [Invariant Feature Representation](#invariant-feature-representation)
      - [Feature normalization](#feature-normalization)
      - [Explicit feature alignment](#explicit-feature-alignment)
      - [Domain adversarial learning](#domain-adversarial-learning)
    - [Feature disentanglement](#feature-disentanglement)
      - [Multi-component learning](#multi-component-learning)
      - [Generative training](#generative-training)
    - [Feature norm](#feature-norm)
- [Datasets](#datasets)
- [Libraries](#libraries)
- [Other Resources](#other-resources)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)
# Papers (ongoing)

## Data Level
### Data Augmentation-Based 
#### (a) Normalization-based

- [Improved Domain Generalization for Cell Detection in Histopathology Images via Test-Time Stain Augmentation](https://link.springer.com/chapter/10.1007/978-3-031-16434-7_15)  
*Publication*: MICCAI 2022 
*Summary*: Propose a test-time stain normalization method for cell detection in histopathology images, which transforms the test images by mixing their stain color with that of the source domain, so that the mixed images may better resemble the source images or their stain-transformed versions used for training.

- [Tackling Mitosis Domain Generalization in Histopathology Images with Color Normalization](https://link.springer.com/chapter/10.1007/978-3-031-33658-4_22)  
*Publication*: MICCAI Challenge 2022 
*Summary*: Employ a color normalization method in their architecture for mitosis detection in histopathology images.

- (**EDT**) [Improve Unseen Domain Generalization via Enhanced Local Color Transformation](https://link.springer.com/chapter/10.1007/978-3-030-59713-9_42)  
*Publication*: MICCAI 2020 
*Summary*: Propose Enhanced Domain Transformation (EDT) for diabetic retinopathy classification, which aims to project the images into a color space that aligns the distribution of source data and unseen target data.

#### (b) Randomization-based
**Image-space**

- (**SLAug**) [Rethinking Data Augmentation for Single-Source Domain Generalization in Medical Image Segmentation](https://ojs.aaai.org/index.php/AAAI/article/view/25332)  
*Publication*: AAAI 2023 
*Summary*: Rethink the data augmentation strategy for DG in medical image segmentation and propose a location-scale augmentation strategy, which performs constrained Bezier transformation on both global and local (i.e. class-level) regions to enrich the informativeness and diversity of augmented
samples.
[[Code]](https://github.com/Kaiseem/SLAug)


**Frequency-space**
- (**AmpMix**) [Fourier-based augmentation with applications to domain generalization](https://www.sciencedirect.com/science/article/pii/S0031320323001747)
*Publication*: Pattern Recognition 2023  
*Summary*: Propose a Fourier-based data augmentation strategy called AmpMix by linearly interpolating the amplitudes of two images while keeping their phases unchanged to simulated domain shift. Additionally a consistency training between different augmentation views is incorporated to learn invariant representation. 


- [Generalizable Cross-modality Medical Image Segmentation via Style Augmentation and Dual Normalization](https://openaccess.thecvf.com/content/CVPR2022/html/Zhou_Generalizable_Cross-Modality_Medical_Image_Segmentation_via_Style_Augmentation_and_Dual_CVPR_2022_paper.html)
*Publication*: CVPR 2022  
*Summary*: Employ Bezier Curves to augment single source domain into different styles and split them into source-similar domain and source-dissimilar domain.
[[Code]](https://github.com/zzzqzhou/Dual-Normalization)


- [Domain Generalization in Restoration of Cataract Fundus Images Via High-Frequency Components](https://ieeexplore.ieee.org/abstract/document/9761606)  
*Publication*: ISBI 2022 
*Summary*: Cataract-like fundus images are randomly synthesized from an identical clear image by adding cataractous blurry. Then, high-frequency components are extracted from the cataract-like images to reduce the domain shift and achieve domain alignment.
[[Code]](https://github.com/HeverLaw/Restoration-of-Cataract-Images-via-Domain-Generalization)


- (**ELCFS**) [FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_FedDG_Federated_Domain_Generalization_on_Medical_Image_Segmentation_via_Episodic_CVPR_2021_paper.html)
*Publication*: CVPR 2021  
*Summary*: Propose a continuous frequency space interpolation mechanism for federated medical domain generalization, which exchanges amplitude spectrum across clients to transmit the distribution information, while keeping the phase spectrum with core semantics locally for privacy protection. 
[[Code]](https://github.com/liuquande/FedDG-ELCFS)


- (**BigAug**) [Generalizing Deep Learning for Medical Image Segmentation to Unseen Domains via Deep Stacked Transformation](https://ieeexplore.ieee.org/abstract/document/8995481)  
*Publication*: IEEE TMI 2020 
*Summary*: Propose a deep stacked transformation approach by applying extensive random typical transformations on a single source domain to simulate the domain shift.


**Feature-space**

- [Improving the Generalizability of Convolutional Neural Network-Based Segmentation on CMR Images](https://www.frontiersin.org/articles/10.3389/fcvm.2020.00105/full)  
*Publication*: Frontiers in Cardiovascular Medicine 2020 
*Summary*: Propose a simple yet effective way for improving the network generalization ability by carefully designing data normalization and augmentation strategies.


#### (c) Adversarial-based
- (**AADG**) [AADG: Automatic Augmentation for Domain Generalization on Retinal Image Segmentation](https://ieeexplore.ieee.org/abstract/document/9837077)
*Publication*: TMI 2022  
*Summary*: Introduce a novel proxy task maximizing the diversity among multiple augmented novel domains as measured by the Sinkhorn distance in a unit sphere space to achieve automated augmentation. Adversarial training and deep reinforcement learning are employed to efficiently search the objectives.
[[Code]](https://github.com/CRazorback/AADG)


- (**ADS**) [Adversarial Consistency for Single Domain Generalization in Medical Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-16449-1_64)
*Publication*: MICCAI 2022  
*Summary*: Synthesize the new domains via learning an adversarial domain synthesizer (ADS), and propose to keep the underlying semantic information between the source image and the synthetic image via a mutual information regularizer.


- (**MaxStyle**) [MaxStyle: Adversarial Style Composition for Robust Medical Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-16443-9_15)
*Publication*: MICCAI 2022  
*Summary*: Propose a data augmentation framework called MaxStyle, which augments data with improved image style diversity and hardness, by expanding the style space with noise and searching for the worst-case style composition of latent features via adversarial training.
[[Code]](https://github.com/cherise215/MaxStyle)


- (**AdverIN**) [Domain Generalization with Adversarial Intensity Attack for Medical Image Segmentation](https://arxiv.org/abs/2304.02720)
*Publication*: Arxiv 2023  
*Summary*: Propose Adversarial Intensity Attack (AdverIN) that introduce an adversarial attack on the data intensity distribution, which leverages adversarial training to generate training data with an infinite number of styles and increase data diversity while preserving essential content information.

- (**TeSLA**) [TeSLA: Test-Time Self-Learning With Automatic Adversarial Augmentation](https://openaccess.thecvf.com/content/CVPR2023/html/Tomar_TeSLA_Test-Time_Self-Learning_With_Automatic_Adversarial_Augmentation_CVPR_2023_paper.html)
*Publication*: CVPR 2023  
*Summary*: Propose a method that combines knowledge distillation with adversarial-based data augmentation for cross-site medical image segmentation tasks.
[[Code]](https://github.com/devavratTomar/TeSLA)


### Data Generation-Based

- [Test-Time Image-to-Image Translation Ensembling Improves Out-of-Distribution Generalization in Histopathology](https://link.springer.com/chapter/10.1007/978-3-031-16434-7_12)
*Publication*: MICCAI 2022  
*Summary*: Utilize multi-domain image-to-image translation model StarGanV2 and projects histopathology test images from unseen domains to the source domains, classify the projected images and ensemble their predictions.
[[Code]](https://gitlab.com/vitadx/articles/test-time-i2i-translation-ensembling)

- (**VFT**) [Domain Generalization for Retinal Vessel Segmentation with Vector Field Transformer](https://proceedings.mlr.press/v172/hu22a.html)
*Publication*: PMLR 2022  
*Summary*: Apply auto-encoder to generate different styles of enhanced vessel maps for augmentation and uses Hessian matrices of an image for segmentation as vector fields better capture the morphological features and suffer less from covariate shift.
[[Code]](https://github.com/MedICL-VU/Vector-Field-Transformer)

- (**CIRCLe**) [CIRCLe: Color Invariant Representation Learning for Unbiased Classification of Skin Lesions](https://link.springer.com/chapter/10.1007/978-3-031-25069-9_14)
*Publication*: ECCV Workshop 2022  
*Summary*: Use a Star Generative Adversarial Network (StarGAN) to transform skin types (style), and enforce the feature representation to be invariant across different skin types. 
[[Code]](https://github.com/arezou-pakzad/CIRCLe)


- [Semantic Segmentation with Generative Models: Semi-Supervised Learning and Strong Out-of-Domain Generalization](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Semantic_Segmentation_With_Generative_Models_Semi-Supervised_Learning_and_Strong_Out-of-Domain_CVPR_2021_paper.html)
*Publication*: CVPR 2021  
*Summary*: Propose a fully generative approach to semantic segmentation based on StyleGAN2, that models the joint image-label distribution and synthesizes both images and their semantic segmentation masks.
[[Code]](https://github.com/nv-tlabs/semanticGAN_code)

- (**GADG**)[Generative Adversarial Domain Generalization via Cross-Task Feature Attention Learning for Prostate Segmentation](https://link.springer.com/chapter/10.1007/978-3-030-92270-2_24)
*Publication*: ICONIP 2021  
*Summary*: Propose a new Generative Adversarial Domain Generalization (GADG) network, which can achieve the domain generalization through the generative adversarial learning on multi-site prostate MRI images. Additionally, to make the prostate segmentation network learned from the source domains still have good performance in the target domain, a Cross-Task Attention Module (CTAM) is designed to transfer the main domain generalized features from the generation branch to the segmentation branch.


- (**STRAP**) [Learning Domain-Agnostic Visual Representation for Computational Pathology Using Medically-Irrelevant Style Transfer Augmentation](https://ieeexplore.ieee.org/document/9503389)
*Publication*: TMI 2021  
*Summary*: Propose a style transfer-based aug- mentation (STRAP) method for a tumor classification task, which applies style transfer from non-medical images to histopathology images.
[[Code]](https://github.com/rikiyay/style-transfer-for-digital-pathology)


- [Multimodal Self-supervised Learning for Medical Image Analysis](https://link.springer.com/chapter/10.1007/978-3-030-78191-0_51)
*Publication*: IPMI 2021  
*Summary*: Propose a novel approach leveraging self-supervised learning through multimodal jigsaw puzzles for cross-modal medical image synthesis tasks. Additionally, to increase the quantity of multimodal data, they design a cross-modal generation step to create synthetic images from one modality to another using the CycleGAN-based translation model. 


- (**STDGNs**)[Random Style Transfer Based Domain Generalization Networks Integrating Shape and Spatial Information](https://link.springer.com/chapter/10.1007/978-3-030-68107-4_21)
*Publication*: STACOM 2020  
*Summary*: Propose novel random style transfer based domain general- ization networks incorporating spatial and shape information based on GANs.




## Feature Level Generalization
### Invariant Feature Representation
#### Feature normalization 

- (**SS-Norm**)[SS-Norm: Spectral-spatial normalization for single-domain generalization with application to retinal vessel segmentation](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.12782)
*Publication*: IET IP 2023  
*Summary*: Decompose the feature into multiple frequency components by performing discrete cosine transform and analyze the semantic contribution degree of each component. Then reweight the frequency components of features and therefore normalize the distribution in the spectral domain.


- [Generalizable Cross-modality Medical Image Segmentation via Style Augmentation and Dual Normalization](https://openaccess.thecvf.com/content/CVPR2022/html/Zhou_Generalizable_Cross-Modality_Medical_Image_Segmentation_via_Style_Augmentation_and_Dual_CVPR_2022_paper.html)
*Publication*: CVPR 2022  
*Summary*: Design a dual-normalization module to estimate domain distribution information. During the test stage, the model select the nearest feature statistics according to style-embeddings in the dual-normalization module to normalize target domain features for generalization.
[[Code]](https://github.com/zzzqzhou/Dual-Normalization)



#### Explicit feature alignment

- [Domain Generalization for Medical Imaging Classification with Linear-Dependency Regularization](https://proceedings.neurips.cc/paper_files/paper/2020/file/201d7288b4c18a679e48b31c72c30ded-Paper.pdf)
*Publication*: NeurIPS 2020  
*Summary*: Adopt Kullback-Leibler (KL) divergence to align the distributions of latent features extracted from multiple source domains with a predefined prior distribution.
[[Code]](https://github.com/wyf0912/LDDG)


- [Measuring Domain Shift for Deep Learning in Histopathology](https://ieeexplore.ieee.org/abstract/document/9234592)
*Publication*: JBHI 2020  
*Summary*: Design a dual-normalization module to estimate domain distribution information. During the test stage, the model select the nearest feature statistics according to style-embeddings in the dual-normalization module to normalize target domain features for generalization.
[[Code]](https://github.com/zzzqzhou/Dual-Normalization)



#### Domain adversarial learning
- (**ARMED**) [Adversarially-Regularized Mixed Effects Deep Learning (ARMED) Models Improve Interpretability, Performance, and Generalization on Clustered (non-iid) Data](https://ieeexplore.ieee.org/abstract/document/10016237)
*Publication*: IEEE TPAMI 2023  
*Summary*: Propose a general-purpose framework for Adversarially-Regularized Mixed Effects Deep learning (ARMED). The ARMED employ an adversarial classifier to regularize the model to learn cluster-invariant fixed effects (domain invariant). The classifier attempts to predict the cluster membership based on the learned features, while the feature extractor is penalized for enabling this prediction.


### Feature disentanglement

#### Multi-component learning
- (**MI-SegNet**) [MI-SegNet: Mutual Information-Based US Segmentation for Unseen Domain Generalization](https://arxiv.org/abs/2303.12649)
*Publication*: MICCAI 2023  
*Summary*: Propose MI-SegNet for ultrasound image segmentation. MI-SegNet employs two encoders that separately extract anatomical and domain features from images, and Mutual Information Neural Estimation (MINE) approximation is used to minimize the mutual information between these features.


- (**CDDSA**) [Contrastive Domain Disentanglement for Generalizable Medical Image Segmentation](https://arxiv.org/abs/2205.06551)
*Publication*: Arxiv 2022  
*Summary*: Propose Contrastive Domain Disentanglement and Style Augmentation (CDDSA) for image segmentation in the fundus and MR images. This method introduce a disentangle network to decompose medical images into an anatomical representation and a modality representation, and a style contrastive loss function is designed to ensures that style representations from the same domain bear similarity while those from different domains diverge significantly.

- (**DoFE**) [DoFE: Domain-Oriented Feature Embedding for Generalizable Fundus Image Segmentation on Unseen Datasets](https://ieeexplore.ieee.org/abstract/document/9163289)
*Publication*: IEEE TMI 2020  
*Summary*: Proposed Domain-oriented Feature Embedding (DoFE) for fundus image segmentation. The DoFE framework incorporates a domain knowledge pool to learn and store the domain prior information (domain-specic) extracted from the multi-source domains. This domain prior knowledge is then dynamically enriched with the image features to make the semantic features more discriminative, improving the generalization ability of the segmentation networks on unseen target domains.


#### Generative training

- (**DarMo**) [Learning domain-agnostic representation for disease diagnosiss](https://www.researchgate.net/profile/Yizhou-Yu-2/publication/372230975_LEARNING_DOMAIN-AGNOSTIC_REPRESENTATION_FOR_DISEASE_DIAGNOSIS/links/64aacef58de7ed28ba8841c2/LEARNING-DOMAIN-AGNOSTIC-REPRESENTATION-FOR-DISEASE-DIAGNOSIS.pdf)
*Publication*: ICLR 2023  
*Summary*: Leverage structural causal modeling to explicitly model disease-related and center-effects. Guided by this, propose a novel Domain Agnostic Representation Model (DarMo) based on variational Auto-Encoder and design domain-agnostic and domain-aware encoders to respectively capture disease-related features and varied center effects by incorporating a domain-aware batch normalization layer.

- (**DIVA**) [DIVA: Domain Invariant Variational Autoencoders](http://proceedings.mlr.press/v121/ilse20a.html)
*Publication*: PLMR 2022  
*Summary*: Propose Domain-invariant variational autoencoder (DIVA) for malaria cell image classification, which disentangles the features into domain information, category information, and other information, which is learned in the VAE framework.
[[Code]](https://github.com/AMLab-Amsterdam/DIVA)

- (**VDN**) [Variational Disentanglement for Domain Generalization](https://arxiv.org/abs/2109.05826)
*Publication*: TMLR 2022
*Summary*: Propose a Variational Disentanglement Network (VDN) to classify breast cancer metastases. VDN disentangles domain-invariant and domain-specific features by estimating the information gain and maximizing the posterior probability.

### Feature norm
- (**FD**) [Frequency Dropout: Feature-Level Regularization via Randomized Filtering](https://link.springer.com/chapter/10.1007/978-3-031-25066-8_14)
*Publication*: ECCV 2022 Workshop
*Summary*: Propose a training strategy named Frequency Dropout, utilizing common image processing filters to prevent convolutional neural networks from learning frequency-specific imaging features.







# Datasets
> We list the widely used benchmark datasets for domain generalization including classification and segmentation. 

|                                                                 Dataset                                                                  | Task                       | #Domain | #Class | #Sample |                                                                 Description                                                                 |
|:----------------------------------------------------------------------------------------------------------------------------------------:|----------------------------|:-------:|:------:|:-------:|:-------------------------------------------------------------------------------------------------------------------------------------------:|
|                                 [PACS](https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd)                                 | Classification             |    4    |   7    |  9,991  |                                                         Art, Cartoon, Photo, Sketch                                                         |
|                                 [VLCS](https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8)                                 | Classification             |    4    |   5    | 10,729  |                                                     Caltech101, LabelMe, SUN09, VOC2007                                                     |
|                             [Office-Home](https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC)                              | Classification             |    4    |   65   | 15,588  |                                                         Art, Clipart, Product, Real                                                         |
|                          [Office-31](https://mega.nz/file/dSpjyCwR#9ctB4q1RIE65a4NoJy0ox3gngh15cJqKq1XpOILJt9s)                          | Classification             |    3    |   31   |  4,110  |                                                            Amazon, Webcam, DSLR                                                             |
|                         [Office-Caltech](https://drive.google.com/file/d/14OIlzWFmi5455AjeBZLak2Ku-cFUrfEo/view)                         | Classification             |    4    |   10   |  2,533  |                                                        Caltech, Amazon, Webcam, DSLR                                                        |
|                           [Digits-DG](https://drive.google.com/file/d/1A4RJOFj4BJkmliiEL7g9WzNIDUHLxfmm/view)                            | Classification             |    4    |   10   | 24,000  |                                                          MNIST, MNIST-M, SVHN, SYN                                                          |
|                      [Digit-5](https://drive.google.com/file/d/15V7EsHfCcfbKgsDmzQKj_DfXt_XYp_P7/view?usp=sharing)                       | Classification             |    5    |   10   | ~10,000 |                                                       MNIST, MNIST-M, SVHN, SYN, USPS                                                       |
|                                            [Rotated MNIST](https://github.com/Emma0118/mate)                                             | Classification             |    6    |   10   |  7,000  |                                                   Rotated degree: {0, 15, 30, 45, 60, 75}                                                   |
|                                            [Colored MNIST](https://github.com/Emma0118/mate)                                             | Classification             |    3    |   2    |  7,000  |                                                       Colored degree: {0.1, 0.3, 0.9}                                                       |
|                                       [CIFAR-10-C](https://zenodo.org/record/2535967#.YuD3ly-KGu4)                                       | Classification             |   --    |   4    | 60,000  |   The test data are damaged by 15 corruptions (each with 5 intensity levels) drawn from 4 categories (noise, blur, weather, and digital)    |
|                                      [CIFAR-100-C](https://zenodo.org/record/3555552#.YuD31C-KGu4)                                       | Classification             |   --    |   4    | 60,000  |   The test data are damaged by 15 corruptions (each with 5 intensity levels) drawn from 4 categories (noise, blur, weather, and digital)    |
|                                                   [DomainNet](http://ai.bu.edu/M3SDA/)                                                   | Classification             |    6    |  345   | 586,575 |                                           Clipart, Infograph, Painting, Quick-draw, Real, Sketch                                            |
|                                            [miniDomainNet](https://arxiv.org/abs/2003.07325)                                             | Classification             |    4    |  345   | 140,006 |                           A smaller and less noisy version of DomainNet including Clipart, Painting, Real, Sketch                           |
|                                  [VisDA-17](https://github.com/VisionLearningGroup/taskcv-2017-public)                                   | Classification             |    3    |   12   | 280,157 |                                                3 domains of synthetic-to-real generalization                                                |
|               [Terra Incognita](https://lilablobssc.blob.core.windows.net/caltechcameratraps/eccv_18_all_images_sm.tar.gz)               | Classification             |    4    |   10   | 24,788  |                                          Wild animal images taken at locations L100, L38, L43, L46                                          |
|                                            [Prostate MRI](https://liuquande.github.io/SAML/)                                             | Medical image segmentation |    6    |   --   |   116   |                                    Contains prostate T2-weighted MRI data from 6 institutions: Site A~F                                     |
|                          [Fundus OC/OD](https://drive.google.com/file/d/1p33nsWQaiZMAgsruDoJLyatoq5XAH-TH/view)                          | Medical image segmentation |    4    |   --   |  1060   |                                            Contains fundus images from 4 institutions: Site A~D                                             |
| [GTA5-Cityscapes]([GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) and [Cityscapes](https://www.cityscapes-dataset.com)) | Semantic segmentation      |    2    |   --   | 29,966  |                                                2 domains of synthetic-to-real generalization                                                |


# Libraries
> We list the libraries of domain generalization.
- [Transfer Learning Library (thuml)](https://github.com/thuml/Transfer-Learning-Library) for Domain Adaptation, Task Adaptation, and Domain Generalization.
- [DomainBed (facebookresearch)](https://github.com/facebookresearch/DomainBed)  is a suite to test domain generalization algorithms.
- [DeepDG (Jindong Wang)](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDG): Deep domain generalization toolkit, which is easier then DomainBed.
- [Dassl (Kaiyang Zhou)](https://github.com/KaiyangZhou/Dassl.pytorch): A PyTorch toolbox for domain adaptation, domain generalization, and semi-supervised learning.
- [TorchSSL (Jindong Wang)](https://github.com/TorchSSL/TorchSSL): A open library for semi-supervised learning.

# Other Resources
- A collection of domain generalization papers organized by  [amber0309](https://github.com/amber0309/Domain-generalization).
- A collection of domain generalization papers organized by [jindongwang](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#domain-generalization).
- A collection of papers on domain generalization, domain adaptation, causality, robustness, prompt, optimization, generative model, etc, organized by [yfzhang114](https://github.com/yfzhang114/Generalization-Causality).
- A collection of awesome things about domain generalization organized by [junkunyuan](https://github.com/junkunyuan/Awesome-Domain-Generalization).

# Contact
- If you would like to add/update the latest publications / datasets / libraries, please directly add them to this `README.md`.
- If you would like to correct mistakes/provide advice, please contact us by email (nzw@zju.edu.cn).
- You are welcomed to update anything helpful.

# Acknowledgements
- We refer to [Generalizing to Unseen Domains: A Survey on Domain Generalization](https://ieeexplore.ieee.org/abstract/document/9782500) to design the hierarchy of the [Contents](#contents).
- We refer to [junkunyuan](https://github.com/junkunyuan/Awesome-Domain-Generalization), [amber0309](https://github.com/amber0309/Domain-generalization), and [yfzhang114](https://github.com/yfzhang114/Generalization-Causality) to design the details of the papers and datasets.
