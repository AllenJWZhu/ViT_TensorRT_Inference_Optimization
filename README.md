# ViT: TensorRT-based inference optimization

At the end of October 2020, Vision Transformers (ViT) was proposed by the Google Brain team in the paper “AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE ViT is a model that applies the Transformer architecture to computer vision tasks.

! [ViT paper] (. \ViT-paper.png)

Transformer was originally designed for Natural Language Processing (NLP) tasks, but ViT has successfully extended it to visual tasks such as image categorization, with very good results. The release of this result provides a crucial boost to the currently hot multimodal macromodeling.


## 1. What ViT is used for and how it works

Vision Transformer (ViT) demonstrates its great potential in the field of computer vision by applying the Transformer architecture to image tasks. Despite some challenges, such as the need for large-scale data and high computational complexity, the excellent performance of ViT in several tasks shows that Transformer has a promising application in vision tasks. The following is a detailed description of the main uses of ViT and its working principle.

### 1.1 Uses of ViT

ViT is mainly used for the following computer vision tasks:

1. **Image classification**:
   - ViT was originally designed for image classification tasks. It can be trained on large-scale datasets (e.g., ImageNet) and used in classification tasks.

2. **Target Detection**:
   - ViT can be extended to target detection tasks by combining other modules (e.g., Region Proposal Network, RPN) to recognize multiple objects in an image.
   - The relationship between the target objects and the global image context is considered and the final set of predictions is directly outputted in a parallel fashion. This architecture is able to process multiple objects simultaneously, thus accomplishing the detection task efficiently.
   
3. **Semantic Segmentation**:
   - ViT can also be used for semantic segmentation tasks by classifying each pixel in an image.
   - The performance of semantic segmentation is improved by utilizing Transformer's self-attention mechanism to capture global contextual information. By considering the global information in the image, it is able to more accurately recognize the classes to which different regions belong and generate finer segmentation results.
   
4. **Image generation and restoration**:
   - ViT can be used for image generation and restoration tasks to generate high-quality images through a self-attentive mechanism.

### 1.2 How ViT works.

! [vit-structure] (. \ViT-model-structure.png)

The working principle of ViT mainly includes the following steps:

1. **Image Chunking (Patch Embedding)**:
   - The input image is divided into non-overlapping chunks (patches) of fixed size. For example, given a 224x224 image, it can be divided into 16x16 patches, resulting in 14x14=196 patches.
   - Each patches is spread and mapped to a high dimensional vector space to form an embedding vector.

2. **Position Embedding**:
   - Since the Transformer does not have built-in position information, position encoding needs to be added to preserve the position information of the small blocks in the image.
   - The position encoding is added to the small block embedding vectors to form the input sequence.

3. **Transfor

Translated with DeepL.com (free version)
