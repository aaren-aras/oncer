<script setup lang="ts">
  import WelcomeItem from './InfoItem.vue'
  import DocumentationIcon from './icons/IconDocumentation.vue'
  import ToolingIcon from './icons/IconTooling.vue'
  import EcosystemIcon from './icons/IconEcosystem.vue'
  import CommunityIcon from './icons/IconCommunity.vue'
  import SupportIcon from './icons/IconSupport.vue'
</script>

<template>
  <WelcomeItem>
    <template #icon>
      <DocumentationIcon />
    </template>
    <template #heading>Background</template>
      <a href="https://pinia.vuejs.org/" target="_blank" rel="noopener">Glioma</a>
      is the most common primary <b>malignant</b> brain tumour.

      <span style="color: rgb(209, 82, 255)">Oncer</span> automates tumour sub-region detection 
      (necrotic core/NCR, edema/ED, and enhancing tumor/ET) from multi-parametric MRI (mpMRI) scans.
  </WelcomeItem>

  <WelcomeItem>
    <template #icon>
      <ToolingIcon />
    </template>
    <template #heading>How to Use</template>
      Upload 4 co-registered, skull-stripped <b>axial</b> MRI slices from the same scan 
      (T1, T2, T1CE, FLAIR), then click <code><b>UPLOAD</b></code> to generate the segmented 
      overlay.
  </WelcomeItem>

  <WelcomeItem>
    <template #icon>
      <EcosystemIcon />
    </template>
    <template #heading>Model Architecture</template>
      <span style="color: rgb(209, 82, 255)">Oncer</span> uses a <a href="https://arxiv.org/abs/1512.03385" target="_blank" rel="noopener">ResU-Net</a>
      deep learning model built with <a href="https://www.tensorflow.org/" target="_blank" rel="noopener">TensorFlow</a>. 
      It combines residual skip connections with deep feature extraction to classify each pixel into background, 
      necrotic core, edema, or enhancing tumor.
  </WelcomeItem>

  <WelcomeItem>
    <template #icon>
      <CommunityIcon />
    </template>
    <template #heading>Future Releases</template>
      Upcoming updates will introduce 3D volumetric patching, multi-planar support (coronal/sagittal),
      native DICOM/NIfTI ingestion, additional imaging sequences (angiography), and model explainability via Grad-CAM.
  </WelcomeItem>

  <WelcomeItem>
    <template #icon>
      <SupportIcon />
    </template>
    <template #heading>Datasets</template>

    The model is trained on the
      <a href="https://www.synapse.org/#!Synapse:syn25829067" target="_blank" rel="noopener">
        RSNA-ASNR-MICCAI BraTS 2021
      </a>
    dataset, found 
      <a href="https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1" target="_blank" rel="noopener">
        here
      </a>, 13.42 GB of multi-institutional clincial mpMRI scans of glioma.
    This app is intended for research purposes only, and NOT for clinical use.
  </WelcomeItem>
</template>
