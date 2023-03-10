# Geometric_Transformation_CMR
A simple self-supervised learning task to recognize geometric transformation in medical images (CMR images).

In the field of medical image, deep convolutional neural networks(ConvNets) have achieved great
success in the classification, segmentation, and registration tasks thanks to their unparalleled capacity
to learn image features. However, these tasks often require large amounts of manually annotated
data and are labor-intensive. Therefore, it is of significant importance for us to study unsupervised
semantic feature learning tasks. In our work, we propose to learn features in medical images by
training ConvNets to recognize the geometric transformation applied to images and present a simple
self-supervised task that can easily predict the geometric transformation. We precisely define a
set of geometric transformations in mathematical terms and generalize this model to 3D, taking
into account the distinction between spatial and time dimensions. We evaluated our self-supervised
method on CMR images of different modalities (bSSFP, T2, LGE) and achieved accuracies of
96.4%, 97.5%, and 96.4%, respectively.
