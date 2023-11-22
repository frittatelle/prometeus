from datasets import load_dataset
from transformers import ViTImageProcessor
import matplotlib.pyplot as plt
import torch


def import_model(model_path: str) -> ViTImageProcessor:
    return ViTImageProcessor.from_pretrained(model_path)


def preprocess(batch: torch.Tensor, feature_extractor: ViTImageProcessor):
    # take a list of PIL images and turn them to pixel values
    inputs = feature_extractor(
        batch['img'],
        return_tensors='pt'
    )
    # include the labels
    inputs['label'] = batch['label']
    return inputs


def main():

    # get dataset
    tr_dataset = load_dataset('cifar10', split='train', verification_mode='all_checks')
    ts_dataset = load_dataset('cifar10', split='test', verification_mode='all_checks')
    
    # dataset exploration
    num_classes = len(set(tr_dataset['label']))
    labels = tr_dataset.features['label']
    print(num_classes, labels)
    idx = 10

    # model
    model_path = 'google/vit-base-patch16-224-in21k'
    feature_extractor = import_model(model_path)
    
    # extract features
    tensor_features = feature_extractor(tr_dataset[idx]['img'], return_tensors='pt')
    features = tensor_features['pixel_values'].squeeze().numpy()
    features = features.reshape(features.shape[1], features.shape[2], features.shape[0])


    # plt.figure(figsize=(10, 6))
    # plt.imshow(features)
    # plt.title(f"{labels.names[tr_dataset[idx]['label']]}")
    # plt.show()


    # # plot img
    # plt.figure(figsize=(10, 6))
    # plt.imshow(tr_dataset[idx]['img'])
    # plt.title(f"{labels.names[tr_dataset[idx]['label']]}")
    # plt.show()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocessed_tr = tr_dataset.map(preprocess, batched=True, fn_kwargs={"feature_extractor":feature_extractor})
    preprocessed_ts = ts_dataset.map(preprocess, batched=True, fn_kwargs={"feature_extractor":feature_extractor})


if __name__ == '__main__':
    main()