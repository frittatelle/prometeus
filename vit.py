from datasets import load_dataset
from transformers import ViTImageProcessor
import matplotlib.pyplot as plt


def import_model(model_path: str) -> ViTImageProcessor:
    return ViTImageProcessor.from_pretrained(model_path)


def main():

    # get dataset
    tr_dataset = load_dataset('cifar10', split='train', verification_mode='all_checks')
    ts_dataset = load_dataset('cifar10', split='test', verification_mode='all_checks')
    
    # dataset exploration
    num_classes = len(set(tr_dataset['label']))
    labels = tr_dataset.features['label']
    print(num_classes, labels)

    # plot img
    idx = 10
    plt.figure(figsize=(10, 6))
    plt.imshow(tr_dataset[idx]['img'])
    plt.title(f"{labels.names[tr_dataset[idx]['label']]}")
    plt.show()

    # model
    model_path = 'google/vit-base-patch16-224-in21k'
    feature_extractor = import_model(model_path)
    print('elle')




if __name__ == '__main__':
    main()