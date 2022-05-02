import os 
import torch
import pandas as pd
from tqdm import tqdm
from skimage import io
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

csv_file = "archive/names.csv"
root_dir = "archive/pokemonPNG"
aug_dir = "archive/pokemon_augfiles"

class PokeData(Dataset):
    # Initializing the variables  
    def __init__(self, csv_file, root_dir, transform=None):
        self.filenames = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):      
        # returns the size of the dataset
        return len(self.filenames)
    
    def __getitem__(self, index): 
        # loads in the images by joining the directory with the file names
        # in the dataframe
        image_path = os.path.join(self.root_dir, self.filenames.iloc[index, 0])
        img = io.imread(image_path)

        if self.transform:
            img = self.transform(img)
        
        return img
    
    def show_img(dataset):
        plt.figure()
        for i in range(len(dataset)):
            sample = dataset[i]

            ax = plt.subplot(2, 2, i +1)
            plt.tight_layout()
            ax.set_title('Sample #{}'.format(i))
            ax.axis('off')
            plt.imshow(sample)

            if i == 3:
                plt.show()
                break
    
    def get_mean_std(dataloader):
    # VAR[X] = E[X**2] - E[X]**2
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0

        for data in dataloader:
            channels_sum += torch.mean(data, dim=[0,2,3])
            channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
            num_batches += 1
    
        mean = channels_sum/num_batches
        std = (channels_squared_sum/num_batches - mean**2)**0.5

        return mean, std
    
    def data_augmentation(dataset, augmented_dir, val=5): 
        if not os.path.exists(augmented_dir):
            os.mkdir(augmented_dir)
            img_num = 0
            for _ in range(val):
                for img in tqdm(dataset):
                    save_image(img, augmented_dir+"/"+'aug'+str(img_num)+'.png')
                    img_num += 1 

# my_transforms = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
#     transforms.RandomVerticalFlip(p=0.4),
#     transforms.RandomPosterize(5, p=0.5),
#     transforms.RandomSolarize(100,p=0.5),
#     transforms.RandomRotation(30),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.1674, 0.1686, 0.1694], 
#                          std=[0.2474, 0.2511, 0.2495])
#     ])


# show_images = PokeData(csv_file=csv_file,
#                             root_dir=root_dir)

# pokemon_dataset = PokeData(csv_file=csv_file,
#                             root_dir=root_dir,
#                             transform=my_transforms)

# pokemon_dataloader = DataLoader(pokemon_dataset,
#                                 batch_size = 64, shuffle=True)


#show_imgs = PokeData.show_img(show_images)
# mean, std = PokeData.get_mean_std(pokemon_dataloader)
# print(mean)
# print(std)

# augmented_data_batchs = "archive/aug_batches"
# aug_data = PokeData.data_augmentation(pokemon_dataset, aug_dir, 13)
# print(aug_data)