class KittiSemanticDataset(Dataset):
    def __init__(self, root='/kaggle/input/kitti-dataset', split='train', mode='semantic', transform=None, transform_train=None):
        self.transform = transform
        self.transform_train = transform_train

        assert split in ['train', 'test']
        self.split = 'training' if split == 'train' else 'testing'

        self.root = root  # Use the root directly, it already points to the dataset

        assert mode in ['semantic', 'color']
        self.mode = mode

        # Update paths for images and labels according to standard KITTI structure
        self.imagesPath = os.path.join(self.root, "data_object_image_2", self.split)  # Adjusted path for images
        self.semanticPath = os.path.join(self.root, "data_object_label_2", self.split)  # Adjusted path for labels

        # List all images / labels paths
        images_names = sorted(os.listdir(self.imagesPath))
        semantic_names = sorted(os.listdir(self.semanticPath))

        # Add the root path to images and labels names
        self.images_paths = [os.path.join(self.imagesPath, name) for name in images_names]
        self.semantic_paths = [os.path.join(self.semanticPath, name) for name in semantic_names]

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        semantic_path = self.semantic_paths[index]

        image = self.read_image(image_path)
        semantic = self.read_image(semantic_path) if self.mode == 'semantic' else None

        image = np.asarray(image)
        semantic = np.asarray(semantic) if semantic is not None else None

        # Resize images and labels
        shape = (1024, 512)
        image = cv2.resize(image, shape)
        if semantic is not None:
            semantic = cv2.resize(semantic, shape, interpolation=cv2.INTER_NEAREST)

        if self.transform_train:
            image_label = self.transform_train(dict(im=image, lb=semantic))
            image = image_label['im'].copy()
            semantic = image_label['lb'].copy()

        if self.transform:
            image = self.transform(image)

        return image, semantic

    def __len__(self):
        return len(self.images_paths)

    def read_image(self, path):
        return cv2.imread(path, cv2.IMREAD_COLOR)
