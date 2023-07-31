import copy

import torch
import numpy as np
from torch._C import dtype
from torchvision import transforms
from torchvision.datasets import MNIST
import torch.nn.functional as F

from dataset import TMNIST
from model import CNets
from utils import get_labels_to_indices, get_mask, get_transformed_imgs_2
from config import ex

@ex.automain
def main(_config):

    _config = copy.deepcopy(_config)
    # init dataset
    mean = np.array([0.5, ])
    std = np.array([0.5, ])

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])
    dataset = TMNIST('data/', train=False, rotation=90., transform=transform)
    dataset_ref = MNIST('data/', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=1)


    # init models
    classifier = CNets(1,32,10)
    d_c_path = _config['classifier_path']
    d_c = torch.load(d_c_path, map_location=torch.device('cpu'))
    classifier.load_state_dict(d_c)

    estimator = CNets(2,32,1)
    d_r_path = _config['estimator_path']
    d_r = torch.load(d_r_path, map_location=torch.device('cpu'))
    estimator.load_state_dict(d_r)

    identifier = CNets(2,32,1)
    d_i_path = _config['identifier_path']
    d_i = torch.load(d_i_path, map_location=torch.device('cpu'))
    identifier.load_state_dict(d_i)

    num_references = [5, 10, 20, 50, 100, 200, 500]
    for n in num_references:
        evaluator = Evaluator(dataset, dataset_ref, dataloader, classifier, estimator, identifier, num_reference=n)
        evaluator.evaluate()

        
class Evaluator(object):
    def __init__(self, dataset, dataset_ref, dataloader, classifier, regressor, identifier, num_reference=200):
        super().__init__()

        self.dataset = dataset
        self.dataloader = dataloader
        self.device = torch.device('cuda:0')

        self.classifier = classifier
        self.regressor = regressor.to(self.device)
        self.identifier = identifier.to(self.device)

        self.num_reference = num_reference #{num_reference} references per class
        self.list_references = torch.zeros(self.num_reference * 10).to(dtype=torch.long) #10 classes
        self.labels_to_indices = get_labels_to_indices(dataset_ref.targets)
        for i in range(10):
            self.list_references[i*self.num_reference:(i+1)*self.num_reference] = torch.from_numpy(np.random.choice(self.labels_to_indices[i], self.num_reference))
        self.ref_data = (dataset_ref.data[self.list_references]/255*2-1).unsqueeze(1)

        self.mask = get_mask((28,28)).to(self.device)
        self.metric_class_T = torch.zeros(self.dataset.data.size(0), 10) #softmax(model(x)) w/ rotation
        self.metric_reason_T = torch.zeros(self.dataset.data.size(0), 10) #  w/ rotation
        


    def evaluate(self):

        self.get_classification_result()
        
        for ndx in range(self.dataset.data.size(0)):
            img = self.dataset.__getitem__(ndx)[0] #size (2,28,28)
            self.metric_reason_T[ndx]  = self.reason(img[1:])
            print(ndx)


        path_mrt = f'results/metric_reason_T_{self.num_reference}.pt'

        torch.save(self.metric_class_T, 'results/metric_class_T.pt')
        torch.save(self.metric_reason_T, path_mrt)


    def get_classification_result(self):
        with torch.no_grad():
            self.classifier.eval()
            ndx = 0

            for img, target in self.dataloader:
                self.metric_class_T[ndx:ndx+target.size(0)] = F.softmax(self.classifier(img[:,1:,:,:]), dim=1)

                ndx = ndx + target.size(0)



    def reason(self, img):
        img_repeat = img.repeat(self.num_reference*10,1,1,1)
        data_r = torch.cat((self.ref_data, img_repeat), dim=1)
        data_r_g = data_r.to(self.device)

        with torch.no_grad():
            self.regressor.eval()
            self.identifier.eval()
            
            degrees_hat = self.regressor(data_r_g).to(torch.device('cpu'))
            img_hat = get_transformed_imgs_2(data_r.size(0), degrees_hat*90, data_r[:,0:1,:,:])
            data_i = torch.cat((img_repeat, img_hat), dim=1)
            data_i_g = data_i.to(self.device)
            data_i_g = data_i_g * self.mask

            output = self.identifier(data_i_g).squeeze(1)

            prediction, _ = torch.max(output.view(10,self.num_reference), dim=1) #find the max in each class

        return prediction



if __name__ == '__main__':
    main()