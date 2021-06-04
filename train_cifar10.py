import sys
from lib.ModelWrapper import ModelWrapper
from tensorboardX import SummaryWriter
import torch
from torchvision import transforms, datasets
from archs.cifar10 import vgg, resnet
import numpy as np
import random
import os
from nngeometry.generator.jacobian import Jacobian
from nngeometry.layercollection import LayerCollection
from nngeometry.object.vector import PVector
from torch.utils.data import TensorDataset, DataLoader
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

data_name = 'cifar10'
model_name = 'resnet'

# setting
lr = 1e-4
train_batch_size = 128
train_epoch = 1200
eval_batch_size = 256
# label_noise = 0.15
k = int(sys.argv[1])
label_noise = float(sys.argv[2])
# k = 64
num_classes = 1


dataset = datasets.CIFAR10
train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()])

eval_transform = transforms.Compose([transforms.ToTensor()])

if model_name == 'vgg16':
    model = vgg.vgg16_bn()
elif model_name == 'resnet':
    model = resnet.resnet18(k, num_classes)
else:
    raise Exception("No such model!")

# load data
train_data = dataset('D:/Datasets', train=True, transform=train_transform, download=True)
train_targets = np.array(train_data.targets)
data_size = len(train_targets)
random_index = random.sample(range(data_size), int(data_size*label_noise))
random_part = train_targets[random_index]
np.random.shuffle(random_part)
train_targets[random_index] = random_part
train_data.targets = train_targets.tolist()

noise_data = dataset('D:/Datasets', train=True, transform=train_transform, download=True)
noise_data.targets = random_part.tolist()
noise_data.data = train_data.data[random_index]


test_data = dataset('D:/Datasets', train=False, transform=eval_transform, download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=0,
                                           drop_last=False)
noise_loader = torch.utils.data.DataLoader(noise_data, batch_size=train_batch_size, shuffle=False, num_workers=0,
                                           drop_last=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=eval_batch_size, shuffle=False, num_workers=0,
                                          drop_last=False)

# build model
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model = model.to(device)




if True:
    # print('load')
    # id_epoch = ''
    # model.load_state_dict(torch.load('/home/pezeshki/scratch/dd/Deep-Double-Descent/runs2/cifar10/resnet_' + str(int(label_noise*100)) + '_k' + str(k) + '/ckpt' + str(id_epoch) + '.pkl')['net'])

    # flat_params = []
    # for p in model.parameters():
    #     flat_params += [p.view(-1)]
    # flat_params = torch.cat(flat_params)
    flat_params = PVector.from_model(model).get_flat_representation()
    sums = torch.zeros(*flat_params.shape).cuda()
    sums_sqr = torch.zeros(*flat_params.shape).cuda()

    model.eval()
    def output_fn(input, target):
        # input = input.to('cuda')
        return model(input)

    layer_collection = LayerCollection.from_model(model)
    layer_collection.numel()

    # loader = torch.utils.data.DataLoader(
    #     test_data, batch_size=150, shuffle=False, num_workers=0,
    #     drop_last=False)
    loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=0,
                                         drop_last=False)

    it = iter(loader)

    for X, y in tqdm(it):

        X = X.cuda()
        y = y.cuda()
        batch = TensorDataset(X, y)
        batch_loader = DataLoader(batch)
        generator = Jacobian(layer_collection=layer_collection,
                             model=model,
                             # loader=batch_loader,
                             function=output_fn,
                             n_output=1)
        jac = generator.get_jacobian(examples=batch_loader)[0]
        sums += jac.sum(0)
        sums_sqr = (jac ** 2).sum(0)

    std = torch.sqrt((sums_sqr / sums.shape[0]) - (sums / sums.shape[0]) ** 2)
    np.save('All_train4_'+str(k)+'_E0', std.data.cpu().numpy())

        # to_save = torch.cat((mean, sq_mean))
        # np.save('bin_' + str(k), to_save.data.cpu().numpy())
        # std = jac.std(0).data.cpu().numpy()[:, None]
        # flat_params = flat_params.data.cpu().numpy()[:, None]
        # np.save('bin_' + str(label_noise) + '_' + str(k) + '_E' + str(id_epoch),
        #         np.concatenate([std, flat_params], 1))

    # import pdb; pdb.set_trace()

    model.train()
    import os; os._exit(0)
    import pdb; pdb.set_trace()




criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
wrapper = ModelWrapper(model, optimizer, criterion, device)

# train the model
save_path = os.path.join('runs2', data_name, "{}_{}_k{}".format(model_name, int(label_noise*100), k))
if not os.path.exists(save_path):
    os.makedirs(save_path)
np.savez(os.path.join(save_path, "label_noise.npz"), index=random_index, value=random_part)
writer = SummaryWriter(logdir=os.path.join(save_path, "log"))

itr_index = 1
wrapper.train()

for id_epoch in range(train_epoch):
    # train loop

    for id_batch, (inputs, targets) in enumerate(train_loader):

        loss, acc, _ = wrapper.train_on_batch(inputs, targets)
        print("epoch:{}/{}, batch:{}/{}, loss={}, acc={}".
              format(id_epoch+1, train_epoch, id_batch+1, len(train_loader), loss, acc))
        if itr_index % 20 == 0:
            writer.add_scalar("train acc", acc, itr_index)
            writer.add_scalar("train loss", loss, itr_index)

        itr_index += 1

    wrapper.eval()
    test_loss, test_acc = wrapper.eval_all(test_loader)
    # noise_loss, noise_acc = wrapper.eval_all(noise_loader)
    print("epoch:{}/{}, batch:{}/{}, testing...".format(id_epoch + 1, train_epoch, id_batch + 1, len(train_loader)))
    print("clean: loss={}, acc={}".format(test_loss, test_acc))
    # print("noise: loss={}, acc={}".format(noise_loss, noise_acc))
    print()
    writer.add_scalar("test acc", test_acc, itr_index)
    writer.add_scalar("test loss", test_loss, itr_index)
    # writer.add_scalar("noise acc", noise_acc, itr_index)
    # writer.add_scalar("noise loss", noise_loss, itr_index)
    state = {
        'net': model.state_dict(),
        'optim': optimizer.state_dict(),
        'acc': test_acc,
        'epoch': id_epoch,
        'itr': itr_index
    }
    torch.save(state, os.path.join(save_path, "ckpt.pkl"))

    # if id_epoch in [0, 10, 25, 50, 100, 300]:
    #     torch.save(state, os.path.join(save_path, "ckpt" + str(id_epoch) + ".pkl"))

    writer.flush()
    # return to train state.
    wrapper.train()

writer.close()

