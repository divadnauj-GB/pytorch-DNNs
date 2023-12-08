import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.common_types as ct
import torchvision
from torchvision import models
import numpy as np
import copy
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import time
import os, sys
import argparse
import nvbitfi_DNN as nvbitDNN
import h5py



def get_argparser():
    parser = argparse.ArgumentParser(description='DNN models')
    parser.add_argument('-g','--golden', required=False, help='golden')
    parser.add_argument('-lt','--layer', required=False, help='golden')
    parser.add_argument('-ln','--layer_number', required=False, type=int, default=0, help='golden')
    parser.add_argument('-bs','--batch_size', required=False, type=int, default=1, help='golden')
    parser.add_argument('-w','--workers', required=False, type=int, default=4, help='golden')
    parser.add_argument('-ims','--num_images', required=False, type=int, default=4, help='golden')
    return parser


def main(args):

    path = os.path.dirname(__file__)
    # os.environ["CUDA_VISIBLE_DEVICES"]=""

    transform = transforms.Compose([            #[1]
        transforms.Resize(256),                    #[2]
        transforms.CenterCrop(224),                #[3]
        transforms.ToTensor(),                     #[4]
        transforms.Normalize(                      #[5]
        mean=[0.485, 0.456, 0.406],                #[6]
        std=[0.229, 0.224, 0.225]                  #[7]
        )])


    # Define relevant variables for the ML task
    batch_size = args.batch_size #512 
    num_classes = 10
    learning_rate = 0.001
    num_epochs = 10

    LeNet_dict = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
    }

    # Device will determine whether to run the training on GPU or CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    # device = 'cpu'
    # Loading the dataset and preprocessing
    train_dataset = torchvision.datasets.ImageFolder(
        root="~/dataset/ilsvrc2012/val",
        transform=transform
    )

    test_dataset = train_dataset = torchvision.datasets.ImageFolder(
        root="~/dataset/ilsvrc2012/val",
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.workers
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.workers
    )
    print(args.golden)

    if args.golden:
        # torch.backends.cudnn.allow_tf32=True
        # torch.backends.cudnn.enabled=False

        # print(torch.backends.cuda.matmul.allow_tf32)
        # print(torch.backends.cudnn.allow_tf32)
        model = models.resnet50(pretrained=True)

        model = model.to(device)
        model.eval()
        
        print(model)

        # Embeddings = nvbitDNN.extract_embeddings_nvbit(
        #     model=model, lyr_type=[nn.Conv2d], lyr_num=args.layer_number, batch_size=batch_size, path_dir=f"conv2d/AlexNet-ln{args.layer_number}"
        # )

        t = time.time()
        tot_imgs=0
        gacc1=0
        gacc5=0
        Inputs =[]
        Labels = []
        Output=[]
        dummy_input = None
        with torch.no_grad():
            for batch, (images, labels) in enumerate(test_loader):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if batch == 0: dummy_input = images
                Inputs.append(images.detach().cpu())
                Labels.append(labels.detach().cpu())
                # if(labels[0].item()==0):
                outputs = model(images)
                Output.append(outputs.detach().cpu())
                # sorted, indices=torch.sort(outputs.data)
                pred, clas=outputs.cpu().topk(5,1,True,True)
                clas = clas.t()
                Res = clas.eq(labels[None].cpu())
                acc1 = Res[:1].sum(dim=0,dtype=torch.float32)
                acc5 = Res[:5].sum(dim=0,dtype=torch.float32)
                tot_imgs+=batch_size
                gacc1 += Res[:1].flatten().sum(dtype=torch.float32)
                gacc5 += Res[:5].flatten().sum(dtype=torch.float32)
                if batch*batch_size+batch_size>=args.num_images:
                    break
            elapsed = time.time() - t
            print(
                "Accuracy of the network on the {} test images: acc1 {} % acc5 {} % in {} sec".format(
                    tot_imgs, 100 * gacc1 / tot_imgs,  100 * gacc5 / tot_imgs, elapsed
                )
            )

        currentPath = os.path.dirname(__file__)
        currentFileName = os.path.basename(__file__).split('.')[0]
        directory = os.path.join(currentPath,"DNNs",currentFileName)

        os.system(f"mkdir -p {directory}")

        embeddings_input = (
                torch.cat(Inputs).cpu().numpy()
            )
        
        embeddings_label = (
                torch.cat(Labels).cpu().numpy()
            )
               
        log_path_file = os.path.join(
                directory, f"Inputs_DNN.h5"
            )
        
        with h5py.File(log_path_file, "w") as hf:
            hf.create_dataset(
                "inputs", data=embeddings_input, compression="gzip"
            )
            hf.create_dataset(
                "labels", data=embeddings_label, compression="gzip"
            )

        embeddings_output = (torch.cat(Output).cpu().numpy())
        
        log_path_file = os.path.join(
                directory, f"Outputs_DNN.h5"
            )
        
        with h5py.File(log_path_file, "w") as hf:
            hf.create_dataset(
                "outputs", data=embeddings_output, compression="gzip"
            )


        onnx_model_name = os.path.join(directory,f"{currentFileName}_pytorch.onnx")

        torch.onnx.export(model, dummy_input, onnx_model_name, verbose=False)


if __name__ == "__main__":
    argparser = get_argparser()
    main(argparser.parse_args())

