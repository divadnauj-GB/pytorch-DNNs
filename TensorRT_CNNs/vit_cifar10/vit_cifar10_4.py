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
import h5py
import nvbitfi_DNN as nvbitDNN



def get_argparser():
    parser = argparse.ArgumentParser(description='DNN models')
    parser.add_argument('-g','--golden', required=False, help='golden')
    parser.add_argument('-t','--type', required=True, type=str, help='golden')
    parser.add_argument('-lt','--layer', required=False, help='golden')
    parser.add_argument('-ln','--layer_number', required=False, type=int, default=0, help='golden')
    parser.add_argument('-bs','--batch_size', required=False, type=int, default=1, help='golden')
    parser.add_argument('-w','--workers', required=False, type=int, default=4, help='golden')
    parser.add_argument('-ims','--num_images', required=False, type=int, default=4, help='golden')
    return parser


def main(args):

    path = os.path.dirname(__file__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    size = 32
    patch = 4
    dimhead = 512
    net = 'vit'

    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Prepare dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if args.golden:
        # torch.backends.cudnn.allow_tf32=True
        # torch.backends.cudnn.enabled=False

        # print(torch.backends.cuda.matmul.allow_tf32)
        # print(torch.backends.cudnn.allow_tf32)
        model = models.VisionTransformer(
        image_size = size,
        patch_size = patch,
        num_layers=6,
        num_heads = 8,        
        hidden_dim= int(dimhead),
        mlp_dim= int(dimhead),
        dropout=0.1,
        attention_dropout=0.1,
        num_classes = 10,
        )

        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/'+net+'-{}-ckpt.t7'.format(patch))
        model.load_state_dict(checkpoint['model'])

        model = model.to(device)
        model.eval()
        
        #print(model)

        #Embeddings = nvbitDNN.extract_embeddings_nvbit(
        #    model=model, lyr_type=[nn.Conv2d], lyr_num=args.layer_number, batch_size=batch_size
        #)

        t = time.time()
        tot_imgs=0
        gacc1=0
        gacc5=0
        Inputs =[]
        Labels = []
        Output = []
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
        #Embeddings.extract_embeddings_target_layer()
                #Embeddings.extract_embeddings_target_layer()
        currentPath = os.path.dirname(__file__)
        currentFileName = os.path.basename(__file__).split('.')[0].split('_')[0]
        directory = os.path.join(currentPath,args.type,currentFileName)

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
        #onnx_model_name = "vit_cifar10_16.onnx"
        TRT_model_name = os.path.join(directory,f"{currentFileName}_pytorch.rtr")#"vit_cifar10_16.rtr"

        torch.onnx.export(model, dummy_input, onnx_model_name, verbose=False)

        
        

        USE_FP16 = True
        target_dtype = np.float16 if USE_FP16 else np.float32
        if USE_FP16:
            cmd=f"/usr/src/tensorrt/bin/trtexec --onnx={onnx_model_name} --saveEngine={TRT_model_name}  --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16"
        else:
            cmd=f"/usr/src/tensorrt/bin/trtexec --onnx={onnx_model_name} --saveEngine={TRT_model_name}  --explicitBatch"
        os.system(cmd)

    else:
        Target_layer = nvbitDNN.load_embeddings(1, args.batch_size)
        Target_layer.layer_inference()


if __name__ == "__main__":
    argparser = get_argparser()
    main(argparser.parse_args())

