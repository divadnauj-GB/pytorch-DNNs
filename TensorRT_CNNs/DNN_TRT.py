import os
import h5py
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse



def get_argparser():
    parser = argparse.ArgumentParser(description='DNN models')
    parser.add_argument('-t','--type', required=True, type=str, help='golden')
    parser.add_argument('-n','--model_name', required=True, type=str, help='golden')
    parser.add_argument('-ln','--layer_number', required=False, type=int, help='golden')
    parser.add_argument('-bs','--batch_size', required=True, type=int, help='golden')
    parser.add_argument('-sz','--shape',required=False, nargs='+', type=int ,help="shape of the output layer")
    parser.add_argument('-onnx','--onnx', required=False, action='store_true', help='golden')
    parser.add_argument('-trt','--run_trt', required=False, action='store_true', help='golden')
    return parser

DEBUG = 0

def main(args):

    path = os.path.dirname(__file__)
    # os.environ["CUDA_VISIBLE_DEVICES"]=""
    
    target_dtype = np.float32
    current_path = os.path.dirname(__file__)
    path_dir = f"{args.type}/{args.model_name}"
    path_dir = os.path.join(current_path,path_dir)
    #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    if args.shape:
        TRT_output_shape = tuple(args.shape)

    layer_results = []

    TRT_model_name = f"{args.model_name}_pytorch.rtr"

    dataset_file = os.path.join(
            path_dir,
            f"Inputs_DNN.h5",
        )
    with h5py.File(dataset_file, "r") as hf:
        Input_dataset = np.array(hf["inputs"])


    with open(os.path.join(path_dir, TRT_model_name), "rb") as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        output = np.empty(TRT_output_shape, dtype = target_dtype) 
        if DEBUG: print(Input_dataset.shape)
        if DEBUG: print(TRT_output_shape)
        if DEBUG: print(output.shape)
        batch = 0
        sample_image = Input_dataset[
                    batch * batch_size : batch * batch_size + batch_size
                ]
        d_input = cuda.mem_alloc(1 * sample_image.nbytes)
        d_output = cuda.mem_alloc(1 * output.nbytes)
        bindings = [int(d_input), int(d_output)]
        stream = cuda.Stream()
        # warming up
        # output=self.__TRT_forward_function(sample_image)
        # end warming up

    max_batches = float(float(len(Input_dataset)) / float(batch_size))
    #with torch.no_grad():
    for batch in range(0, int(np.ceil(max_batches))):
        img = Input_dataset[
            batch * batch_size : batch * batch_size + batch_size
        ]

        cuda.memcpy_htod_async(d_input, img, stream)
        # execute model
        context.execute_async_v2(bindings, stream.handle, None)
        # transfer predictions back
        cuda.memcpy_dtoh_async(output, d_output, stream)
        # syncronize threads
        stream.synchronize()

        layer_results.append(output)
        if DEBUG: print(output)

    embeddings_outputs = np.concatenate(layer_results)

    if DEBUG: print(embeddings_outputs.shape)

    log_path_file = os.path.join(
        path_dir,
        f"Outputs_DNN.h5",
    )

    with h5py.File(log_path_file, "w") as hf:
        hf.create_dataset(
            "outputs", data=embeddings_outputs, compression="gzip"
        )

if __name__ == "__main__":
    argparser = get_argparser()
    main(argparser.parse_args())

