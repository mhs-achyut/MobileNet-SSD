# =============================================================================
# Copyright (c) 2001-2019 FLIR Systems, Inc. All Rights Reserved.
#
# This software is the confidential and proprietary information of FLIR
# Integrated Imaging Solutions, Inc. ("Confidential Information"). You
# shall not disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with FLIR Integrated Imaging Solutions, Inc. (FLIR).
#
# FLIR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
# SOFTWARE, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, OR NON-INFRINGEMENT. FLIR SHALL NOT BE LIABLE FOR ANY DAMAGES
# SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
# THIS SOFTWARE OR ITS DERIVATIVES.
# =============================================================================

# Deserialize, modify, and serialize back a caffe model

import os
import argparse
import caffe_pb2
import google.protobuf.text_format as pb_text
from shutil import copyfile

# DEBUG:  Debug output info
# TODO: Remove references to this global after testing is done
out_str = ''
output_folder = 'output'
logging = False


# Try converting string to a number that can be a used as container index.
# Usable index can't be negative
def str2index(txt):
    try:
        r = int(txt)
        return r if r >= 0 else -1
    except ValueError:
        return -1
    else:
        return -1


def confirm_output_dir_exists():
    global output_folder
    if not os.path.exists(os.path.join(os.getcwd(), output_folder)):
        print('Creating output dir: ' + os.path.join(os.getcwd(), output_folder))
        os.mkdir(output_folder)
    return


def update_proto_file(file_name_no_ext, network):
    global output_folder
    with open(output_folder + '/' + 'optimized.prototxt', 'w') as f:
        f.write(pb_text.MessageToString(network))


def update_model_file(file_name_no_ext, network):
    global output_folder
    with open(output_folder + '/' + 'optimized.caffemodel', 'wb') as f:
        f.write(network.SerializeToString())


def get_file_name_no_extension(file_full_path):
        _, file_name = os.path.split(file_full_path)
        file_name, extension = os.path.splitext(file_name)
        return file_name, extension


# TODO: there can be more special cases?
def is_special_case(top_name):
    if top_name == 'label':
        return True
    if top_name == 'data':
        return True
    return False


def ignore_tail(layer_info, name_tail):
    if name_tail.lower() == layer_info.type.lower():
        return True
    if layer_info.type.lower() == 'batchnorm' and name_tail == 'bn': # TODO: there can be more special cases?:
        return True
    return False


def analyze_layer(layer_info, layer_idx):
    global out_str
    # Check if the tail of the layer name follows pattern layer_name/layer_type.
    # In this case the tail part after '/' is ignored in the top and bottom names.
    # TODO: (!!!) Check if '_' separator is ever used for appending layer type
    name_base, _, name_tail = layer_info.name.rpartition('/')
    is_different, old_name, new_name = (False, '', '')

    return_val = []

    if not ignore_tail(layer_info, name_tail) :
        # use full name in comparison
        name_base = layer_info.name
        name_tail = ''

    for top_idx, top in enumerate(layer_info.top):
        if is_special_case(top):
            continue

        # Skipp unneccessary rename, example: when 'ReLu' layer has conv1 for both top and bottom
        if top_idx == 0 and top == layer_info.bottom[top_idx]:
            continue

        # if there is more than one output  layer, the top_name can have its top_index appended to the top_name, and needs to be ignored in comparison
        top_base, _, top_tail = top.rpartition('_')
        if str2index(top_tail) != top_idx:
            # use full top name
            top_base = top
            top_tail = ''
        # Always compare the top name without the index appendd
        if layer_info.name.lower() == top_base.lower() or name_base.lower() == top_base.lower():
            continue
        else:
            # rename is needed
            old_name = top_base
            new_name = name_base + top_tail
            return_val.append((old_name, new_name))
            # DEBUG:
            out_str = out_str +  str(layer_idx) + ' Renamed top ' + old_name + ' to match layer name '+ new_name + '\n'

            return_val.append((old_name, new_name))
            layer_info.top[top_idx] = new_name

    return return_val # [(old_name, new_name)]


def inspect_network_layers(network):
    # DEBUG:
    global out_str
    renamed_nodes_map = {}

    for idx, layer in enumerate(network.layer):
        ret_val = analyze_layer(layer, idx)
        for (old_name, new_name) in ret_val:
            #DEBUG:
            if new_name == '' or new_name == old_name:
                print('Something is wrong with ' + idx + layer.name)
                continue
            renamed_nodes_map[old_name] = new_name

    if len(renamed_nodes_map) > 0:
        # If a 'top' for some layer was renamed, rename all references to it in 'bottom'
        for _, layer in enumerate(network.layer):
            for idx, bot in enumerate(layer.bottom):
                if bot in renamed_nodes_map:
                    # DEBUG:
                    out_str = out_str + 'Renamed bottom ' + layer.bottom[idx] + ' to ' + renamed_nodes_map[bot] + '\n'

                    layer.bottom[idx] = renamed_nodes_map[bot]

        return True

    return False


def inspect_model_file( model_file):
    # DEBUG:
    global out_str
    global output_folder
    out_str = ''
    network = caffe_pb2.NetParameter()

    with open(model_file, 'rb') as f:
        network.ParseFromString(f.read())

    file_name, ext = get_file_name_no_extension(model_file)

    if inspect_network_layers(network):
        print('updating model file ' + file_name)
        update_model_file( file_name , network)
        if logging:
            # DEBUG: Writes some debug info about layers there
            result_file = open(output_folder + '/' + file_name + '_model.log', 'w')
            result_file.write(out_str)
    else:
        # DEBUG: Writes some debug info about layers there
        print('nothing to update in ' + file_name)
        copyfile(model_file, output_folder + '/' + 'optimized.caffemodel')

    return


def inspect_proto_file( proto_file):
    # DEBUG:
    global out_str
    global output_folder
    out_str = ''
    network = caffe_pb2.NetParameter()

    with open(proto_file, 'r') as f:
        pb_text.Merge(f.read(), network)

    file_name, ext = get_file_name_no_extension(proto_file)

    if inspect_network_layers(network):
        print('updating proto file ' + file_name)
        update_proto_file( file_name , network)
        if logging:
            # DEBUG: Writes some debug info about layers there
            result_file = open(output_folder + '/' + file_name + '_prototxt.log', 'w')
            result_file.write(out_str)
    else:
        print('nothing to update in ' + file_name)
        copyfile(proto_file, output_folder + '/' + 'optimized.prototxt')

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Deserialize, modify, and serialize back a Caffe Model or Prototxt file so that the output can be '
                    'passed to mvNCCompile'
    )
    parser.add_argument('files', metavar='file', type=str, nargs='+',
                        help='file path of Caffe Model or Prototxt to be updated')
    parser.add_argument('-output',
                        default='output',
                        help='output folder')
    parser.add_argument('-logging', dest='logging', action='store_true')
    parser.add_argument('-no-logging', dest='logging', action='store_false')
    parser.set_defaults(logging=False)
    args = parser.parse_args()

    input_files = args.files
    output_folder = args.output
    logging = args.logging


    model_files = []
    prototxt_files = []

    for input_file in input_files:
        if input_file.lower().endswith('.prototxt'):
            prototxt_files.append(input_file)
        elif input_file.lower().endswith('.caffemodel'):
            model_files.append(input_file)
        else:
            print(input_file + ' - [Unknown file extension]')

    confirm_output_dir_exists()

    for model_file in model_files:
        inspect_model_file(model_file)

    for proto_file in prototxt_files:
        inspect_proto_file(proto_file)

    print('\nThe End')

