# -*- coding: utf-8 -*-
import inspect
import multiprocessing
import os
import traceback
from copy import deepcopy
from time import sleep
from typing import Tuple, Union, List, Optional

import warnings
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import SimpleITK as sitk

from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, save_json

import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.data_iterators import (
    PreprocessAdapterFromNpy,
    preprocessing_iterator_fromfiles,
    preprocessing_iterator_fromnpy,
)
from nnunetv2.inference.export_prediction import (
    export_prediction_from_logits,
    convert_predicted_logits_to_segmentation_with_correct_shape,
)
from nnunetv2.inference.sliding_window_prediction import (
    compute_gaussian,
    compute_steps_for_sliding_window,
)
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder


class nnUNetPredictor(object):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_gpu: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True):
        self.verbose = verbose
        self.verbose_preprocessing = verbose_preprocessing
        self.allow_tqdm = allow_tqdm

        self.plans_manager = None
        self.configuration_manager = None
        self.list_of_parameters = None
        self.network = None
        self.dataset_json = None
        self.trainer_name = None
        self.allowed_mirroring_axes = None
        self.label_manager = None

        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        if device.type != 'cuda':
            print('perform_everything_on_gpu=True is only supported for cuda devices! Setting this to False')
            perform_everything_on_gpu = False
        self.device = device
        self.perform_everything_on_gpu = perform_everything_on_gpu

        # Grad-CAM buffers
        self._cam_features = None
        self._cam_grads = None
        self._cam_hook_handle = None

    # ========== 初始化 / 原生推理 ==========
    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        if use_folds is None:
            use_folds = nnUNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'))
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint.get('inference_allowed_mirroring_axes', None)

            parameters.append(checkpoint['network_weights'])

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(
            join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
            trainer_name, 'nnunetv2.training.nnUNetTrainer'
        )
        network = trainer_class.build_network_architecture(plans_manager, dataset_json, configuration_manager,
                                                           num_input_channels, enable_deep_supervision=False)

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, OptimizedModule):
            print('compiling network')
            self.network = torch.compile(self.network)

    def manual_initialization(self, network: nn.Module, plans_manager: PlansManager,
                              configuration_manager: ConfigurationManager, parameters: Optional[List[dict]],
                              dataset_json: dict, trainer_name: str,
                              inference_allowed_mirroring_axes: Optional[Tuple[int, ...]]):
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)

        allow_compile = True
        allow_compile = allow_compile and ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't'))
        allow_compile = allow_compile and not isinstance(self.network, OptimizedModule)
        if isinstance(self.network, DistributedDataParallel):
            allow_compile = allow_compile and isinstance(self.network.module, OptimizedModule)
        if allow_compile:
            print('compiling network')
            self.network = torch.compile(self.network)

    @staticmethod
    def auto_detect_available_folds(model_training_output_dir, checkpoint_name):
        print('use_folds is None, attempting to auto detect available folds')
        fold_folders = subdirs(model_training_output_dir, prefix='fold_', join=False)
        fold_folders = [i for i in fold_folders if i != 'fold_all']
        fold_folders = [i for i in fold_folders if isfile(join(model_training_output_dir, i, checkpoint_name))]
        use_folds = [int(i.split('_')[-1]) for i in fold_folders]
        print(f'found the following folds: {use_folds}')
        return use_folds

    def _manage_input_and_output_lists(self, list_of_lists_or_source_folder: Union[str, List[List[str]]],
                                       output_folder_or_list_of_truncated_output_files: Union[None, str, List[str]],
                                       folder_with_segs_from_prev_stage: str = None,
                                       overwrite: bool = True,
                                       part_id: int = 0,
                                       num_parts: int = 1,
                                       save_probabilities: bool = False):
        if isinstance(list_of_lists_or_source_folder, str):
            list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(
                list_of_lists_or_source_folder, self.dataset_json['file_ending'])
        print(f'There are {len(list_of_lists_or_source_folder)} cases in the source folder')
        list_of_lists_or_source_folder = list_of_lists_or_source_folder[part_id::num_parts]
        caseids = [os.path.basename(i[0])[:-(len(self.dataset_json['file_ending']) + 5)] for i in
                   list_of_lists_or_source_folder]
        print(f'I am process {part_id} out of {num_parts} (max process ID is {num_parts - 1}, we start counting with 0!)')
        print(f'There are {len(caseids)} cases that I would like to predict')

        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_filename_truncated = [join(output_folder_or_list_of_truncated_output_files, i) for i in caseids]
        else:
            output_filename_truncated = output_folder_or_list_of_truncated_output_files

        seg_from_prev_stage_files = [join(folder_with_segs_from_prev_stage, i + self.dataset_json['file_ending']) if
                                     folder_with_segs_from_prev_stage is not None else None for i in caseids]
        if not overwrite and output_filename_truncated is not None:
            tmp = [isfile(i + self.dataset_json['file_ending']) for i in output_filename_truncated]
            if save_probabilities:
                tmp2 = [isfile(i + '.npz') for i in output_filename_truncated]
                tmp = [i and j for i, j in zip(tmp, tmp2)]
            not_existing_indices = [i for i, j in enumerate(tmp) if not j]

            output_filename_truncated = [output_filename_truncated[i] for i in not_existing_indices]
            list_of_lists_or_source_folder = [list_of_lists_or_source_folder[i] for i in not_existing_indices]
            seg_from_prev_stage_files = [seg_from_prev_stage_files[i] for i in not_existing_indices]
            print(f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. '
                  f'That\'s {len(not_existing_indices)} cases.')
        return list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files

    def predict_from_files(self,
                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           save_probabilities: bool = False,
                           overwrite: bool = True,
                           num_processes_preprocessing: int = default_num_processes,
                           num_processes_segmentation_export: int = default_num_processes,
                           folder_with_segs_from_prev_stage: str = None,
                           num_parts: int = 1,
                           part_id: int = 0):
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
        else:
            output_folder = None

        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(my_init_kwargs)
            recursive_fix_for_json_export(my_init_kwargs)
            maybe_mkdir_p(output_folder)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))
            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)

        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, \
                f'This configuration is cascaded. Provide previous stage preds via -prev_stage_predictions'

        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, part_id, num_parts,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return

        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(list_of_lists_or_source_folder,
                                                                                 seg_from_prev_stage_files,
                                                                                 output_filename_truncated,
                                                                                 num_processes_preprocessing)

        return self.predict_from_data_iterator(data_iterator, save_probabilities, num_processes_segmentation_export)

    def _internal_get_data_iterator_from_lists_of_filenames(self,
                                                            input_list_of_lists: List[List[str]],
                                                            seg_from_prev_stage_files: Union[List[str], None],
                                                            output_filenames_truncated: Union[List[str], None],
                                                            num_processes: int):
        return preprocessing_iterator_fromfiles(input_list_of_lists, seg_from_prev_stage_files,
                                                output_filenames_truncated, self.plans_manager, self.dataset_json,
                                                self.configuration_manager, num_processes, self.device.type == 'cuda',
                                                self.verbose_preprocessing)

    def get_data_iterator_from_raw_npy_data(self,
                                            image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                            segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None, np.ndarray, List[np.ndarray]],
                                            properties_or_list_of_properties: Union[dict, List[dict]],
                                            truncated_ofname: Union[str, List[str], None],
                                            num_processes: int = 3):
        list_of_images = [image_or_list_of_images] if not isinstance(image_or_list_of_images, list) else image_or_list_of_images
        if isinstance(segs_from_prev_stage_or_list_of_segs_from_prev_stage, np.ndarray):
            segs_from_prev_stage_or_list_of_segs_from_prev_stage = [segs_from_prev_stage_or_list_of_segs_from_prev_stage]
        if isinstance(truncated_ofname, str):
            truncated_ofname = [truncated_ofname]
        if isinstance(properties_or_list_of_properties, dict):
            properties_or_list_of_properties = [properties_or_list_of_properties]

        num_processes = min(num_processes, len(list_of_images))
        pp = preprocessing_iterator_fromnpy(
            list_of_images,
            segs_from_prev_stage_or_list_of_segs_from_prev_stage,
            properties_or_list_of_properties,
            truncated_ofname,
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_processes,
            self.device.type == 'cuda',
            self.verbose_preprocessing
        )
        return pp

    def predict_from_list_of_npy_arrays(self,
                                        image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                        segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None, np.ndarray, List[np.ndarray]],
                                        properties_or_list_of_properties: Union[dict, List[dict]],
                                        truncated_ofname: Union[str, List[str], None],
                                        num_processes: int = 3,
                                        save_probabilities: bool = False,
                                        num_processes_segmentation_export: int = default_num_processes):
        iterator = self.get_data_iterator_from_raw_npy_data(image_or_list_of_images,
                                                            segs_from_prev_stage_or_list_of_segs_from_prev_stage,
                                                            properties_or_list_of_properties,
                                                            truncated_ofname,
                                                            num_processes)
        return self.predict_from_data_iterator(iterator, save_probabilities, num_processes_segmentation_export)

    def predict_from_data_iterator(self,
                                   data_iterator,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = default_num_processes):
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            for preprocessed in data_iterator:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                if ofile is not None:
                    print(f'\nPredicting {os.path.basename(ofile)}:')
                else:
                    print(f'\nPredicting image of shape {data.shape}:')

                print(f'perform_everything_on_gpu: {self.perform_everything_on_gpu}')
                properties = preprocessed['data_properties']

                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                prediction = self.predict_logits_from_preprocessed_data(data).cpu()

                if ofile is not None:
                    print('sending off prediction to background worker for resampling and export')
                    r.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits,
                            ((prediction, properties, self.configuration_manager, self.plans_manager,
                              self.dataset_json, ofile, save_probabilities),)
                        )
                    )
                else:
                    print('sending off prediction to background worker for resampling')
                    r.append(
                        export_pool.starmap_async(
                            convert_predicted_logits_to_segmentation_with_correct_shape, (
                                (prediction, self.plans_manager,
                                 self.configuration_manager, self.label_manager,
                                 properties,
                                 save_probabilities),)
                        )
                    )
                if ofile is not None:
                    print(f'done with {os.path.basename(ofile)}')
                else:
                    print(f'\nDone with image of shape {data.shape}:')
            ret = [i.get()[0] for i in r]

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        compute_gaussian.cache_clear()
        empty_cache(self.device)
        return ret

    def predict_single_npy_array(self, input_image: np.ndarray, image_properties: dict,
                                 segmentation_previous_stage: np.ndarray = None,
                                 output_file_truncated: str = None,
                                 save_or_return_probabilities: bool = False):
        ppa = PreprocessAdapterFromNpy([input_image], [segmentation_previous_stage], [image_properties],
                                       [output_file_truncated],
                                       self.plans_manager, self.dataset_json, self.configuration_manager,
                                       num_threads_in_multithreaded=1, verbose=self.verbose)
        if self.verbose:
            print('preprocessing')
        dct = next(ppa)

        if self.verbose:
            print('predicting')
        predicted_logits = self.predict_logits_from_preprocessed_data(dct['data']).cpu()

        if self.verbose:
            print('resampling to original shape')
        if output_file_truncated is not None:
            export_prediction_from_logits(predicted_logits, dct['data_properties'], self.configuration_manager,
                                          self.plans_manager, self.dataset_json, output_file_truncated,
                                          save_or_return_probabilities)
        else:
            ret = convert_predicted_logits_to_segmentation_with_correct_shape(
                predicted_logits, self.plans_manager, self.configuration_manager, self.label_manager,
                dct['data_properties'], return_probabilities=save_or_return_probabilities)
            if save_or_return_probabilities:
                return ret[0], ret[1]
            else:
                return ret

    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        original_perform_everything_on_gpu = self.perform_everything_on_gpu
        with torch.no_grad():
            prediction = None
            if self.perform_everything_on_gpu:
                try:
                    for params in self.list_of_parameters:
                        if not isinstance(self.network, OptimizedModule):
                            self.network.load_state_dict(params)
                        else:
                            self.network._orig_mod.load_state_dict(params)

                        if prediction is None:
                            prediction = self.predict_sliding_window_return_logits(data)
                        else:
                            prediction += self.predict_sliding_window_return_logits(data)

                    if len(self.list_of_parameters) > 1:
                        prediction /= len(self.list_of_parameters)
                except RuntimeError:
                    print('Prediction with perform_everything_on_gpu=True failed due to insufficient GPU memory. '
                          'Falling back to perform_everything_on_gpu=False. Not a big deal, just slower...')
                    traceback.print_exc()
                    prediction = None
                    self.perform_everything_on_gpu = False

            if prediction is None:
                for params in self.list_of_parameters:
                    if not isinstance(self.network, OptimizedModule):
                        self.network.load_state_dict(params)
                    else:
                        self.network._orig_mod.load_state_dict(params)
                    if prediction is None:
                        prediction = self.predict_sliding_window_return_logits(data)
                    else:
                        prediction += self.predict_sliding_window_return_logits(data)
                if len(self.list_of_parameters) > 1:
                    prediction /= len(self.list_of_parameters)

            print('Prediction done, transferring to CPU if needed')
            prediction = prediction.to('cpu')
            self.perform_everything_on_gpu = original_perform_everything_on_gpu
        return prediction

    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...]):
        slicers = []
        if len(self.configuration_manager.patch_size) < len(image_size):
            assert len(self.configuration_manager.patch_size) == len(image_size) - 1, \
                'tile_size length must be either equal to image_size length or one shorter.'
            steps = compute_steps_for_sliding_window(image_size[1:], self.configuration_manager.patch_size, self.tile_step_size)
            if self.verbose:
                print(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size {image_size}')
            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(tuple([slice(None), d, *[slice(si, si + ti) for si, ti in zip((sx, sy), self.configuration_manager.patch_size)]]))
        else:
            steps = compute_steps_for_sliding_window(image_size, self.configuration_manager.patch_size, self.tile_step_size)
            if self.verbose:
                print(f'n_steps {np.prod([len(i) for i in steps])}, image size {image_size}')
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(tuple([slice(None), *[slice(si, si + ti) for si, ti in zip((sx, sy, sz), self.configuration_manager.patch_size)]]))
        return slicers

    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction = self.network(x)

        if mirror_axes is not None:
            if isinstance(mirror_axes, (list, tuple)):
                mirror_axes = tuple(int(a) for a in mirror_axes)
            else:
                mirror_axes = (int(mirror_axes),)

            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

            num_predictons = 2 ** len(mirror_axes)
            if 0 in mirror_axes:
                prediction += torch.flip(self.network(torch.flip(x, (2,))), (2,))
            if 1 in mirror_axes:
                prediction += torch.flip(self.network(torch.flip(x, (3,))), (3,))
            if 2 in mirror_axes:
                prediction += torch.flip(self.network(torch.flip(x, (4,))), (4,))
            if 0 in mirror_axes and 1 in mirror_axes:
                prediction += torch.flip(self.network(torch.flip(x, (2, 3))), (2, 3))
            if 0 in mirror_axes and 2 in mirror_axes:
                prediction += torch.flip(self.network(torch.flip(x, (2, 4))), (2, 4))
            if 1 in mirror_axes and 2 in mirror_axes:
                prediction += torch.flip(self.network(torch.flip(x, (3, 4))), (3, 4))
            if 0 in mirror_axes and 1 in mirror_axes and 2 in mirror_axes:
                prediction += torch.flip(self.network(torch.flip(x, (2, 3, 4))), (2, 3, 4))
            prediction /= num_predictons
        return prediction

    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) -> Union[np.ndarray, torch.Tensor]:
        assert isinstance(input_image, torch.Tensor)
        self.network = self.network.to(self.device)
        self.network.eval()

        empty_cache(self.device)

        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=(self.device.type == 'cuda')) if self.device.type == 'cuda' else dummy_context():
                assert input_image.ndim == 4, 'input_image must be (c, x, y, z)'

                data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                           'constant', {'value': 0}, True, None)
                slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

                results_device = self.device if self.perform_everything_on_gpu else torch.device('cpu')
                try:
                    data = data.to(self.device)
                    predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                                   dtype=torch.half, device=results_device)
                    n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)
                    if self.use_gaussian:
                        gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1./8,
                                                    value_scaling_factor=10, device=results_device)
                except RuntimeError:
                    results_device = torch.device('cpu')
                    data = data.to(results_device)
                    predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                                   dtype=torch.half, device=results_device)
                    n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)
                    if self.use_gaussian:
                        gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1./8,
                                                    value_scaling_factor=10, device=results_device)
                finally:
                    empty_cache(self.device)

                for sl in tqdm(slicers, disable=not self.allow_tqdm):
                    workon = data[sl][None].to(self.device, non_blocking=False)
                    prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)
                    predicted_logits[sl] += (prediction * gaussian if self.use_gaussian else prediction)
                    n_predictions[sl[1:]] += (gaussian if self.use_gaussian else 1)

                predicted_logits /= n_predictions

        empty_cache(self.device)
        return predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])]

    # ========== Grad-CAM 相关 ==========
    def _find_target_cam_layer(self):
        """优先选择最后一个 out_channels != num_classes 的 Conv3d，找不到就用最后一个 Conv3d。"""
        num_classes = self.label_manager.num_segmentation_heads
        convs = []
        for name, m in self.network.named_modules():
            if isinstance(m, nn.Conv3d):
                convs.append((name, m))
        if not convs:
            raise RuntimeError("No Conv3d layer for CAM.")
        for name, m in reversed(convs):
            if getattr(m, "out_channels", None) != num_classes:
                return name, m
        return convs[-1]

    def _register_cam_hooks(self, target_layer):
        self._cam_features = None
        self._cam_grads = None

        def fwd_hook(module, inp, out):
            self._cam_features = out

            def _grad_hook(grad):
                self._cam_grads = grad
            out.register_hook(_grad_hook)

        self._cam_hook_handle = target_layer.register_forward_hook(fwd_hook)

    def _remove_cam_hooks(self):
        try:
            if self._cam_hook_handle is not None:
                self._cam_hook_handle.remove()
        except Exception:
            pass
        self._cam_features = None
        self._cam_grads = None
        self._cam_hook_handle = None

    @torch.inference_mode(False)
    def _gradcam_single_patch(self, x, target_class=1, use_amp=True, cam_enable_tta=False):
        """
        针对一个 patch 计算 Grad-CAM（稳健版）：
        - 用 softmax 概率 + 高置信掩膜构造 score
        - 若为全 0，退化到 |grad|*activation
        """
        self.network.zero_grad(set_to_none=True)

        if cam_enable_tta:
            logits = self._internal_maybe_mirror_and_predict(x)
        else:
            with torch.autocast(self.device.type, enabled=(self.device.type == 'cuda' and use_amp)):
                logits = self.network(x)

        prob = torch.softmax(logits.float(), dim=1)[:, target_class]  # (B, X, Y, Z)
        mask = (prob > 0.5).float()
        score = (prob * mask).sum() / mask.sum() if mask.sum() > 0 else prob.mean()

        score.backward(retain_graph=False)

        feats = self._cam_features
        grads = self._cam_grads
        if feats is None or grads is None:
            raise RuntimeError("CAM hooks not triggered. Check target layer.")

        alphas = grads.mean(dim=(2, 3, 4), keepdim=True)      # (1,F,1,1,1)
        cam = (alphas * feats).sum(dim=1, keepdim=True)       # (1,1,fx,fy,fz)
        cam = F.relu(cam)

        cam_up = F.interpolate(cam, size=logits.shape[2:], mode='trilinear', align_corners=False)[0, 0]
        cam_up = cam_up - cam_up.min()
        maxv = cam_up.max()

        if not torch.isfinite(maxv) or maxv <= 0:
            cam_alt = (grads.abs() * feats).sum(dim=1, keepdim=True)
            cam_alt = F.relu(cam_alt)
            cam_up = F.interpolate(cam_alt, size=logits.shape[2:], mode='trilinear', align_corners=False)[0, 0]
            cam_up = cam_up - cam_up.min()
            maxv = cam_up.max()

        if torch.isfinite(maxv) and maxv > 0:
            cam_up = (cam_up / (maxv + 1e-8)).float()
        else:
            cam_up = torch.zeros_like(cam_up, dtype=torch.float32)

        self.network.zero_grad(set_to_none=True)
        del logits, prob, mask, feats, grads, cam, alphas
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return cam_up

    # ---------- SITK 工具 ----------
    @staticmethod
    def _resample_like(img_to_resample: sitk.Image, ref: sitk.Image, is_label: bool) -> sitk.Image:
        """
        将 img_to_resample 重采样到 ref 的几何（origin/spacing/direction/size）。
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0)
        return resampler.Execute(img_to_resample)

    def _write_nifti_like(self, ref_nifti_path: str, vol_xyz: np.ndarray, out_path: str):
        """
        将 numpy 体数据写为 NIfTI，并复制 ref 的几何信息。
        输入 vol_xyz 需为 (Z, Y, X)！
        """
        ref = sitk.ReadImage(ref_nifti_path)
        img = sitk.GetImageFromArray(vol_xyz)
        img.CopyInformation(ref)
        sitk.WriteImage(img, out_path, True)

    # ---------- “贴表面增强” ----------
    def _enhance_cam_to_surface(self,
                                cam_path: str,
                                seg_path: str,
                                out_inseg_path: str,
                                out_shell_path: str,
                                pool_k: int = 5,
                                shell_mm: float = 1.0,
                                pct_clip: Tuple[float, float] = (2.0, 98.0)) -> None:
        """
        让 CAM 更贴近表面，便于模型 Probe 后直接看到高值。
        步骤：
          1) CAM 与分割重采样到同一几何（以 CAM 为基准）
          2) 在分割内部做 3D max-pool 扩散（窗口 pool_k）
          3) 在分割内按百分位归一化（提升可视化对比度）
          4) 另导出“表面带”体积（<= shell_mm）
        """
        cam_img = sitk.ReadImage(cam_path)
        seg_img = sitk.ReadImage(seg_path)

        # 重采样分割到 CAM 几何
        seg_on_cam = self._resample_like(seg_img, cam_img, is_label=True)

        cam_arr = sitk.GetArrayFromImage(cam_img).astype(np.float32)   # (Z,Y,X)
        seg_arr = (sitk.GetArrayFromImage(seg_on_cam) > 0).astype(np.uint8)

        # 1) 分割内的 max-pool 扩散
        cam_t = torch.from_numpy(cam_arr).unsqueeze(0).unsqueeze(0)    # (1,1,Z,Y,X)
        mask_t = torch.from_numpy(seg_arr).unsqueeze(0).unsqueeze(0).float()

        neg = torch.full_like(cam_t, -1e6)
        cam_neg = torch.where(mask_t > 0, cam_t, neg)
        pad = pool_k // 2
        pooled = F.max_pool3d(cam_neg, kernel_size=pool_k, stride=1, padding=pad)
        pooled = torch.where(mask_t > 0, pooled, torch.zeros_like(pooled))

        enhanced = torch.maximum(cam_t, pooled).squeeze(0).squeeze(0).numpy()

        # 2) 仅在分割内做百分位缩放
        inside_vals = enhanced[seg_arr > 0]
        if inside_vals.size > 0:
            lo = np.percentile(inside_vals, pct_clip[0])
            hi = np.percentile(inside_vals, pct_clip[1])
            denom = max(hi - lo, 1e-6)
            enhanced_norm = np.clip((enhanced - lo) / denom, 0.0, 1.0)
        else:
            enhanced_norm = np.zeros_like(enhanced, dtype=np.float32)

        enhanced_inseg = enhanced_norm * seg_arr

        # 3) 写 *_inSeg
        self._write_nifti_like(cam_path, enhanced_inseg.astype(np.float32), out_inseg_path)

        # 4) 构造“表面带”（<= shell_mm，单位 mm）
        seg_bin_img = sitk.Cast(seg_on_cam, sitk.sitkUInt8)
        dist_map = sitk.SignedMaurerDistanceMap(seg_bin_img,
                                                insideIsPositive=True,
                                                squaredDistance=False,
                                                useImageSpacing=True)
        dist_arr = sitk.GetArrayFromImage(dist_map)  # (Z,Y,X), mm
        shell_mask = ((dist_arr >= 0) & (dist_arr <= float(shell_mm))).astype(np.uint8)

        shell_vals = enhanced_norm * shell_mask
        nz = shell_vals[shell_vals > 0]
        if nz.size > 0:
            p98 = np.percentile(nz, 98.0)
            shell_vis = np.clip(shell_vals / max(p98, 1e-6), 0.0, 1.0)
        else:
            shell_vis = shell_vals

        self._write_nifti_like(cam_path, shell_vis.astype(np.float32), out_shell_path)

        print(f"[Enhance] Saved inSeg -> {out_inseg_path}")
        print(f"[Enhance] Saved shell  -> {out_shell_path}")
        print(f"[Enhance] Stats: CAM[min,max]={float(cam_arr.min()):.4f},{float(cam_arr.max()):.4f} | "
              f"inSeg[min,max]={float(enhanced_inseg.min()):.4f},{float(enhanced_inseg.max()):.4f} | "
              f"shell[min,max]={float(shell_vis.min()):.4f},{float(shell_vis.max()):.4f}")

    # ========== 统一流程：分割 + CAM + 贴表面增强 ==========
    def predict_seg_and_cam_from_files(self,
                                       list_of_lists_or_source_folder: Union[str, List[List[str]]],
                                       output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                                       target_class: int = 1,
                                       overwrite: bool = True,
                                       num_processes_preprocessing: int = default_num_processes,
                                       num_processes_segmentation_export: int = default_num_processes,
                                       cam_overwrite: bool = True,
                                       cam_enable_tta: bool = False,
                                       cam_use_amp: bool = True,
                                       pool_k: int = 5,
                                       shell_mm: float = 1.0):
        """
        导出分割(.nii.gz) + CAM(.nii.gz) + 贴表面增强版（*_inSeg、*_shell）
        """
        # 1) 先做分割（原流程）
        self.predict_from_files(list_of_lists_or_source_folder,
                                output_folder_or_list_of_truncated_output_files,
                                save_probabilities=False,
                                overwrite=overwrite,
                                num_processes_preprocessing=num_processes_preprocessing,
                                num_processes_segmentation_export=num_processes_segmentation_export,
                                folder_with_segs_from_prev_stage=None,
                                num_parts=1, part_id=0)

        # 2) 注册 CAM hook
        self.network.eval()
        name, layer = self._find_target_cam_layer()
        if self.verbose:
            print(f"[Grad-CAM] Using layer: {name}")
        self._register_cam_hooks(layer)

        try:
            lst, ofiles, prevstage = self._manage_input_and_output_lists(
                list_of_lists_or_source_folder,
                output_folder_or_list_of_truncated_output_files,
                overwrite=True,
                part_id=0, num_parts=1,
                save_probabilities=False
            )
            if len(lst) == 0:
                return

            data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(
                lst, prevstage, ofiles, num_processes_preprocessing
            )

            for preprocessed in data_iterator:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(delfile))
                    os.remove(delfile)

                props = preprocessed['data_properties']
                ofile = preprocessed['ofile']
                assert ofile is not None, "Need output filename to save CAM."

                # 最终输出路径
                final_cam_path = ofile + f"_CAM_class{target_class}.nii.gz"
                out_inseg_path = ofile + f"_CAM_class{target_class}_inSeg.nii.gz"
                out_shell_path = ofile + f"_CAM_class{target_class}_shell{shell_mm:.1f}mm.nii.gz"

                if (not cam_overwrite) and isfile(final_cam_path) and isfile(out_inseg_path) and isfile(out_shell_path):
                    print(f"[Grad-CAM] Skip existing CAM & enhanced: {final_cam_path}")
                    continue

                # A) 生成原始 CAM (重采样到原图几何)
                data, slicer_revert_padding = pad_nd_image(
                    data, self.configuration_manager.patch_size, 'constant', {'value': 0}, True, None)
                slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

                results_device = torch.device('cpu')
                data = data.to(self.device)
                cam_vol = torch.zeros(data.shape[1:], dtype=torch.float32, device=results_device)
                n_pred = torch.zeros(data.shape[1:], dtype=torch.float32, device=results_device)
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size),
                                            sigma_scale=1./8, value_scaling_factor=10, device=results_device)

                with torch.set_grad_enabled(True):
                    self.network = self.network.to(self.device)
                    self.network.eval()
                    for sl in tqdm(slicers, disable=not self.allow_tqdm):
                        work = data[sl][None].to(self.device, non_blocking=False)
                        cam_patch = self._gradcam_single_patch(
                            work, target_class=target_class, use_amp=cam_use_amp, cam_enable_tta=cam_enable_tta
                        )
                        cam_patch_cpu = cam_patch.detach().cpu()
                        cam_vol[sl[1:]] += cam_patch_cpu * gaussian
                        n_pred[sl[1:]] += gaussian

                        del work, cam_patch, cam_patch_cpu
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()

                cam_vol /= torch.clamp_min(n_pred, 1e-8)
                cam_vol = cam_vol[tuple(slicer_revert_padding[1:])]
                cam_np = cam_vol.detach().cpu().numpy().astype(np.float32)

                # 把 CAM 当“二类概率”的 logit 导出到原图尺寸
                eps = 1e-6
                cam_sig = np.clip(cam_np, eps, 1 - eps)
                logit = np.log(cam_sig / (1 - cam_sig))
                fake_logits = np.stack([np.zeros_like(logit, dtype=np.float32), logit.astype(np.float32)], axis=0)
                fake_logits_t = torch.from_numpy(fake_logits)

                temp_ofile = ofile + f"_camproxy_class{target_class}"
                export_prediction_from_logits(
                    fake_logits_t, props, self.configuration_manager, self.plans_manager,
                    self.dataset_json, temp_ofile, save_probabilities=True
                )
                npz_path = temp_ofile + ".npz"
                prob = np.load(npz_path)["probabilities"]  # (2, X0, Y0, Z0)
                cam_resampled = prob[1].astype(np.float32)
                ref_nii = temp_ofile + ".nii.gz"
                self._write_nifti_like(ref_nii, cam_resampled, final_cam_path)
                print(f"[Grad-CAM] Saved -> {final_cam_path}")

                # B) 贴表面增强（使用分割）
                seg_path = ofile + self.dataset_json['file_ending']  # 分割文件（与原图同几何）
                try:
                    self._enhance_cam_to_surface(final_cam_path, seg_path,
                                                 out_inseg_path, out_shell_path,
                                                 pool_k=pool_k, shell_mm=shell_mm, pct_clip=(2.0, 98.0))
                except Exception as e:
                    print(f"[Enhance][WARN] Failed to enhance CAM near surface: {e}")

                # 删除中间文件
                try:
                    os.remove(ref_nii)
                    os.remove(npz_path)
                    pkl_path = temp_ofile + ".pkl"
                    if os.path.isfile(pkl_path):
                        os.remove(pkl_path)
                except Exception:
                    pass

            if isinstance(data_iterator, MultiThreadedAugmenter):
                data_iterator._finish()
        finally:
            self._remove_cam_hooks()

        empty_cache(self.device)


# ========== 命令行入口 ==========
def predict_entry_point_modelfolder():
    import argparse
    parser = argparse.ArgumentParser(description='nnU-Net inference with manual model folder.')
    parser.add_argument('-i', type=str, required=True, help='input folder.')
    parser.add_argument('-o', type=str, required=True, help='output folder.')
    parser.add_argument('-m', type=str, required=True, help='trained model folder with fold_X subfolders.')
    parser.add_argument('-f', nargs='+', type=str, default=(0, 1, 2, 3, 4), help='folds to use')
    parser.add_argument('-step_size', type=float, default=0.5)
    parser.add_argument('--disable_tta', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save_probabilities', action='store_true')
    parser.add_argument('--continue_prediction', '--c', action='store_true')
    parser.add_argument('-chk', type=str, default='checkpoint_final.pth')
    parser.add_argument('-npp', type=int, default=3)
    parser.add_argument('-nps', type=int, default=3)
    parser.add_argument('-prev_stage_predictions', type=str, default=None)
    parser.add_argument('-device', type=str, default='cuda')

    print(
        "\n#######################################################################\nPlease cite the following paper when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    assert args.device in ['cpu', 'cuda', 'mps']
    if args.device == 'cpu':
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    predictor = nnUNetPredictor(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_gpu=True,
                                device=device,
                                verbose=args.verbose)
    predictor.initialize_from_trained_model_folder(args.m, args.f, args.chk)

    predictor.predict_seg_and_cam_from_files(
        args.i, args.o,
        target_class=1,
        overwrite=not args.continue_prediction,
        num_processes_preprocessing=args.npp,
        num_processes_segmentation_export=args.nps,
        cam_overwrite=True,
        cam_enable_tta=False,
        cam_use_amp=True,
        pool_k=5,
        shell_mm=1.0
    )


def predict_entry_point():
    import argparse
    parser = argparse.ArgumentParser(description='nnU-Net inference when nnUNet_results is set.')
    parser.add_argument('-i', type=str, required=True, help='input folder.')
    parser.add_argument('-o', type=str, required=True, help='output folder.')
    parser.add_argument('-d', type=str, required=True, help='dataset name or id.')
    parser.add_argument('-p', type=str, default='nnUNetPlans')
    parser.add_argument('-tr', type=str, default='nnUNetTrainer')
    parser.add_argument('-c', type=str, required=True, help='configuration')
    parser.add_argument('-f', nargs='+', type=str, default=(0, 1, 2, 3, 4))
    parser.add_argument('-step_size', type=float, default=0.5)
    parser.add_argument('--disable_tta', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save_probabilities', action='store_true')
    parser.add_argument('--continue_prediction', action='store_true')
    parser.add_argument('-chk', type=str, default='checkpoint_final.pth')
    parser.add_argument('-npp', type=int, default=3)
    parser.add_argument('-nps', type=int, default=3)
    parser.add_argument('-prev_stage_predictions', type=str, default=None)
    parser.add_argument('-num_parts', type=int, default=1)
    parser.add_argument('-part_id', type=int, default=0)
    parser.add_argument('-device', type=str, default='cuda')

    print(
        "\n#######################################################################\nPlease cite the following paper when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    assert args.part_id < args.num_parts, 'See nnUNetv2_predict -h.'

    assert args.device in ['cpu', 'cuda', 'mps']
    if args.device == 'cpu':
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    predictor = nnUNetPredictor(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_gpu=True,
                                device=device,
                                verbose=args.verbose,
                                verbose_preprocessing=False)
    predictor.initialize_from_trained_model_folder(model_folder, args.f, checkpoint_name=args.chk)

    predictor.predict_seg_and_cam_from_files(
        args.i, args.o,
        target_class=1,
        overwrite=not args.continue_prediction,
        num_processes_preprocessing=args.npp,
        num_processes_segmentation_export=args.nps,
        cam_overwrite=True,
        cam_enable_tta=False,
        cam_use_amp=True,
        pool_k=5,
        shell_mm=1.0
    )


if __name__ == '__main__':
    # 示例：直接调用（按你的路径修改）
    from nnunetv2.paths import nnUNet_results, nnUNet_raw
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset012_Airwayfull/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres'),
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth',
    )
    predictor.predict_seg_and_cam_from_files(
        join(nnUNet_raw, 'Dataset012_Airwayfull/imagesTs-test'),
        join(nnUNet_raw, 'Dataset012_Airwayfull/imagesTs_predlowres'),
        target_class=1,
        overwrite=False,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        cam_overwrite=True,
        cam_enable_tta=False,
        cam_use_amp=True,
        pool_k=5,           # 3/5/7，越大扩散越强
        shell_mm=1.0        # “表面带”厚度（mm）
    )
