""" 
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
"""

import os
import warnings
from collections import OrderedDict

from omegaconf import ListConfig

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
from PIL import Image
from transformers import PreTrainedTokenizerFast

from utils.model import DonutConfig, DonutModel, SwinEncoder
from utils.processor import DolphinProcessor


def try_rename_lagacy_weights(ckpt, output_path=""):
    if "state_dict" in ckpt.keys():
        ckpt = ckpt["state_dict"]
    if "module" in ckpt.keys():
        ckpt = ckpt["module"]
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith("model."):
            k = k[len("model.") :]
        if k.startswith("encoder"):
            new_ckpt["vpm" + k[len("encoder") :]] = v
        elif k.startswith("decoder"):
            new_ckpt["llm" + k[len("encoder") :]] = v
        else:
            new_ckpt[k] = v
    if output_path:
        torch.save(new_ckpt, output_path)
    return new_ckpt


def convert_listconfig_to_list(config):
    new_config = {}
    for k, v in config.items():
        if isinstance(v, ListConfig):
            new_config[k] = list(v)
        else:
            new_config[k] = v
    return new_config


class DOLPHIN:
    def __init__(self, config, ckpt_path="") -> None:
        self.model_args = config.model
        self.swin_args = config.model.pop("swin_args")
        self.swin_args = convert_listconfig_to_list(self.swin_args)

        vision_tower = SwinEncoder(
            input_size=self.swin_args["img_size"],
            patch_size=self.swin_args["patch_size"],
            embed_dim=self.swin_args["embed_dim"],
            window_size=self.swin_args["window_size"],
            encoder_layer=self.swin_args["encoder_layer"],
            num_heads=self.swin_args["num_heads"],
            align_long_axis=self.swin_args["align_long_axis"],
        )

        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.model_args.tokenizer_path)
        self.tokenizer.pad_token = "<pad>"
        self.tokenizer.bos_token = "<s>"
        self.tokenizer.eos_token = "</s>"
        self.tokenizer.unk_token = "<unk>"

        if self.model_args.get("extra_answer_tokens", False):
            # print("Allowing multitask training: adding <Answer/> to the tokenizer.")
            prompt_end_token = " <Answer/>"
            self.tokenizer.add_special_tokens({"additional_special_tokens": sorted(set([prompt_end_token]))})
            self.tokenizer._prompt_end_token = prompt_end_token
            self.tokenizer._prompt_end_token_id = self.tokenizer.convert_tokens_to_ids(prompt_end_token)

        donut_config = DonutConfig(
            decoder_layer=self.model_args.decoder_layer,
            max_length=self.model_args.max_length,
            max_position_embeddings=self.model_args.max_position_embeddings,
            hidden_dimension=self.model_args.hidden_dimension,
        )

        self.model = DonutModel(config=donut_config, vision_tower=vision_tower, tokenizer=self.tokenizer)
        if self.model_args.model_name_or_path:
            ckpt = torch.load(self.model_args.model_name_or_path)
            ckpt = try_rename_lagacy_weights(ckpt)
            self.model.load_state_dict(ckpt, strict=True)

        # Determine the best available device:
        # 1. CUDA (NVIDIA GPUs)
        # 2. Apple Silicon / macOS Metal Performance Shaders (MPS)
        # 3. Fallback to CPU
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"
        else:
            device = "cpu"

        print(f"\nProcessing on device: {device}")

        self.model.to(device)
        self.model.eval()
        transform_args = {
            "input_size": self.swin_args["img_size"],
            "max_length": self.model_args.max_length,
        }
        self.processor = DolphinProcessor({}, self.tokenizer, transform_args=transform_args)

    def chat(
        self,
        question,
        image,
        return_raw=False,
        return_score=False,
        return_img_size=False,
        only_return_img_size=False,
        max_batch_size=16,
    ):

        def _preprocess_image(image):
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            if return_img_size or only_return_img_size:
                image_tensor, ori_size = self.processor.process_image_for_inference(image, return_img_size=True)
            else:
                image_tensor = self.processor.process_image_for_inference(image, return_img_size=False)
                ori_size = None
            return image_tensor, ori_size

        def _preprocess_prompt(question):
            if self.model_args.get("extra_answer_tokens", False):
                if self.tokenizer._prompt_end_token not in question:
                    question = question + self.tokenizer._prompt_end_token
            prompt_ids = self.processor.process_prompt_for_inference(question)
            return prompt_ids

        def _preprocess_prompt_batch(question):
            if self.model_args.get("extra_answer_tokens", False):
                for i in range(len(question)):
                    if self.tokenizer._prompt_end_token not in question[i]:
                        question[i] = question[i] + self.tokenizer._prompt_end_token
                    if not question[i].startswith("<s>"):
                        question[i] = "<s>" + question[i]
            return question

        def _postprocess(output, question):
            output = output.replace("<s>", "").replace(question, "").replace("</s>", "").replace("<pad>", "")
            if self.model_args.get("extra_answer_tokens", False):
                output = output.split(self.tokenizer._prompt_end_token)[-1]
            return output

        if isinstance(question, list):
            image_tensor_list = []
            for i in image:
                image_tensor, ori_size = _preprocess_image(i)
                image_tensor_list.append(image_tensor)
            image_tensor = torch.cat(image_tensor_list, dim=0)

            question = _preprocess_prompt_batch(question)
            self.processor.tokenizer.padding_side = "left"
            prompt_ids = self.processor.tokenizer(
                question, add_special_tokens=False, return_tensors="pt", padding=True
            ).input_ids
        else:
            image_tensor, ori_size = _preprocess_image(image)
            prompt_ids = _preprocess_prompt(question)

        if only_return_img_size:
            return ori_size

        model_output_batch = []
        for i in range(0, image_tensor.shape[0], max_batch_size):
            image_tensor_batch = image_tensor[i : i + max_batch_size]
            prompt_ids_batch = prompt_ids[i : i + max_batch_size]
            model_output = self.model.inference(image_tensors=image_tensor_batch, prompt_ids=prompt_ids_batch)
            model_output_batch.append(model_output)
        model_output = {}
        for k, v in model_output_batch[0].items():
            if isinstance(v, torch.Tensor):
                model_output[k] = sum(
                    [v_batch[k].cpu().numpy().tolist() for v_batch in model_output_batch],
                    [],
                )
            else:
                model_output[k] = sum([v_batch[k] for v_batch in model_output_batch], [])

        if return_raw:
            if return_img_size:
                return model_output, ori_size
            return model_output
        else:
            if isinstance(question, list):
                output = [_postprocess(model_output["repetitions"][i], question[i]) for i in range(len(question))]
                score = model_output["scores"]
            else:
                output = _postprocess(model_output["repetitions"][0], question)
                score = model_output["scores"][0]
            if return_score:
                return output, score
            if return_img_size:
                return output, ori_size
            return output
