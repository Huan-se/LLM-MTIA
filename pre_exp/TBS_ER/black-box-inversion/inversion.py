import os
import argparse
import copy
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Union, Optional
import evaluate
import torch
import torch.nn as nn
import transformers
from datasets import load_from_disk


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    dataset_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to dataset"
        },
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Output directory for training saves. If not set, will output to saves/<random hash>."
        },
    )
    num_train_epochs: float = field(
        default=30.0,
        metadata={"required": False, "help": "Number of epochs for training"},
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for AdamW on the backbone model."},
    )

    per_device_train_batch_size: int = field(
        default=128, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    bf16: bool = field(
        default=False,
        metadata={"help": ("Whether to use bf16 (mixed) precision instead of 32-bit.")},
    )

    remove_unused_columns: bool = False

    eval_strategy: str = "steps"
    logging_strategy: str = "steps"
    save_strategy: str = "steps"

    save_total_limit: int = 2  # Maximum number of checkpoints to save.

    warmup_steps: int = field(
        default=4000, metadata={"help": "Number of steps of warmup"}
    )
    logging_steps: int = field(
        default=400, metadata={"help": "Number of steps between logging metrics"}
    )
    save_steps: int = field(
        default=4000,
        metadata={"help": "Number of steps per save"},
    )
    eval_steps: int = field(
        default=40000,
        metadata={
            "help": "Number of steps between eval (will be scaled as if batch size is 32)"
        },
    )


    include_inputs_for_metrics: bool = True

    def __setattr__(self, name, value):
        super(transformers.TrainingArguments, self).__setattr__(name, value)

    def __post_init__(self):
        super().__post_init__()
        self._frozen = True
        self.report_to = ([])
        self.dataloader_pin_memory = True
        num_workers = torch.cuda.device_count()
        os.environ["RAYON_RS_NUM_CPUS"] = str(num_workers)
        self.dataloader_num_workers = num_workers
        print(f"Set num workers to {num_workers}")

        self.dataloader_drop_last = False

        self.warmup_steps = round(self.warmup_steps * (32 / self.train_batch_size))
        self.logging_steps = round(self.logging_steps * (32 / self.train_batch_size))
        self.eval_steps = round(self.eval_steps * (32 / self.train_batch_size))
        self.save_steps = round(self.save_steps * (32 / self.train_batch_size))

        self.adam_epsilon = 1e-6

        self.group_by_length = True
        self.length_column_name = "length"

        self.load_best_model_at_end = True
        self.greater_is_better = False

        self.do_eval = False

def disable_dropout(model: nn.Module):
    dropout_modules = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    for m in dropout_modules:
        m.p = 0.0
    print(
        f"Disabled {len(dropout_modules)} dropout modules from model type {type(model)}"
    )


def freeze_params(model: nn.Module):
    total_num_params = 0
    for name, params in model.named_parameters():
        params.requires_grad = False
        total_num_params += params.numel()


class InversionConfig(transformers.configuration_utils.PretrainedConfig):
    """We create a dummy configuration class that will just set properties
    based on whatever kwargs we pass in.

    When this class is initialized (see experiments.py) we pass in the
    union of all data, model, and training args, all of which should
    get saved to the config json.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            try:
                json.dumps(value)
                setattr(self, key, value)
            except TypeError:

                continue
        super().__init__()

    def __getattribute__(self, key):
        try:
            return super().__getattribute__(key)
        except AttributeError as e:
            raise e

def args_from_config(args_cls, config):
    args = args_cls()
    for key, value in vars(config).items():
        if key in dir(args):
            setattr(args, key, value)
    return args



def load_encoder_decoder(
    model_name: str, 
) -> transformers.AutoModelForSeq2SeqLM:
    model_kwargs = {
        "low_cpu_mem_usage": True,
    }
    return transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_name, **model_kwargs
    )



class HSInversionModel(transformers.PreTrainedModel):
    """A class of model that conditions on embeddings from a pre-trained sentence embedding model
    to decode text autoregressively.
    """

    config_class = InversionConfig

    encoder_decoder: transformers.AutoModelForSeq2SeqLM
    embedding_transform: nn.Module
    bottleneck_dim: int  # Bottleneck dimension for embedding_transform
    embedder_dim: int  # Hidden dimension of embedding model
    embedder_fake_with_zeros: bool
    embedding_transform_strategy: str  # Way to transform bottleneck embedding into input for encoder-decoder

    embedded_tokens: torch.Tensor  # used for decoding
    # embedder_model_api: Optional[str]

    def __init__(self, config: InversionConfig):
        super().__init__(config=config)


        encoder_dropout_disabled = False
        decoder_dropout_disabled = False

        self.encoder_decoder = load_encoder_decoder(
            model_name=config.model_name_or_path,
        )

        self.embedder_is_decoder = False

        encoder_hidden_dim = self.encoder_decoder.config.hidden_size

        self.embedder_dim = 4096 # Note: Change to the width of inverted LLM if encountered error
        bottleneck_dim = self.embedder_dim

        self.bottleneck_dim = bottleneck_dim

        self.hs_transform = nn.Sequential(
            nn.Linear(self.embedder_dim, bottleneck_dim),
            nn.Dropout(self.encoder_decoder.config.dropout_rate),
            nn.GELU(),
            nn.Linear(bottleneck_dim, encoder_hidden_dim),
        )
        if encoder_dropout_disabled:
            disable_dropout(self.encoder_decoder.encoder)
        if decoder_dropout_disabled:
            disable_dropout(self.encoder_decoder.decoder)
            disable_dropout(self.encoder_decoder.lm_head)

        self.noise_level = 0

    def _freeze_encoder(self):
        freeze_params(self.encoder_decoder.encoder)

    def _freeze_decoder(self):
        freeze_params(self.encoder_decoder.decoder)
        freeze_params(self.encoder_decoder.lm_head)

    def freeze(self, freeze_strategy: str):

        if freeze_strategy == "decoder":
            self._freeze_decoder()
        elif freeze_strategy == "encoder":
            self._freeze_encoder()
        elif freeze_strategy == "encoder_and_decoder":
            self._freeze_encoder()
            self._freeze_decoder()
            freeze_params(self.encoder_decoder.shared)
        elif freeze_strategy == "none":
            pass
        else:
            raise ValueError(f"invalid freezing strategy {freeze_strategy}")

    @property
    def embedder_device(self) -> torch.device:
        return next(self.embedder.parameters()).device

    def embed_and_project(
        self,
        hidden_states: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:


        assert len(hidden_states.shape) == 3  # batch by length by d 

        transformed_hs = self.hs_transform(hidden_states)


        attention_mask = torch.ones(
            (transformed_hs.shape[0], transformed_hs.shape[1]), device=transformed_hs.device
        )
        return transformed_hs, attention_mask

    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        generation_kwargs = copy.copy(generation_kwargs)
        inputs_embeds, attention_mask = self.embed_and_project(

            hidden_states=inputs.get("hidden_states"),
        )

        if "decoder_input_ids" in inputs:
            return self.encoder_decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=inputs["decoder_input_ids"],
                **generation_kwargs,
            )
        else:
            return self.encoder_decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generation_kwargs,
            )

    def forward(
        self,
        labels: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        inputs_embeds, attention_mask = self.embed_and_project(
            hidden_states=hidden_states,)

        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,)



class HS2OutputCollator:
    def __call__(self, features, return_tensors=None):
        hidden_states = []
        labels = []

        for feature in features:
            hidden_states.append(feature['hidden_states'].clone().detach())
            labels.append(feature['input_ids'].clone().detach())

        return {'hidden_states': torch.stack(hidden_states),
            'labels': torch.stack(labels)}



class HS2OutputTrainer(transformers.Trainer):
    additional_metrics: List[Callable[..., Dict[str, float]]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.metric_accuracy = evaluate.load("accuracy")
        self.metric_bleu = evaluate.load("sacrebleu")
        self.metric_rouge = evaluate.load("rouge")
        self.additional_metrics = []

        self.gen_kwargs = {
            "early_stopping": False,
            "num_beams": 1,
            "do_sample": False,
            "no_repeat_ngram_size": 0,
        }

    def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
            return self.model.generate(inputs, generation_kwargs)
    
    @property
    def pad_token_id(self) -> int:
        try:
            return self.model.encoder_decoder.config.pad_token_id
        except AttributeError:
            return self.tokenizer.pad_token_id

    @property
    def bos_token_id(self) -> int:
        try:
            return self.model.encoder_decoder.decoder_start_token_id
        except AttributeError:
            return self.tokenizer.bos_token_id



def train():
    parser = argparse.ArgumentParser(description="Training script with configuration file")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration JSON file')
    args = parser.parse_args()

    with open(args.config) as f:
        config_dict = json.load(f)
    
    config = InversionConfig.from_dict(config_dict)
    training_args = args_from_config(TrainingArguments, config)
    model = HSInversionModel(config=config,)


    train_ds = load_from_disk(training_args.dataset_path)
    train_ds.set_format("pt", columns=["texts", 'input_ids', 'attention_mask', 'hidden_states'], output_all_columns=True)
    print(train_ds)
    trainer = HS2OutputTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None,
        data_collator=HS2OutputCollator(),
    )
    trainer.args.metric_for_best_model = None
    trainer.train()


if __name__ == '__main__':

    train()
