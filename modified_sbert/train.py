import math
import os
from typing import Dict, Optional, Tuple, Union, Type, Callable
from tqdm.autonotebook import tqdm, trange

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from transformers import AutoTokenizer, AutoConfig
from transformers import RobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

import sentence_transformers
from sentence_transformers import models, datasets, LoggingHandler, SentenceTransformer, util 
from sentence_transformers.datasets import SentenceLabelDataset, SentencesDataset
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.evaluation import SentenceEvaluator

import logging
from transformers import logging as lg
import wandb    

import losses, data_loaders, evaluation


lg.set_verbosity_error()
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


def train_biencoder(
        train_data: dict = None,
        dev_data: dict = None,
        base_model='sentence-transformers/all-MiniLM-L12-v2',
        add_pooling_layer=False,
        train_batch_size=64,
        num_epochs=10,
        warm_up_perc=0.1,
        loss_fn='contrastive',
        loss_params=None,
        model_save_path="output",
        wandb_names=None,  
):

    if loss_fn not in ['contrastive', 'cosine', 'triplet', 'mnrl', 'supcon', 'supcon_batchhard', 'contrastive_batchhard']:
        raise ValueError("loss_fn must be 'contrastive', 'cosine', 'triplet', 'mnrl', 'supcon', 'supcon_batchhard', 'contrastive_batchhard'")
    if loss_fn == "constrastive" or loss_fn == "triplet":
        if "distance_metric" not in loss_params:
            raise ValueError('loss_params must contain "distance_metric"')
        if "margin" not in loss_params:
            raise ValueError('loss_params must contain "margin"')

    # Logging
    if wandb_names: 
        if 'run' in wandb_names:
            wandb.init(project=wandb_names['project'], entity=wandb_names['id'], reinit=True, name=wandb_names['run'])
        else:
            wandb.init(project=wandb_names['project'], entity=wandb_names['id'], reinit=True)

        wandb.config = {
            "epochs": num_epochs,
            "batch_size": train_batch_size,
            "warm_up": warm_up_perc,
        }
        

    os.makedirs(model_save_path, exist_ok=True)

    # Base language model
    if add_pooling_layer:
        word_embedding_model = models.Transformer(base_model, max_seq_length=512)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    elif type(base_model) == SentenceTransformer:
        model = base_model
    else:
        model = SentenceTransformer(base_model)

    wandb.watch(model)

    # Loss functions
    if loss_fn == "contrastive":
        train_loss = losses.OnlineContrastiveLoss_wandb(   
            model=model,
            distance_metric=loss_params['distance_metric'],
            margin=loss_params['margin']
        )

        train_samples = data_loaders.load_data_as_pairs(train_data, type="training")
        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

    if loss_fn == "contrastive_batchhard":
        train_loss = losses.OnlineContrastiveLoss_wandb(   
            model=model,
            distance_metric=loss_params['distance_metric'],
            margin=loss_params['margin']
        )

        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)

    elif loss_fn == "cosine":
        train_loss = losses.CosineSimilarityLoss_wandb(model=model) 

        train_samples = data_loaders.load_data_as_pairs(train_data, type="training")
        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

    elif loss_fn == "triplet":
        train_loss = losses.TripletLoss_wandb(   
            model=model,
            distance_metric=loss_params['distance_metric'],
            triplet_margin=loss_params['margin']
        )

        train_samples = data_loaders.load_data_as_triplets(train_data, type="training")
        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

    elif loss_fn == "mnrl":
        train_loss = losses.MultipleNegativesRankingLoss_wandb(model=model)  

        train_samples = data_loaders.load_data_as_triplets(train_data, type="training")

        # Special dataloader that avoid duplicates within a batch
        train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)

    elif loss_fn == "supcon":
        train_loss = losses.SupConLoss_wandb(model=model)    

        train_samples = data_loaders.load_data_as_individuals(train_data, type="training")

        # Special dataset "SentenceLabelDataset" to wrap out train_set
        # It yields batches that contain at least two samples with the same label
        train_data_sampler = SentenceLabelDataset(train_samples)
        train_dataloader = DataLoader(train_data_sampler, batch_size=train_batch_size)

    elif loss_fn == "supcon_batchhard":

        # Assumes that you already have the data set up as a list of input examples, with hard negatives 

        train_loss = losses.SupConLoss_wandb(model=model, contrast_mode='all')    
    
        train_data_sampler = SentencesDataset(train_data, model=model)

        train_dataloader = DataLoader(train_data_sampler, shuffle=False, batch_size=train_batch_size)


    # Evaluate with multiple evaluators
    if loss_fn == "supcon_batchhard" or loss_fn == "contrastive_batchhard":

        evaluators = [
            evaluation.BinaryClassificationEvaluator_wandb.from_input_examples(dev_data),  
            # evaluation.ClusterEvaluator_wandb.from_input_examples(dev_data, cluster_type="agglomerative")  
        ]

    else:
        dev_pairs = data_loaders.load_data_as_pairs(dev_data, type="dev")
        # dev_triplets = data_loaders.load_data_as_triplets(dev_data, type="dev")

        evaluators = [
            evaluation.BinaryClassificationEvaluator_wandb.from_input_examples(dev_pairs),  
            # evaluation.EmbeddingSimilarityEvaluator_wandb.from_input_examples(dev_pairs), 
            # evaluation.TripletEvaluator_wandb.from_input_examples(dev_triplets),   
            # evaluation.ClusterEvaluator_wandb.from_input_examples(dev_pairs, cluster_type="agglomerative")  
        ]

    seq_evaluator = sentence_transformers.evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])

    logger.info("Evaluate model without training")
    seq_evaluator(model, epoch=0, steps=0, output_path=model_save_path)

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=seq_evaluator,
        epochs=num_epochs,
        warmup_steps=math.ceil(len(train_dataloader) * num_epochs * warm_up_perc),
        output_path=model_save_path,
        evaluation_steps=math.ceil(len(train_dataloader)/10),
        checkpoint_save_steps=math.ceil(len(train_dataloader)/10),
        checkpoint_path=model_save_path,
        save_best_model=True,
        checkpoint_save_total_limit=10
    )


def train_crossencoder(
        train_data,
        dev_data,
        model_name,
        lr,
        train_batch_size,
        num_epochs,
        warm_up_perc,
        eval_per_epoch,
        model_save_path,
        run_name=None,
        wandb_project_name=None,  
        lambert=False  
):

    # Logging
    if run_name:
        wandb.init(project=wandb_project_name, entity="emilys", reinit=True, name=run_name)
    else:
        wandb.init(project=wandb_project_name, entity="emilys", reinit=True)

    wandb.config = {
        "learning_rate": lr,
        "epochs": num_epochs,
        "batch_size": train_batch_size,
        "warm_up": warm_up_perc
    }

    if lambert:
        model = CrossEncoder_lambert(model_name, num_labels=2)

        train = data_loaders.load_data_as_lambert_pairs(train_data, type="training")
        dev = data_loaders.load_data_as_lambert_pairs(dev_data, type="dev")

    else:

        # Add special NIL token to tokenizer
        tokens = ["[NIL]"]  

        org_model = models.Transformer(model_name)
        org_model.tokenizer.add_tokens(tokens, special_tokens=True)
        org_model.auto_model.resize_token_embeddings(len(org_model.tokenizer))

        model = CrossEncoder(model_name, num_labels=1)

        train = data_loaders.load_data_as_pairs(train_data, type="training")
        dev = data_loaders.load_data_as_pairs(dev_data, type="dev")

    wandb.watch(model.model) 

    # Wrap train_samples, which is a list of InputExample, in a pytorch DataLoader
    train_dataloader = DataLoader(train, shuffle=True, batch_size=train_batch_size)

    # Evaluate with multiple evaluators
    evaluators = [
        evaluation.CEBinaryClassificationEvaluator_wandb.from_input_examples(dev, name='dev'),   
        evaluation.CEClusterEvaluator_wandb.from_input_examples(dev, name='dev'),    
    ]

    seq_evaluator = sentence_transformers.evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[0])   # Todo: maybe change back scores[-1]

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * warm_up_perc)
    logger.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_dataloader=train_dataloader,
              evaluator=seq_evaluator,
              epochs=num_epochs,
              evaluation_steps=int(len(train_dataloader)*(1/eval_per_epoch)),
              loss_fct=losses.LoggingLoss(torch.nn.BCEWithLogitsLoss(), wandb),   
              optimizer_params={"lr": lr},
              warmup_steps=warmup_steps,
              output_path=model_save_path)


class LambertForSequenceClassification(nn.Module):

    """
    A wrapper over `RobertaForSequenceClassification`, providing patched `forward` method accepting
    an additional `bboxes` argument
    Args:
        roberta (RobertaForTokenClassification): original RobertaModel instance
        base (int): `base` parameter of `LayoutEmbeddings`
    """
    def __init__(self, roberta: RobertaForSequenceClassification, config, base: int = 500) -> None:
        super().__init__()

        self.roberta = roberta

        # add attributes to avoid modifications in code copied from `transformers`
        self.lambert = LambertModel(roberta.roberta, base)
        self.config = config
        self.classifier = roberta.classifier
        self.num_labels = roberta.num_labels

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        bboxes=None                                              # added argument
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.lambert(       # substituted `roberta` with `lambert`
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            bboxes=bboxes  # added argument
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# class RobertaForSequenceClassification_test(RobertaForSequenceClassification):
#
#     def forward(
#             self,
#             input_ids: Optional[torch.LongTensor] = None,
#             attention_mask: Optional[torch.FloatTensor] = None,
#             token_type_ids: Optional[torch.LongTensor] = None,
#             position_ids: Optional[torch.LongTensor] = None,
#             head_mask: Optional[torch.FloatTensor] = None,
#             inputs_embeds: Optional[torch.FloatTensor] = None,
#             labels: Optional[torch.LongTensor] = None,
#             output_attentions: Optional[bool] = None,
#             output_hidden_states: Optional[bool] = None,
#             return_dict: Optional[bool] = None,
#             bboxes=None
#     ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         print(input_ids.shape)
#         print(attention_mask.shape)
#
#         outputs = self.roberta(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         sequence_output = outputs[0]
#         logits = self.classifier(sequence_output)
#
#         print("3333333333333333333333")
#         print(logits.shape)
#         print("3333333333333333333333")
#
#         loss = None
#         if labels is not None:
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"
#
#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss()
#                 if self.num_labels == 1:
#                     loss = loss_fct(logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(logits, labels)
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(logits, labels)
#
#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output
#
#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


class CrossEncoder_lambert(CrossEncoder):

    def __init__(
            self,
            model_name: str,
            num_labels: int = None,
            max_length: int = None,
            device: str = None,
            tokenizer_args: Dict = {},
            automodel_args: Dict = {},
            default_activation_function=None
    ):
        """
        Update of sbert's cross encoder to allow integration with LAMBert
        """

        self.config = AutoConfig.from_pretrained(model_name)

        classifier_trained = True
        if self.config.architectures is not None:
            classifier_trained = any([arch.endswith('ForSequenceClassification') for arch in self.config.architectures])

        if num_labels is None and not classifier_trained:
            num_labels = 1

        if num_labels is not None:
            self.config.num_labels = num_labels

        base_model = RobertaForSequenceClassification.from_pretrained(model_name)
        self.model = LambertForSequenceClassification(base_model, config=self.config, **automodel_args)

        # self.model = RobertaForSequenceClassification_test.from_pretrained(model_name, config=self.config, **automodel_args)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))

        self._target_device = torch.device(device)

        if default_activation_function is not None:
            self.default_activation_function = default_activation_function
            try:
                self.config.sbert_ce_default_activation_function = util.fullname(self.default_activation_function)
            except Exception as e:
                logger.warning("Was not able to update config about the default_activation_function: {}".format(str(e)) )
        elif hasattr(self.config, 'sbert_ce_default_activation_function') and self.config.sbert_ce_default_activation_function is not None:
            self.default_activation_function = util.import_from_string(self.config.sbert_ce_default_activation_function)()
        else:
            self.default_activation_function = nn.Sigmoid() if self.config.num_labels == 1 else nn.Identity()

    def smart_batching_collate(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        bboxes = [[] for _ in range(len(batch[0].bboxes))]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())

            labels.append(example.label)

            for idx, bbox in enumerate(example.bboxes):
                bboxes[idx].append("and the he it")

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)

        # Add bboxes
        tokenized['bboxes'] = torch.rand((128, 512, 4))   # Todo: modify for actual coordinates

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        # print("********")
        # print(type(tokenized['bboxes']))
        # print(tokenized['bboxes'].shape)
        # print("********")

        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self._target_device)

        # bboxes = self.tokenizer(*bboxes, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
        # for name in bboxes:
        #     print(name.shape)
        #     bboxes[name] = bboxes[name].to(self._target_device)

        return tokenized, labels

    def fit(self,
            train_dataloader: DataLoader,
            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            loss_fct = None,
            activation_fct = nn.Identity(),
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = torch.optim.AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True
            ):

        train_dataloader.collate_fn = self.smart_batching_collate

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.model.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        if loss_fct is None:
            loss_fct = nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            for features, labels in tqdm(train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                if use_amp:
                    with autocast():
                        model_predictions = self.model(**features, return_dict=True)

                        logits = activation_fct(model_predictions.logits)
                        if self.config.num_labels == 1:
                            logits = logits.view(-1)
                        loss_value = loss_fct(logits, labels)

                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    model_predictions = self.model(**features, return_dict=True)
                    logits = activation_fct(model_predictions.logits)

                    print("***************")
                    print(logits.shape)

                    if self.config.num_labels == 1:
                        logits = logits.view(-1)

                    print(logits.shape)
                    print("***************")

                    loss_value = loss_fct(logits, labels)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1

                if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback)

                    self.model.zero_grad()
                    self.model.train()

            if evaluator is not None:
                self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)