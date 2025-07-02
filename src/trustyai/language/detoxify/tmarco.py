# -*- coding: utf-8 -*-
# pylint: disable = invalid-name, line-too-long, use-dict-literal, consider-using-f-string, too-many-nested-blocks, self-assigning-variable, broad-exception-raised, possibly-used-before-assignment
"""TMaRCo detoxification."""
import os
import math
import itertools
import numpy as np

__SUCCESSFUL_IMPORT = True

try:
    import torch
    from datasets import load_dataset, DatasetDict
    from scipy.spatial.distance import jensenshannon
    from transformers import (
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        pipeline,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        BartForConditionalGeneration,
        BartTokenizer,
        AutoModelForSeq2SeqLM,
        Conversation,
    )
    from torch.nn.functional import softmax
    from torch import Tensor, topk
except ImportError as e:
    print(
        "Warning: detoxify dependencies not found. "
        "Dependencies can be installed with 'pip install trustyai[detoxify]'."
    )
    __SUCCESSFUL_IMPORT = False

DEFAULT_MODEL = "facebook/bart-large"

if __SUCCESSFUL_IMPORT:

    class TMaRCo:
        """TMaRCo detoxification."""

        base = None
        experts = []
        expert_weights = []
        tokenizer = None

        # pylint: disable = too-many-arguments
        def __init__(
            self,
            base_model=None,
            expert_weights=None,
            tokenizer=None,
            max_length=150,
            model_type: str = "causal_lm",
            device=None,
        ):
            if expert_weights is None:
                expert_weights = [-0.5, 0.5]
            self.expert_weights = expert_weights

            if isinstance(tokenizer, str):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer, is_split_into_words=True, add_prefix_space=True
                )
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            elif tokenizer is not None:
                self.tokenizer = tokenizer
            else:
                self.tokenizer = BartTokenizer.from_pretrained(
                    DEFAULT_MODEL, is_split_into_words=True, add_prefix_space=True
                )
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            if isinstance(base_model, str):
                if model_type == "seq2seq_lm":
                    self.base = AutoModelForSeq2SeqLM.from_pretrained(
                        base_model,
                        max_length=max_length,
                        forced_bos_token_id=self.tokenizer.bos_token_id,
                    )
                elif model_type == "causal_lm":
                    self.base = AutoModelForCausalLM.from_pretrained(
                        base_model,
                        max_length=max_length,
                        forced_bos_token_id=self.tokenizer.bos_token_id,
                    )
                else:
                    raise Exception(f"unsupported model type {model_type}")
            elif base_model is not None:
                self.base = base_model
            else:
                self.base = BartForConditionalGeneration.from_pretrained(
                    DEFAULT_MODEL,
                    max_length=max_length,
                    forced_bos_token_id=self.tokenizer.bos_token_id,
                )
            self.content_feature = "comment_text"

            if isinstance(device, str):
                self.device = torch.device(device)
            else:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )

        def load_models(
            self, experts: list[str] = None, expert_weights: list = None
        ):  # pylint: disable=unsubscriptable-object
            """Load expert models."""
            if expert_weights is not None:
                self.expert_weights = expert_weights
            expert_models = []
            for expert in experts:
                # Load TMaRCO models
                if expert in ["trustyai/gplus", "trustyai/gminus"]:
                    expert = BartForConditionalGeneration.from_pretrained(
                        expert,
                        forced_bos_token_id=self.tokenizer.bos_token_id,
                        device_map="auto",
                    )
                # Load local models
                elif os.path.exists(os.path.dirname(expert)):
                    expert = AutoModelForMaskedLM.from_pretrained(
                        expert,
                        forced_bos_token_id=self.tokenizer.bos_token_id,
                        device_map="auto",
                    )
                # Load HuggingFace models
                else:
                    expert = AutoModelForCausalLM.from_pretrained(
                        expert,
                        forced_bos_token_id=self.tokenizer.bos_token_id,
                        device_map="auto",
                    )
                expert_models.append(expert)
            self.experts = expert_models

        def tokenize_function(self, examples):
            """Tokenize function."""
            return self.tokenizer(
                examples[self.content_feature], max_length=1024, truncation=True
            )

        @staticmethod
        def group_texts(examples, block_size=128):
            """Group texts."""
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it
            # instead of this drop, you can customize this part to your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        def mask_tokens(self, sentence, mask_token):
            """Mask tokens."""
            masked_sentences = []
            tokens = self.tokenizer.tokenize(sentence)
            for idx in range(len(tokens)):
                masked_sentence = tokens.copy()
                masked_sentence[idx] = mask_token
                masked_sentence = self.tokenizer.convert_tokens_to_string(
                    masked_sentence
                )
                masked_sentences.append(masked_sentence)
            return masked_sentences

        # pylint: disable = too-many-locals
        def train_models(
            self,
            dataset_name: str = "jigsaw_toxicity_pred",
            perc: int = 100,
            expert_feature: str = "toxic",
            data_dir: str = "jigsaw-toxic-comment-classification-challenge",
            td_columns=None,
            base_model=DEFAULT_MODEL,
            content_feature: str = "comment_text",
            model_type=None,
            model_prefix: str = "g_",
        ):
            """Train models."""

            self.content_feature = content_feature

            if td_columns is None:
                td_columns = [
                    "comment_text",
                    "toxic",
                    "severe_toxic",
                    "obscene",
                    "threat",
                    "insult",
                    "identity_hate",
                ]

            ds_size = ["train[:" + str(perc) + "%]", "test[:" + str(perc) + "%]"]

            datasets_split = load_dataset(
                dataset_name, data_dir=data_dir, split=ds_size
            )
            datasets_split = DatasetDict(
                {"train": datasets_split[0], "test": datasets_split[1]}
            )

            toxic_datasets = datasets_split.filter(
                lambda x: int(x[expert_feature]) == 1
            )

            tokenized_datasets = toxic_datasets.map(
                self.tokenize_function,
                batched=True,
                num_proc=4,
                remove_columns=td_columns,
            )

            lm_datasets = tokenized_datasets.map(
                self.group_texts,
                batched=True,
                batch_size=100,
                num_proc=4,
            )

            if model_type is None:
                gminus = BartForConditionalGeneration.from_pretrained(
                    base_model,
                    forced_bos_token_id=self.tokenizer.bos_token_id,
                    device_map="auto",
                )
            elif model_type == "causal_lm":
                gminus = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    forced_bos_token_id=self.tokenizer.bos_token_id,
                    device_map="auto",
                )
            elif model_type == "seq2seq_lm":
                gminus = AutoModelForSeq2SeqLM.from_pretrained(
                    base_model,
                    forced_bos_token_id=self.tokenizer.bos_token_id,
                    device_map="auto",
                )
            else:
                raise Exception(f"unsupported model type {model_type}")

            training_args = TrainingArguments(
                model_prefix + "minus",
                evaluation_strategy="epoch",
                learning_rate=2e-5,
                weight_decay=0.01,
            )

            trainer = Trainer(
                model=gminus,
                args=training_args,
                train_dataset=lm_datasets["train"],
                eval_dataset=lm_datasets["test"],
            )

            trainer.train()

            eval_results = trainer.evaluate()
            print(f"G- perplexity: {math.exp(eval_results['eval_loss']):.2f}")
            trainer.save_model(model_prefix + "minus")

            # train gplus model
            nontoxic_datasets = datasets_split.filter(
                lambda x: int(x[expert_feature]) == 0
            )

            nontoxic_tokenized_datasets = nontoxic_datasets.map(
                self.tokenize_function,
                batched=True,
                num_proc=4,
                remove_columns=td_columns,
            )

            nontoxic_lm_datasets = nontoxic_tokenized_datasets.map(
                self.group_texts,
                batched=True,
                batch_size=100,
                num_proc=4,
            )

            if model_type is None:
                gplus = BartForConditionalGeneration.from_pretrained(
                    base_model,
                    forced_bos_token_id=self.tokenizer.bos_token_id,
                    device_map="auto",
                )
            elif model_type == "causal_lm":
                gplus = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    forced_bos_token_id=self.tokenizer.bos_token_id,
                    device_map="auto",
                )
            elif model_type == "seq2seq_lm":
                gplus = AutoModelForSeq2SeqLM.from_pretrained(
                    base_model,
                    forced_bos_token_id=self.tokenizer.bos_token_id,
                    device_map="auto",
                )
            else:
                raise Exception(f"unsupported model type {model_type}")

            nt_training_args = TrainingArguments(
                model_prefix + "plus",
                evaluation_strategy="epoch",
                learning_rate=2e-5,
                weight_decay=0.01,
                save_total_limit=2,
            )

            nt_trainer = Trainer(
                model=gplus,
                args=nt_training_args,
                train_dataset=nontoxic_lm_datasets["train"],
                eval_dataset=nontoxic_lm_datasets["test"],
            )

            nt_trainer.train()

            nt_eval_results = nt_trainer.evaluate()
            print(f"G+ perplexity: {math.exp(nt_eval_results['eval_loss']):.2f}")
            print("training finished")

            nt_trainer.save_model(model_prefix + "plus")
            self.experts = [gminus, gplus]

        def mask(
            self,
            sentences: list,
            threshold: float = 1.2,
            normalize: bool = True,
            use_logits: bool = True,
            scores: list = None,
        ):
            """Mask sentences."""
            masked_sentences = []
            if scores is None:
                scores = self.score(
                    sentences, use_logits=use_logits, normalize=normalize
                )
            # pylint: disable = consider-using-enumerate
            for s_idx in range(len(sentences)):
                sentence = sentences[s_idx]
                tokens = self.tokenizer.tokenize(sentence)
                masked_output = []
                for idx in range(len(tokens)):
                    if scores[s_idx][idx] > threshold:
                        masked_output.append(self.tokenizer.mask_token)
                    else:
                        masked_output.append(tokens[idx])
                masked_sentence = self.tokenizer.convert_tokens_to_string(masked_output)
                masked_sentences.append(masked_sentence)
            return masked_sentences

        @staticmethod
        def compute_mask_probs(fmp, text_sentence):
            """Compute mask probabilities."""
            fm_result = fmp(text_sentence)
            score_list = [(d["token"], d["score"]) for d in fm_result]
            token_scores = sorted(score_list, key=lambda x: x[0])
            token_ids = [e[1] for e in token_scores]
            token_scores = [e[1] for e in token_scores]
            return token_ids, token_scores

        # pylint: disable = too-many-branches, too-many-statements
        def rephrase(
            self,
            originals: list,
            scores: list = None,
            masked_outputs: list = None,
            compute_probs: bool = False,
            verbose: bool = False,
            combine_original: bool = False,
            expert_weights: list = None,
            conditional: bool = True,
            threshold: float = 1.2,
        ):
            """Rephrase sentences."""
            rephrased_texts = []
            if expert_weights is None:
                expert_weights = self.expert_weights
            if scores is None:
                scores = self.score(originals)
            if masked_outputs is None:
                masked_outputs = self.mask(originals, scores=scores)
            # pylint: disable = consider-using-enumerate, too-many-nested-blocks
            for in_idx in range(len(originals)):
                original = originals[in_idx]
                if len(original.strip()) == 0:
                    rephrased_texts.append(original)
                elif conditional and max(scores[in_idx]) <= threshold:
                    # do not rephrase if all scores are below the threshold
                    rephrased_texts.append(original)
                    continue
                else:
                    base_logits = self.compute_mask_logits(
                        self.base, original, mask=False
                    )
                    rephrased_tokens_ids = []
                    masked_sentence_tokens = self.tokenizer.tokenize(
                        masked_outputs[in_idx]
                    )
                    if combine_original:
                        original_sentence_tokens = self.tokenizer(original)[
                            "input_ids"
                        ][1:-1]
                    fmp_experts = []
                    if compute_probs:
                        for expert in self.experts:
                            fmp_experts.append(
                                pipeline(
                                    "fill-mask",
                                    model=expert,
                                    tokenizer=self.tokenizer,
                                    top_k=self.tokenizer.vocab_size,
                                    device=self.device,
                                )
                            )
                    for idx in range(len(masked_sentence_tokens)):
                        if masked_sentence_tokens[idx] == self.tokenizer.mask_token:
                            next_token_logits = base_logits[0, 1 + idx]
                            if verbose:
                                self.print_token(next_token_logits)
                            expert_logits = []
                            current_sentence_ids = rephrased_tokens_ids + [
                                self.tokenizer.mask_token_id
                            ]
                            if (
                                combine_original
                                and idx < len(masked_sentence_tokens) - 2
                            ):
                                current_sentence_ids = (
                                    current_sentence_ids
                                    + original_sentence_tokens[idx + 1 :]
                                )
                                if verbose:
                                    print(
                                        self.tokenizer.convert_ids_to_tokens(
                                            current_sentence_ids
                                        )
                                    )
                            if compute_probs:
                                masked_sentence = (
                                    self.tokenizer.convert_tokens_to_string(
                                        self.tokenizer.convert_ids_to_tokens(
                                            current_sentence_ids
                                        )
                                    )
                                )
                                for expert in fmp_experts:
                                    _, scores = self.compute_mask_probs(
                                        expert, masked_sentence
                                    )
                                    expert_logits.append(scores)
                                for eidx in range(len(expert_logits)):
                                    tensor = Tensor(expert_logits[eidx])
                                    next_token_logits *= expert_weights[eidx] * tensor
                                log_prob = next_token_logits
                            else:
                                masked_sequence = (
                                    self.tokenizer.convert_tokens_to_string(
                                        self.tokenizer.convert_ids_to_tokens(
                                            current_sentence_ids
                                        )
                                    )
                                )
                                eidx = 0
                                for expert in self.experts:
                                    next_token_logits += expert_weights[
                                        eidx
                                    ] * self.compute_mask_logits(
                                        expert, masked_sequence
                                    )
                                    eidx += 1
                                log_prob = next_token_logits
                            if verbose:
                                self.print_token(next_token_logits)
                            argmaxed = np.argmax(log_prob).item()
                            rephrased_token_id = argmaxed
                            rephrased_tokens_ids.append(rephrased_token_id)
                        else:
                            rephrased_tokens_ids.append(
                                self.tokenizer.convert_tokens_to_ids(
                                    [masked_sentence_tokens[idx]]
                                )[0]
                            )
                    rephrased_texts.append(
                        self.tokenizer.decode(
                            rephrased_tokens_ids,
                            clean_up_tokenization_spaces=True,
                            skip_special_tokens=True,
                        )
                    )
            return rephrased_texts

        def print_token(self, token_logits):
            """Print token."""
            log_prob = softmax(token_logits, dim=0)
            argmaxed = np.argmax(log_prob).item()
            rephrased_token = self.tokenizer.decode(argmaxed)
            print(rephrased_token)
            print(
                [
                    self.tokenizer.decode(i.item()).strip()
                    for i in topk(token_logits, 5)[1]
                ]
            )

        # pylint: disable = no-else-return
        def compute_mask_logits(
            self, model, sequence, verbose: bool = False, mask: bool = True
        ):
            """Compute mask logits."""
            model.to(self.device)
            if verbose:
                print(f"input sequence: {sequence}")
            subseq_ids = self.tokenizer(sequence, return_tensors="pt").to(self.device)
            if verbose:
                raw_outputs = model.generate(**subseq_ids)
                print(sequence)
                print(
                    self.tokenizer.batch_decode(raw_outputs, skip_special_tokens=True)[
                        0
                    ]
                )
            with torch.no_grad():
                if mask:
                    mt_idx = torch.nonzero(
                        subseq_ids.input_ids[0] == self.tokenizer.mask_token_id
                    ).item()
                    return model.forward(**subseq_ids).logits[0, mt_idx]
                else:
                    return model.forward(**subseq_ids).logits

        # pylint: disable = consider-using-enumerate
        def compute_mask_logits_multiple(
            self, model, sequences, verbose: bool = False, mask: bool = True
        ):
            """Compute mask logits multiple."""
            model.to(self.device)
            if verbose:
                print(f"input sequences: {sequences}")
            subseq_ids = self.tokenizer(
                sequences, return_tensors="pt", padding=True
            ).to(self.device)
            if verbose:
                raw_outputs = model.generate(**subseq_ids)
                print(sequences)
                print(
                    self.tokenizer.batch_decode(raw_outputs, skip_special_tokens=True)
                )
            with torch.no_grad():
                if mask:
                    raw_outputs = model.forward(**subseq_ids).logits
                    mt_idx = torch.nonzero(
                        subseq_ids.input_ids == self.tokenizer.mask_token_id
                    )[:, 1]
                    tensors = []
                    for idx in range(len(mt_idx)):
                        tensors.append(raw_outputs[idx, mt_idx[idx]].unsqueeze(0))
                    return torch.cat(tensors, dim=0)
                else:
                    return model.forward(**subseq_ids).logits

        def score(
            self,
            sentences: list,
            use_logits: bool = True,
            normalize: bool = True,
            verbose: bool = False,
        ):
            """Score sentences."""
            scores = []
            for sentence in sentences:
                masked_sentences = self.mask_tokens(
                    sentence, self.tokenizer.pad_token + self.tokenizer.mask_token
                )
                if len(masked_sentences) == 0:
                    scores.append([])
                else:
                    distributions = []
                    for model in self.experts:
                        if use_logits:
                            logits = self.compute_mask_logits_multiple(
                                model, masked_sentences, verbose=verbose
                            )
                            mask_substitution_scores = softmax(logits, dim=1)
                        else:
                            mask_substitution_scores = []
                            fmp = pipeline(
                                "fill-mask",
                                model=model,
                                tokenizer=self.tokenizer,
                                top_k=10,
                                device=self.device,
                            )
                            for masked_sentence in masked_sentences:
                                # approximated probabilities for top_k tokens
                                distr = fmp(masked_sentence)
                                mask_substitution_score = [x["score"] for x in distr]
                                mask_substitution_scores.append(
                                    mask_substitution_score.numpy()
                                )
                        distributions.append(mask_substitution_scores)
                    distr_pairs = itertools.combinations(distributions, 2)
                    js_distances = []
                    for distr_pair in distr_pairs:
                        js_distance = jensenshannon(
                            distr_pair[0].cpu().clone().numpy(),
                            distr_pair[1].cpu().clone().numpy(),
                            axis=1,
                        )
                        if normalize:
                            js_distance = js_distance / np.average(js_distance)
                        js_distances.append(js_distance)
                    js_distance = np.average(js_distances, axis=0)
                    scores.append(js_distance)
            return scores

        # pylint: disable = no-else-break
        def rephrase_incrementally(
            self, text, threshold=1.2, verbose=False, max_attempts=10
        ):
            """Rephrase incrementally."""
            incrementally_rephrased = text
            idx = 1
            while True:
                if len(text.strip()) == 0:
                    break
                scores = self.score(text, verbose=verbose)
                if min(scores) > threshold:
                    break
                masked = self.mask(text, scores=scores, threshold=threshold)
                incrementally_rephrased = self.rephrase(
                    [text], masked_outputs=masked, scores=scores, verbose=verbose
                )[0]
                if verbose:
                    print(f"step{idx}: {incrementally_rephrased}")
                if text == incrementally_rephrased or idx == max_attempts:
                    break
                else:
                    text = incrementally_rephrased
                    idx += 1
            return incrementally_rephrased

        def to_dict(self, t, s):
            """To dict."""
            sd = dict()
            ts = self.tokenizer.tokenize(t)
            for idx in range(len(ts)):
                sd[ts[idx][1:]] = s[idx]
            return sd

        default_user_prompt = (
            'In the text "{input}", the following words are considered {toxic_trait}: {scored_words_text}. '
            'How should the text "{input}" be adjusted to make it non-{toxic_trait} (e.g. "{rephrased}")?'
        )
        default_system_prompt = "You are a taciturn chatbot for removing {toxic_trait}, harmful and abusive language."

        def reflect(
            self,
            texts: list,
            threshold=1.2,
            verbose=False,
            messages=None,
            user_prompt=default_user_prompt,
            system_prompt: str = default_system_prompt,
            chat_model=None,
            chat_tokenizer=None,
            toxic_trait="toxic",
            end_tag="<|",
            conversation_type: str = "rephrase",
            chat_template=None,
            chain_of_thought: bool = False,
            conditional: bool = True,
        ):
            """Reflect."""
            reflected_outputs = []

            scores = self.score(texts)
            masks = self.mask(texts, scores=scores, threshold=threshold)
            rephrased_texts = self.rephrase(texts, scores=scores, masked_outputs=masks)

            if chat_model is None:
                chat_model = self.base
                chat_tokenizer = self.tokenizer
            else:
                if chat_tokenizer is None:
                    chat_tokenizer = AutoTokenizer.from_pretrained(chat_model)
                else:
                    chat_tokenizer = chat_tokenizer

            if chat_template is not None:
                chat_tokenizer.chat_template = chat_template

            converse_pipeline = pipeline(
                "conversational",
                model=chat_model,
                tokenizer=chat_tokenizer,
                device=self.device,
            )

            for text_id in range(len(texts)):
                text = texts[text_id]
                scores_current = scores[text_id]
                if len(text.strip()) == 0 or (
                    conditional and max(scores_current) < threshold
                ):
                    reflected_outputs.append(text)
                else:
                    scores_dict = self.to_dict(text, scores_current)
                    toxic_words = []
                    for t, s in scores_dict.items():
                        if s > threshold:
                            toxic_words.append("'" + t + "'")
                    scored_words_text = ", ".join(toxic_words)
                    rephrased_text = rephrased_texts[text_id]
                    if conversation_type == "chat":
                        if messages is None:
                            if chain_of_thought:
                                system_prompt = system_prompt.format(
                                    toxic_trait=toxic_trait
                                )
                                user1 = 'In the text "{input}", which words are considered {toxic_trait}?'.format(
                                    input=text, toxic_trait=toxic_trait
                                )
                                assistant1 = scored_words_text
                                user2 = 'How could the text "{input}" be adjusted to make it non-{toxic_trait}?'.format(
                                    input=text, toxic_trait=toxic_trait
                                )
                                user3 = 'How could the text "{input}" be adjusted to make it non-{toxic_trait}?'.format(
                                    input=text, toxic_trait=toxic_trait
                                )
                                formatted_messages = [
                                    {
                                        "role": "system",
                                        "content": system_prompt,
                                    },
                                    {"role": "user", "content": user1},
                                    {"role": "assistant", "content": assistant1},
                                    {"role": "user", "content": user2},
                                    {"role": "assistant", "content": rephrased_text},
                                    {"role": "user", "content": user3},
                                ]
                            else:
                                user_prompt = user_prompt.format(
                                    input=text,
                                    scored_words_text=scored_words_text,
                                    rephrased=rephrased_text,
                                    toxic_trait=toxic_trait,
                                )
                                system_prompt = system_prompt.format(
                                    toxic_trait=toxic_trait
                                )
                                formatted_messages = [
                                    {
                                        "role": "system",
                                        "content": system_prompt,
                                    },
                                    {"role": "user", "content": user_prompt},
                                ]
                        else:
                            formatted_messages = []
                            for message in messages:
                                formatted_messages.append(
                                    message.format(
                                        input=text,
                                        scored_words_text=scored_words_text,
                                        rephrased=rephrased_text,
                                        toxic_trait=toxic_trait,
                                    )
                                )
                        conversation_output = converse_pipeline(
                            formatted_messages,
                            pad_token_id=converse_pipeline.tokenizer.eos_token_id,
                            device=self.device,
                        )
                        if verbose:
                            print(f"chat conversation:\n{conversation_output}")
                        reflected_texts = [conversation_output.messages[-1]["content"]]
                    elif conversation_type == "rephrase":
                        user_prompt = f"Make sure the text '{text}' doesn't contain {toxic_trait} content.\n"
                        rephrase_command = "You may rephrase it as"
                        user_prompt += f"The following words are {toxic_trait}: {scored_words_text}. "
                        user_prompt += rephrase_command + f" '{rephrased_text}'."
                        reflection_conversation = Conversation(user_prompt)
                        reflected_output = converse_pipeline(
                            [reflection_conversation],
                            pad_token_id=converse_pipeline.tokenizer.eos_token_id,
                        )
                        if verbose:
                            print(f"{reflected_output}")
                        reflected_texts = []
                        for generated_response in reflected_output.generated_responses:
                            if (
                                rephrase_command in generated_response
                                and end_tag in generated_response
                            ):
                                start_index = generated_response.index(
                                    rephrase_command
                                ) + len(rephrase_command)
                                end_index = generated_response.index(
                                    end_tag, start_index
                                )
                                if 0 < start_index < end_index:
                                    gras = generated_response[start_index:end_index]
                                    reflected_texts.append(gras)
                                else:
                                    reflected_texts.append(generated_response)
                            else:
                                reflected_texts.append(generated_response)
                    else:
                        raise Exception(
                            f"unsupported conversation type '{conversation_type}'"
                        )
                    reflected_outputs.append(reflected_texts[0])
            return reflected_outputs

else:
    print(
        "Module 'trustyai.language.detoxify' did not load completely due to missing dependencies."
    )
