import time
import warnings
from abc import ABC
from copy import deepcopy
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import functional as F

from ..tokenization_utils_base import PreTrainedTokenizerBase
from ..utils import add_start_docstrings, logging


logger = logging.get_logger(__name__)


STOPPING_CRITERIA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax
            or scores for each vocabulary token after SoftMax. If this stopping criteria depends on the `scores` input,
            make sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional stopping criteria specific kwargs.

    Return:
        `torch.BoolTensor`. (`torch.BoolTensor` of shape `(batch_size, 1)`), where `True` indicates we stop generation
            for a particular row, `True` indicates we should continue.

"""


class StoppingCriteria(ABC):
    """Abstract base class for all stopping criteria that can be applied during generation.

    If your stopping criteria depends on the `scores` input, make sure you pass `return_dict_in_generate=True,
    output_scores=True` to `generate`.
    """

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        raise NotImplementedError("StoppingCriteria needs to be subclassed")


class MaxLengthCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generated number of tokens exceeds `max_length`. Keep
    in mind for decoder-only type of transformers, this will include the initial prompted tokens.

    Args:
        max_length (`int`):
            The maximum length that the output sequence can have in number of tokens.
        max_position_embeddings (`int`, *optional*):
            The maximum model length, as defined by the model's `config.max_position_embeddings` attribute.
    """

    def __init__(self, max_length: int, max_position_embeddings: Optional[int] = None):
        self.max_length = max_length
        self.max_position_embeddings = max_position_embeddings

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        cur_len = input_ids.shape[-1]
        is_done = cur_len >= self.max_length
        if self.max_position_embeddings is not None and not is_done and cur_len >= self.max_position_embeddings:
            logger.warning_once(
                "This is a friendly reminder - the current text generation call will exceed the model's predefined "
                f"maximum length ({self.max_position_embeddings}). Depending on the model, you may observe "
                "exceptions, performance degradation, or nothing at all."
            )
        return torch.full((input_ids.shape[0],), is_done, device=input_ids.device)


class MaxNewTokensCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the generated number of tokens exceeds `max_new_tokens`. Keep in
    mind for decoder-only type of transformers, this will **not** include the initial prompted tokens. This is very
    close to `MaxLengthCriteria` but ignores the number of initial tokens.

    Args:
        start_length (`int`):
            The number of initial tokens.
        max_new_tokens (`int`):
            The maximum number of tokens to generate.
    """

    def __init__(self, start_length: int, max_new_tokens: int):
        warnings.warn(
            "The class `MaxNewTokensCriteria` is deprecated. "
            f"Please use `MaxLengthCriteria(max_length={start_length + max_new_tokens})` "
            "with `max_length = start_length + max_new_tokens` instead.",
            FutureWarning,
        )
        self.start_length = start_length
        self.max_new_tokens = max_new_tokens
        self.max_length = start_length + max_new_tokens

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        is_done = input_ids.shape[-1] >= self.max_length
        return torch.full((input_ids.shape[0],), is_done, device=input_ids.device)


class MaxTimeCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generation exceeds some amount of time. By default, the
    time will start being counted when you initialize this function. You can override this by passing an
    `initial_time`.

    Args:
        max_time (`float`):
            The maximum allowed time in seconds for the generation.
        initial_time (`float`, *optional*, defaults to `time.time()`):
            The start of the generation allowed time.
    """

    def __init__(self, max_time: float, initial_timestamp: Optional[float] = None):
        self.max_time = max_time
        self.initial_timestamp = time.time() if initial_timestamp is None else initial_timestamp

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        is_done = time.time() - self.initial_timestamp > self.max_time
        return torch.full((input_ids.shape[0],), is_done, device=input_ids.device)


class StopStringCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever specific string sequences are encountered. It preprocesses
    the strings together with the tokenizer vocab to find positions where tokens can validly complete the stop strings.

    Args:
        tokenizer (`PreTrainedTokenizer`):
            The model's associated tokenizer (necessary to extract vocab and tokenize the termination sequences)
        stop_strings (`Union[str, List[str]]`):
            A list of strings that should end generation. If a string is passed, it will be treated like a
            list with a single element.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, stop_strings: Union[str, List[str]]):
        if isinstance(stop_strings, str):
            stop_strings = [stop_strings]

        self.vocab = tokenizer.get_vocab()
        self.token_list, self.tok_indices = tuple(self.vocab.keys()), tuple(self.vocab.values())
        self.stop_strings: Tuple[str, ...] = tuple(stop_strings)

        self.embedding_vec, self.max_valid_positions, self.max_valid_end_lens = _stop_string_create_embedding_vec(
            self.token_list, self.tok_indices, self.stop_strings
        )
        self.maximum_token_len = max([len(stop_string) for stop_string in self.stop_strings])
        self.num_stop_strings = len(self.stop_strings)
        self.target_lens = torch.tensor([len(stop_string) for stop_string in stop_strings], dtype=torch.int32)

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.Tensor:
        self.embedding_vec = self.embedding_vec.to(input_ids.device)
        self.target_lens = self.target_lens.to(input_ids.device)
        # The maximum length we need to consider is 1 token per character. Note that input_ids can also be
        # *shorter* than the global max, and the code below should be ready for that
        input_ids = input_ids[:, -self.maximum_token_len :]

        # Flip input_ids because we're only matching strings at the end of the generated sequence
        flipped_ids = torch.flip(input_ids, (1,))

        # Size of the vector of positions a single token can match
        max_valid_positions = self.max_valid_positions

        # The embedding vec contains the valid positions, end_lengths and total lengths for each token
        embedded = F.embedding(flipped_ids, self.embedding_vec)

        # Now we split the embedding vector. valid_positions is the positions in the stop string the token can fit
        valid_positions = embedded[:, 1:, : max_valid_positions * self.num_stop_strings].unflatten(
            -1, (self.num_stop_strings, -1)
        )
        # end_lengths is the number of characters from the string, counting from the end, that the token
        # contains. It can have multiple values if the same token can overlap different end lengths
        end_lengths = embedded[:, :1, max_valid_positions * self.num_stop_strings : -1].unflatten(
            -1, (self.num_stop_strings, -1)
        )
        # Lengths is the total length of each token. Unlike the others, it always has a single value
        lengths = embedded[:, 1:, None, -1:]  # Insert a dummy dimension for stop_strings even though lengths are const

        # Concatenate lengths onto each possible end_lengths value
        lengths = lengths.expand((-1, -1, end_lengths.shape[-2], end_lengths.shape[-1]))
        lengths_with_ends = torch.cat([end_lengths, lengths], dim=1)

        # cumsum() to get the number of matched characters in the stop string after each token
        cumsum = lengths_with_ends.cumsum(dim=1)  # B x maximum_token_len x num_stop_strings x max_valid_end_lens

        # The calculation above assumes that all tokens are in valid positions. Now we mask the ones that are not.
        # First, tokens match the start of the string if they have a positive value in the end_lengths vector
        initial_match = end_lengths > 0

        # Tokens continue the string if the cumsum() so far is one of the valid positions for that token
        # Note that we're actually tracking one cumsum() for for each possible end_length
        later_match = torch.any(cumsum[:, :-1, :, None] == valid_positions[:, :, :, :, None], axis=-2)

        # The match vector is a boolean vector that indicates which positions have valid tokens
        match = torch.cat([initial_match, later_match], dim=1)

        # Once a single position does not match, all positions following that position are masked
        mask = (~match).cumsum(dim=1, dtype=torch.int32)
        mask = mask == 0

        # The string is matched if we reached a cumsum equal to or greater than the length of the string
        # before hitting the mask
        string_matches = torch.amax(cumsum * mask, dim=(1, -1)) >= self.target_lens[None, :]

        # We return a per-sample vector that is True if any stop string is matched for that sample
        return torch.any(string_matches, dim=-1)


class StoppingCriteriaList(list):
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        is_done = torch.full((input_ids.shape[0],), False, device=input_ids.device)
        for criteria in self:
            is_done = is_done | criteria(input_ids, scores, **kwargs)
        return is_done

    @property
    def max_length(self) -> Optional[int]:
        for stopping_criterium in self:
            if isinstance(stopping_criterium, MaxLengthCriteria):
                return stopping_criterium.max_length
            elif isinstance(stopping_criterium, MaxNewTokensCriteria):
                return stopping_criterium.max_length
        return None


def validate_stopping_criteria(stopping_criteria: StoppingCriteriaList, max_length: int) -> StoppingCriteriaList:
    stopping_max_length = stopping_criteria.max_length
    new_stopping_criteria = deepcopy(stopping_criteria)
    if stopping_max_length is not None and stopping_max_length != max_length:
        warnings.warn("You set different `max_length` for stopping criteria and `max_length` parameter", UserWarning)
    elif stopping_max_length is None:
        new_stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
    return new_stopping_criteria


def _stop_string_get_matching_positions(
    token_list, tok_indices, stop_strings
) -> Tuple[Dict[str, Dict[str, List[int]]], Dict[str, Dict[str, List[int]]]]:
    """This function preprocesses stop strings and the tokenizer vocabulary to determine where tokens can
    validly appear in the stop strings. For each stop string, it returns a dictionary mapping tokens to a list of
    valid positions, as well as a dictionary mapping tokens to a list of possible overlap lengths at the
    end of the stop string."""

    def _cleanup_token(token: str) -> str:
        if token[0] in ["▁", "Ġ"]:
            token = " " + token[1:]
        elif token[0] == "##":
            token = token[2:]
        return token

    reversed_filtered_token_list = [_cleanup_token(token)[::-1] for token in token_list]
    token_valid_positions = {}
    token_end_overlaps = {}
    for stop_string in stop_strings:
        reversed_stop_string = stop_string[::-1]
        token_valid_positions[stop_string] = {}
        token_end_overlaps[stop_string] = {}
        for token, reversed_filtered_token, tok_idx in zip(token_list, reversed_filtered_token_list, tok_indices):
            matching_positions = []
            possible_end_lengths = []
            for i in range(1 - len(token), len(stop_string)):
                if i < 0:
                    tok = reversed_filtered_token[-i:]
                    i = 0
                else:
                    tok = reversed_filtered_token
                stop = reversed_stop_string[i : i + len(tok)]
                if tok.startswith(stop):
                    if i == 0:
                        possible_end_lengths.append(min(len(tok), len(stop)))
                    else:
                        matching_positions.append(i)

            if matching_positions:
                token_valid_positions[stop_string][tok_idx] = matching_positions
            if possible_end_lengths:
                token_end_overlaps[stop_string][tok_idx] = possible_end_lengths
    return token_valid_positions, token_end_overlaps


@lru_cache(8)
def _stop_string_create_embedding_vec(token_list, tok_indices, stop_strings) -> Dict[str, torch.tensor]:
    """
    This function builds an embedding matrix for each stop string, consisting of possible valid positions
    and possible end lengths for each token, and the total length of the token string. When tokens have
    fewer valid positions or end lengths than the maximum, we pad the vectors with -1.
    """
    token_valid_positions, token_end_overlaps = _stop_string_get_matching_positions(
        token_list, tok_indices, stop_strings
    )

    max_valid_positions = max(len(val) for positions in token_valid_positions.values() for val in positions.values())
    max_valid_end_lens = max(len(val) for positions in token_end_overlaps.values() for val in positions.values())
    vec_size = len(stop_strings) * (max_valid_positions + max_valid_end_lens) + 1
    gather_vec = np.full((len(token_list), vec_size), dtype=np.int32, fill_value=-1)

    for i, stop_string in enumerate(stop_strings):
        positions = token_valid_positions[stop_string]
        end_lens = token_end_overlaps[stop_string]

        # Since this is lots of very small assignments of lists, we build it with numpy rather
        # than torch for speed + simplicity, then convert to torch at the end
        for token_idx, valid_positions in positions.items():
            gather_vec[
                token_idx, max_valid_positions * i : max_valid_positions * i + len(valid_positions)
            ] = valid_positions
        for token_idx, possible_end_lens in end_lens.items():
            gather_vec[
                token_idx,
                max_valid_positions * len(stop_strings) + max_valid_end_lens * i : max_valid_positions
                * len(stop_strings)
                + max_valid_end_lens * i
                + len(possible_end_lens),
            ] = possible_end_lens
        for token, token_idx in zip(token_list, tok_indices):
            gather_vec[token_idx, -1] = len(token)

    gather_vec = torch.tensor(gather_vec, dtype=torch.int32)

    return gather_vec, max_valid_positions, max_valid_end_lens
