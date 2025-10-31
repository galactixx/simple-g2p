import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, TypeAlias

from sklearn.model_selection import train_test_split

from constants import DECODE, ENCODE, SEED
from utils import CMUDictType

Graphemes: TypeAlias = List[Tuple[int, str, List[List[str]]]]
RefMap: TypeAlias = Dict[int, List[List[int]]]


@dataclass(frozen=True)
class TokenConfig:
    # Encoder side
    encode_vocab: List[str]
    encode_char_to_id: Dict[str, int]

    # Decoder side
    decode_vocab: List[str]
    decode_char_to_id: Dict[str, int]
    decode_id_to_char: Dict[int, str]


@dataclass(frozen=True)
class PhonemePair:
    index: int
    grapheme: List[int]
    phoneme: List[int]


def build_ref_map(
    train_pairs: List[PhonemePair],
    val_pairs: List[PhonemePair],
    test_pairs: List[PhonemePair],
    graphemes: Graphemes,
) -> RefMap:
    ref_map: RefMap = dict()
    for phoneme_pair in train_pairs + val_pairs + test_pairs:
        idx = phoneme_pair.index
        phoneme_variants = graphemes[idx][2]
        if len(phoneme_variants) <= 1:
            continue
        if idx not in ref_map:
            ref_map[idx] = []
        ref_map[idx].append(phoneme_pair.phoneme)
    return ref_map


def generate_pairs(graphemes: Graphemes, config: TokenConfig) -> List[PhonemePair]:
    phoneme_pairs: List[PhonemePair] = []
    for index, (index, word, phonemes) in enumerate(graphemes):
        word_ids = [config.encode_char_to_id[char] for char in word]
        for phoneme in phonemes:
            phoneme_ids = [config.decode_char_to_id[char] for char in phoneme]
            pair = PhonemePair(index=index, grapheme=word_ids, phoneme=phoneme_ids)
            phoneme_pairs.append(pair)
    return phoneme_pairs


def get_id_to_char(char_to_id: Dict[str, int]) -> Dict[int, str]:
    return {i: c for c, i in char_to_id.items()}


def get_char_to_id(vocab: List[str]) -> Dict[str, int]:
    return {c: i for i, c in enumerate(vocab)}


def parse_cmu_dict(cmu_dict: CMUDictType) -> Tuple[Graphemes, TokenConfig]:
    encode_vocab: Set[str] = set()
    decode_vocab: Set[str] = set()
    graphemes: Graphemes = []
    index = 0
    for word, phonemes in cmu_dict.items():
        word = unicodedata.normalize("NFC", word.strip())

        # Skip non-alphabetic entries
        if not re.match(r"^[a-z'-]+$", word):
            continue

        phonemes_clean = []
        for phoneme in phonemes:
            new_phoneme = (
                ["<BOS>"] + [re.sub(r"\d$", "", p) for p in phoneme] + ["<EOS>"]
            )

            for p in new_phoneme:
                decode_vocab.add(p)

            phonemes_clean.append(new_phoneme)

        cmu_tuple = (index, word, phonemes_clean)
        graphemes.append(cmu_tuple)
        for char in word:
            if char not in encode_vocab:
                encode_vocab.add(char)
        index += 1

    encode_vocab = encode_vocab.union(ENCODE)
    encode_vocab = sorted(list(encode_vocab))
    encode_char_to_id = get_char_to_id(vocab=encode_vocab)

    decode_vocab = decode_vocab.union(DECODE)
    decode_vocab = sorted(list(decode_vocab))
    decode_char_to_id = get_char_to_id(vocab=decode_vocab)
    decode_id_to_char = get_id_to_char(char_to_id=decode_char_to_id)

    return graphemes, TokenConfig(
        encode_vocab=encode_vocab,
        encode_char_to_id=encode_char_to_id,
        decode_vocab=decode_vocab,
        decode_char_to_id=decode_char_to_id,
        decode_id_to_char=decode_id_to_char,
    )


def split_and_generate_pairs(
    graphemes: Graphemes,
    config: TokenConfig,
    test_size: float = 0.2,
    val_size: float = 0.5,
) -> Tuple[List[PhonemePair], List[PhonemePair], List[PhonemePair]]:
    train_g, test_g = train_test_split(
        graphemes, test_size=test_size, random_state=SEED
    )
    test_g, val_g = train_test_split(test_g, test_size=val_size, random_state=SEED)

    train_pairs = generate_pairs(train_g, config)
    val_pairs = generate_pairs(val_g, config)
    test_pairs = generate_pairs(test_g, config)

    return train_pairs, val_pairs, test_pairs
